"""Обвязка сквозных сценариев: живые PostgreSQL, RabbitMQ и MinIO.

Сервис поднимается своим настоящим композиционным корнем: подменённая здесь
сборка проверяла бы не тот сервис, который уезжает в образ.

Изоляция — `TRUNCATE` между сценариями, а не откатываемая транзакция: в
сквозном пути участвуют несколько соединений, и внешняя транзакция их не
накрывает.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from alembic import command
from faststream.rabbit import RabbitBroker, RabbitQueue
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from document_worker.bootstrap.composition import build_services
from document_worker.bootstrap.outbox import OutboxRelay, running
from document_worker.domain.value_objects.storage import ObjectRef
from document_worker.infrastructure.config.settings import AppSettings
from document_worker.infrastructure.messaging.topology import (
    AUDIT_QUEUE,
    COMMANDS_EXCHANGE,
    DLQ_QUEUE,
    RK_PROCESS_REQUESTED,
)
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.presentation.messaging.contracts.commands import (
    DOCUMENT_KEY_PREFIX,
)
from tests.conftest import (
    _create_database,
    _drop_database,
    _dsn_for,
    alembic_config,
)
from tests.factories import make_document
from tests.fakes.network import BreakableLink

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

    from minio import Minio
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.bootstrap.composition import Services
    from document_worker.domain.entities.document import Document
    from document_worker.infrastructure.storage.s3_object_storage import S3Config

# Распознавание страницы занимает секунды, а реле публикует раз в полсекунды.
EVENT_TIMEOUT_S = 120.0
POLL_S = 0.2
# Столько ждёт утверждение «повтор ничего не изменил»: обработка повторной
# доставки укладывается в миллисекунды, запас — на планировщик.
SETTLE_S = 5.0

TABLES = (
    "outbox_events",
    "processed_messages",
    "document_chunks",
    "document_illegible_spans",
    "document_pages",
    "processing_jobs",
    "documents",
)

E2E_BUCKET = "e2e-documents"


@dataclass(frozen=True, slots=True)
class Harness:
    """Всё, чем сквозной сценарий разговаривает с поднятым сервисом."""

    services: Services
    probe: RabbitBroker
    minio: Minio
    bucket: str
    sessions: async_sessionmaker[AsyncSession]

    def put_object(self, ref: ObjectRef, payload: bytes) -> None:
        """Кладёт исходный файл в хранилище."""
        self.minio.put_object(
            ref.bucket, ref.key, io.BytesIO(payload), length=len(payload)
        )

    async def store(self, document: Document) -> Document:
        """Строку документа создаёт сервис приёма файлов, здесь — сценарий."""
        # Ключ обязан лежать в префиксе своего документа: этого требует
        # входящий контракт, и только он.
        placed = replace(
            document,
            source=replace(
                document.source,
                ref=ObjectRef(
                    bucket=self.bucket,
                    key=f"{DOCUMENT_KEY_PREFIX}/{document.id}/source.pdf",
                ),
            ),
        )
        async with self.sessions() as session:
            session.add(document_to_row(placed))
            await session.commit()
        return placed

    async def request_processing(self, document: Document) -> str:
        """Публикует команду обработки документа."""
        event_id = str(uuid.uuid4())
        await self.probe.publish(
            {
                "event_id": event_id,
                "document_id": str(document.id),
                "object_key": document.source.ref.key,
                "bucket": document.source.ref.bucket,
                "mime_type": document.source.mime_type.value,
                "correlation_id": str(document.correlation_id),
                "occurred_at": datetime.now(UTC).isoformat(),
            },
            exchange=COMMANDS_EXCHANGE,
            routing_key=RK_PROCESS_REQUESTED,
        )
        return event_id

    async def wait_for_event(self, document: Document) -> dict[str, Any]:
        """Дожидается события об этом документе в очереди аудита."""
        return await self._wait(AUDIT_QUEUE, str(document.id))

    async def wait_for_dlq(self, document: Document) -> dict[str, Any]:
        """Дожидается копии сообщения в очереди разбора."""
        return await self._wait(DLQ_QUEUE, str(document.id))

    async def _wait(self, queue: str, document_id: str) -> dict[str, Any]:
        # Очередь уже объявлена топологией сервиса: повторное объявление с
        # другими аргументами разошлось бы с ней и упало.
        declared = await self.probe.declare_queue(
            RabbitQueue(queue, durable=True, declare=False)
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + EVENT_TIMEOUT_S
        while loop.time() < deadline:
            message = await declared.get(no_ack=True, fail=False)
            if message is not None:
                payload: dict[str, Any] = json.loads(message.body)
                if str(payload.get("document_id")) == document_id:
                    return payload
                continue
            await asyncio.sleep(POLL_S)
        pytest.fail(f"события о документе {document_id} нет в очереди {queue}")

    async def wait_for_release(self, document: Document) -> None:
        """Дожидается, пока провалившаяся попытка отпустит захват сообщения.

        Отпущенный захват — единственный след, который проваленная попытка
        оставляет в базе: по нему сценарий и узнаёт, что она уже кончилась.
        Сравниваются два поля одной строки, а не время с часами базы: захват
        отпускают, приравнивая срок лиза к отметке изменения.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + EVENT_TIMEOUT_S
        while loop.time() < deadline:
            released = await self.rows(
                "SELECT 1 FROM processed_messages"
                " WHERE document_id = :id AND lease_expires_at <= updated_at",
                id=document.id.value,
            )
            if released:
                return
            await asyncio.sleep(POLL_S)
        pytest.fail(f"захват сообщения документа {document.id} не отпущен")

    async def settle(self) -> None:
        """Даёт повторной доставке дойти до конца.

        Утверждение «ничего не изменилось» требует границы ожидания: повтор
        события не публикует и строки в барьере идемпотентности не создаёт,
        поэтому ждать нечего, кроме самого факта обработки сообщения.
        """
        await asyncio.sleep(SETTLE_S)

    async def rows(self, statement: str, **params: object) -> list[Any]:
        """Читает строки прямо из базы, минуя репозитории."""
        async with self.services.engine.connect() as connection:
            return list(await connection.execute(text(statement), params))


@pytest.fixture(scope="session")
def e2e_bucket(minio_client: Minio) -> str:
    """Бакет исходных файлов сквозных сценариев."""
    if not minio_client.bucket_exists(E2E_BUCKET):
        minio_client.make_bucket(E2E_BUCKET)
    return E2E_BUCKET


@pytest.fixture(scope="session")
def e2e_vhost(management: Any) -> Iterator[str]:
    """Свой vhost на весь прогон сквозных сценариев."""
    name = f"e2e-{uuid.uuid4().hex[:12]}"
    management.create_vhost(name)
    try:
        yield name
    finally:
        management.delete_vhost(name)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def e2e_dsn(base_dsn: str) -> AsyncIterator[str]:
    """Своя база со схемой: сквозной сценарий транзакцией не накрыть."""
    name = f"docworker_e2e_{uuid.uuid4().hex[:8]}"
    await _create_database(base_dsn, name)
    dsn = _dsn_for(base_dsn, name)
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")
    try:
        yield dsn
    finally:
        await _drop_database(base_dsn, name)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def storage_link(s3_config: S3Config) -> AsyncIterator[BreakableLink]:
    """Связь сервиса с хранилищем, которую сценарий может порвать."""
    target = urlsplit(s3_config.endpoint_url)
    link = BreakableLink(target.hostname or "127.0.0.1", target.port or 80)
    await link.start()
    try:
        yield link
    finally:
        await link.stop()


@pytest.fixture(scope="session")
def e2e_settings(  # noqa: PLR0913, PLR0917 — настройки собираются из всех этих частей
    e2e_dsn: str,
    rabbitmq_url: str,
    e2e_vhost: str,
    s3_config: S3Config,
    storage_link: BreakableLink,
    e2e_bucket: str,
    model_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> AppSettings:
    """Настройки сервиса, направленные на поднятые контейнеры."""
    return AppSettings.model_validate(
        {
            "database": {"dsn": e2e_dsn},
            "rabbit": {
                "url": f"{rabbitmq_url.rstrip('/')}/{e2e_vhost}",
                "declare_audit_queue": True,
            },
            "s3": {
                "endpoint_url": storage_link.url,
                "access_key": s3_config.access_key,
                "secret_key": s3_config.secret_key,
                "default_bucket": e2e_bucket,
            },
            "processing": {"temp_dir": str(tmp_path_factory.mktemp("work"))},
            "ocr": {"model_dir": str(model_dir)},
        }
    )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def harness(
    e2e_settings: AppSettings,
    minio_client: Minio,
    e2e_bucket: str,
) -> AsyncIterator[Harness]:
    """Поднятый сервис вместе с реле публикации и отдельным подключением."""
    async with build_services(e2e_settings) as services:
        relay = OutboxRelay(
            publish=services.publish_outbox,
            config=e2e_settings.processing_config().outbox,
        )
        probe = RabbitBroker(e2e_settings.rabbit.url.get_secret_value())
        await probe.connect()
        async with running(relay):
            await services.broker.start()
            try:
                yield Harness(
                    services=services,
                    probe=probe,
                    minio=minio_client,
                    bucket=e2e_bucket,
                    sessions=async_sessionmaker(
                        bind=services.engine, expire_on_commit=False
                    ),
                )
            finally:
                await probe.stop()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_database(harness: Harness) -> AsyncIterator[None]:
    """Чистит таблицы после каждого сценария."""
    yield
    async with harness.services.engine.begin() as connection:
        await connection.execute(
            text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
        )


@pytest_asyncio.fixture(loop_scope="session")
async def document(harness: Harness, clean_database: None) -> Document:  # noqa: ARG001 — чистка идёт после сценария
    """Документ, ожидающий обработки, вместе со строкой в базе."""
    return await harness.store(make_document())
