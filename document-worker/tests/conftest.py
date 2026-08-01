"""Контейнеры на весь прогон: PostgreSQL, RabbitMQ, MinIO и модели.

Живут в корне, а не в conftest интеграции: conftest каталога виден только
своему поддереву, а те же контейнеры нужны и сквозным сценариям.

Аргументы очередей проверяются только настоящим брокером: in-memory брокер их
не хранит, а AMQP их не отдаёт — passive-declare возвращает лишь имя и
счётчики. Поэтому рядом поднимается management API.
"""

from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pytest
import pytest_asyncio
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.minio import MinioContainer
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from document_worker.infrastructure.cpu.executor import CpuPool
from document_worker.infrastructure.ocr.model_registry import (
    download_missing,
    model_dir_from_env,
    verify,
)
from document_worker.infrastructure.storage.s3_object_storage import S3Config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from minio import Minio

POSTGRES_IMAGE = "postgres:18-alpine"


SERVICE_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_IMAGE = "postgres:18-alpine"


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Поднимает PostgreSQL один раз на весь прогон."""
    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
def base_dsn(postgres_container: PostgresContainer) -> str:
    """DSN административной базы контейнера."""
    return str(postgres_container.get_connection_url())


def _dsn_for(base: str, database: str) -> str:
    head, _, _ = base.rpartition("/")
    return f"{head}/{database}"


def alembic_config(dsn: str) -> Config:
    """Конфигурация Alembic, направленная на указанную базу."""
    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(SERVICE_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", dsn)
    return config


async def _create_database(base: str, name: str) -> None:
    engine = create_async_engine(base, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        await engine.dispose()


async def _drop_database(base: str, name: str) -> None:
    engine = create_async_engine(base, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            )
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def cpu_pool() -> AsyncIterator[CpuPool]:
    """Пул процессов под синхронные вызовы PDF-библиотек.

    Контекст `spawn` поднимает интерпретатор заново, поэтому пул один на весь
    прогон: иначе каждый тест платил бы за старт процесса больше, чем за разбор.
    """
    async with CpuPool(max_workers=2) as pool:
        yield pool


RABBITMQ_IMAGE = "rabbitmq:4.2-management-alpine"
AMQP_PORT = 5672
MANAGEMENT_PORT = 15672
DEFAULT_VHOST = "/"
USER = "guest"
PASSWORD = "guest"  # noqa: S105 — учётные данные образа по умолчанию
READY_LOG = r"Server startup complete"


@dataclass(frozen=True, slots=True)
class Management:
    """Тонкий клиент management API: только то, чего не отдаёт AMQP."""

    base_url: str
    auth: str
    user: str

    def queue(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Описание очереди так, как его хранит брокер."""
        described: dict[str, Any] = self._request(
            "GET", f"queues/{_q(vhost)}/{_q(name)}"
        )
        return described

    def arguments_of(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Аргументы очереди."""
        return dict(self.queue(name, vhost=vhost).get("arguments", {}))

    def message_count(self, name: str, *, vhost: str = DEFAULT_VHOST) -> int:
        """Сколько сообщений лежит в очереди по данным статистики."""
        return int(self.queue(name, vhost=vhost).get("messages", 0))

    def fetch(
        self,
        name: str,
        *,
        count: int = 10,
        vhost: str = DEFAULT_VHOST,
    ) -> list[dict[str, Any]]:
        """Читает сообщения очереди, возвращая их обратно.

        Счётчик из статистики обновляется с задержкой, особенно у quorum-очередей,
        а чтение показывает содержимое очереди прямо сейчас.
        """
        messages = self._request(
            "POST",
            f"queues/{_q(vhost)}/{_q(name)}/get",
            body={
                "count": count,
                "ackmode": "ack_requeue_true",
                "encoding": "auto",
                "truncate": 50_000,
            },
        )
        return list(messages or [])

    def peek(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Первое сообщение очереди, не снимая его оттуда."""
        return self.fetch(name, count=1, vhost=vhost)[0]

    def create_vhost(self, name: str) -> None:
        """Создаёт vhost и даёт на него права пользователю теста."""
        self._request("PUT", f"vhosts/{_q(name)}", body={})
        self._request(
            "PUT",
            f"permissions/{_q(name)}/{_q(self.user)}",
            body={"configure": ".*", "write": ".*", "read": ".*"},
        )

    def delete_vhost(self, name: str) -> None:
        """Удаляет vhost вместе со всем содержимым."""
        self._request("DELETE", f"vhosts/{_q(name)}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> Any:
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(  # noqa: S310 — адрес контейнера теста
            f"{self.base_url}/api/{path}",
            method=method,
            data=payload,
            headers={
                "Authorization": f"Basic {self.auth}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            raw = response.read()
        return json.loads(raw) if raw else None


def _q(value: str) -> str:
    return quote(value, safe="")


@pytest.fixture(scope="session")
def rabbitmq_container() -> Iterator[DockerContainer]:
    """Поднимает RabbitMQ один раз на весь прогон.

    Готовность определяется по журналу, а не пробным подключением: пробник
    готового контейнера иногда попадает на порт управления и принимает ответ
    HTTP за несовместимую версию протокола.
    """
    container = (
        DockerContainer(RABBITMQ_IMAGE)
        .with_exposed_ports(AMQP_PORT, MANAGEMENT_PORT)
        .with_env("RABBITMQ_DEFAULT_USER", USER)
        .with_env("RABBITMQ_DEFAULT_PASS", PASSWORD)
        .waiting_for(LogMessageWaitStrategy(READY_LOG))
    )
    with container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_url(rabbitmq_container: DockerContainer) -> str:
    """Базовый URL подключения; vhost дописывает тот, кому он нужен."""
    host = rabbitmq_container.get_container_host_ip()
    port = rabbitmq_container.get_exposed_port(AMQP_PORT)
    return f"amqp://{USER}:{PASSWORD}@{host}:{port}/"


@pytest.fixture(scope="session")
def management(rabbitmq_container: DockerContainer) -> Management:
    """Доступ к management API того же контейнера."""
    host = rabbitmq_container.get_container_host_ip()
    port = rabbitmq_container.get_exposed_port(MANAGEMENT_PORT)
    return Management(
        base_url=f"http://{host}:{port}",
        auth=b64encode(f"{USER}:{PASSWORD}".encode()).decode(),
        user=USER,
    )


@pytest.fixture(scope="session")
def minio_container() -> Iterator[MinioContainer]:
    """Поднимает MinIO один раз на весь прогон."""
    with MinioContainer() as container:
        yield container


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Каталог с моделями распознавания; недостающие докачиваются один раз."""
    directory = model_dir_from_env()
    download_missing(directory)
    verify(directory)
    return directory


@pytest.fixture(scope="session")
def s3_config(minio_container: MinioContainer) -> S3Config:
    """Параметры подключения к поднятому MinIO."""
    config = minio_container.get_config()
    return S3Config(
        endpoint_url=f"http://{config['endpoint']}",
        region="us-east-1",
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        connect_timeout_s=5.0,
        read_timeout_s=30.0,
        max_attempts=1,
    )


@pytest.fixture(scope="session")
def minio_client(minio_container: MinioContainer) -> Iterator[Minio]:
    """Один клиент на прогон.

    `get_client()` делает нового на каждый вызов, и его пул соединений потом
    закрывает сборщик мусора — с предупреждением, которое падает как ошибка.
    """
    client = minio_container.get_client()
    try:
        yield client
    finally:
        client._http.clear()
