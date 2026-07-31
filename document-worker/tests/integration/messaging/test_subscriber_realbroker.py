"""Подписчик и лестница повторов на живом RabbitMQ.

Проверять их на in-memory брокере нельзя: там нет ни TTL, ни dead-lettering,
ни подтверждений — то есть ровно того, чем занят этот модуль.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from document_worker.application.dto.commands import ProcessDocumentCommand
from document_worker.application.dto.results import ProcessDocumentResult
from document_worker.application.errors import (
    ConcurrentProcessingError,
    CorruptedDocumentError,
    StorageUnavailableError,
)
from document_worker.domain.value_objects.enums import DocumentStatus
from document_worker.domain.value_objects.identifiers import (
    CorrelationId,
    DocumentId,
    EventId,
)
from document_worker.domain.value_objects.storage import MimeType, ObjectRef
from document_worker.infrastructure.messaging.broker import build_broker
from document_worker.infrastructure.messaging.declare import declare_topology
from document_worker.infrastructure.messaging.retry_publisher import RetryPublisher
from document_worker.infrastructure.messaging.topology import (
    DLQ_QUEUE,
    PROCESS_REQUESTED_QUEUE,
    RK_PROCESS_REQUESTED,
    RK_RETRY_BASE,
    build_topology,
    retry_queue_name,
)
from document_worker.presentation.messaging.headers import (
    HEADER_ATTEMPT,
    HEADER_ATTEMPTS_EXHAUSTED,
    HEADER_ERROR_CODE,
)
from document_worker.presentation.messaging.subscribers.process_document import (
    build_process_document_router,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from faststream.rabbit import RabbitBroker, RabbitExchange

    from document_worker.infrastructure.messaging.topology import Topology
    from tests.integration.messaging.conftest import Management

pytestmark = pytest.mark.integration

CONSUMER_TIMEOUT_MS = 7_200_000
DEFAULT_BUCKET = "documents"
# Лестница из реальных интервалов растянула бы прогон на 42 минуты.
FAST_LADDER = (("200ms", 200), ("500ms", 500), ("3s", 3_000))
DOCUMENT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


@dataclass
class StubProcessor:
    """Прикладной слой, поведение которого задаёт тест."""

    result: ProcessDocumentResult | None = None
    error: Exception | None = None
    commands: list[ProcessDocumentCommand] = field(default_factory=list)

    async def execute(self, command: ProcessDocumentCommand) -> ProcessDocumentResult:
        """Возвращает заданный исход или поднимает заданную ошибку."""
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return self.result or _processed(command)


def _processed(command: ProcessDocumentCommand) -> ProcessDocumentResult:
    return ProcessDocumentResult(
        document_id=command.document_id,
        status=DocumentStatus.PROCESSED,
        pages_total=1,
        chunks_total=0,
    )


def _failed(command: ProcessDocumentCommand) -> ProcessDocumentResult:
    return ProcessDocumentResult(
        document_id=command.document_id,
        status=DocumentStatus.FAILED,
        pages_total=0,
        chunks_total=0,
    )


def _body(**overrides: object) -> bytes:
    payload: dict[str, object] = {
        "event_id": str(uuid.uuid4()),
        "document_id": str(DOCUMENT_ID),
        "object_key": f"documents/{DOCUMENT_ID}/source.pdf",
        "mime_type": "application/pdf",
        "occurred_at": datetime.now(UTC).isoformat(),
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


@dataclass(frozen=True, slots=True)
class Consumer:
    """Запущенный подписчик вместе с тем, чем управляет тест."""

    broker: RabbitBroker
    processor: StubProcessor
    vhost: str
    topology: Topology


@pytest.fixture
def processor() -> StubProcessor:
    return StubProcessor()


@pytest.fixture
async def consumer(
    rabbitmq_url: str,
    isolated_vhost: str,
    processor: StubProcessor,
) -> AsyncIterator[Consumer]:
    """Брокер с объявленной топологией и запущенным подписчиком."""
    connected = build_broker(f"{rabbitmq_url}{isolated_vhost}")
    topology = build_topology(
        consumer_timeout_ms=CONSUMER_TIMEOUT_MS, retry_ladder=FAST_LADDER
    )
    connected.include_router(
        build_process_document_router(
            queue=topology.process_requested,
            exchange=topology.commands,
            processor=processor,
            retrier=RetryPublisher(connected, topology),
            default_bucket=DEFAULT_BUCKET,
            max_retries=len(FAST_LADDER),
        )
    )
    await connected.connect()
    await declare_topology(connected, topology)
    await connected.start()
    try:
        yield Consumer(
            broker=connected,
            processor=processor,
            vhost=isolated_vhost,
            topology=topology,
        )
    finally:
        await connected.stop()


async def _publish(consumer: Consumer, body: bytes, **kwargs: object) -> None:
    await consumer.broker.publish(
        body,
        exchange=consumer.topology.commands,
        routing_key=RK_PROCESS_REQUESTED,
        **kwargs,  # type: ignore[arg-type]
    )


async def _wait_until(predicate: Callable[[], bool]) -> bool:
    for _ in range(100):
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


async def _wait_for(
    management: Management,
    consumer: Consumer,
    queue: str,
    expected: int,
) -> int:
    for _ in range(100):
        count = len(management.fetch(queue, vhost=consumer.vhost))
        if count >= expected:
            return count
        await asyncio.sleep(0.1)
    return len(management.fetch(queue, vhost=consumer.vhost))


async def _settled(management: Management, consumer: Consumer) -> None:
    """Ждёт, пока рабочая очередь опустеет и не останется неподтверждённых."""
    for _ in range(80):
        queue = management.queue(PROCESS_REQUESTED_QUEUE, vhost=consumer.vhost)
        if not queue.get("messages", 0) and not queue.get("messages_unacknowledged", 0):
            return
        await asyncio.sleep(0.1)


async def test_successful_processing_acknowledges_the_message(
    consumer: Consumer,
    management: Management,
) -> None:
    await _publish(consumer, _body())

    assert await _wait_until(lambda: bool(consumer.processor.commands))
    await _settled(management, consumer)
    assert management.fetch(DLQ_QUEUE, vhost=consumer.vhost) == []


async def test_message_becomes_a_command_with_its_attempt(
    consumer: Consumer,
) -> None:
    await _publish(consumer, _body(), headers={HEADER_ATTEMPT: 2})

    assert await _wait_until(lambda: bool(consumer.processor.commands))

    command = consumer.processor.commands[0]
    assert command.attempt == 3
    assert command.max_attempts == len(FAST_LADDER) + 1
    assert command.object_ref.bucket == DEFAULT_BUCKET


async def test_transient_error_comes_back_through_the_ladder(
    consumer: Consumer,
) -> None:
    # Копия уходит на ступень задержки, там ждёт TTL и возвращается в рабочую
    # очередь: наблюдаем это по повторному вызову с выросшей попыткой.
    consumer.processor.error = StorageUnavailableError("хранилище недоступно")

    await _publish(consumer, _body())

    assert await _wait_until(lambda: len(consumer.processor.commands) >= 2)
    assert consumer.processor.commands[1].attempt == 2


async def test_retry_survives_because_the_delay_lives_in_the_broker(
    consumer: Consumer,
) -> None:
    # Задержка хранится в очереди, а не в памяти обработчика: иначе она
    # исчезала бы при первом же перезапуске воркера.
    consumer.processor.error = StorageUnavailableError("хранилище недоступно")
    await _publish(consumer, _body())
    await _wait_until(lambda: bool(consumer.processor.commands))

    consumer.processor.error = None

    assert await _wait_until(lambda: len(consumer.processor.commands) >= 2)


async def test_attempt_grows_with_every_retry(consumer: Consumer) -> None:
    consumer.processor.error = StorageUnavailableError("хранилище недоступно")

    await _publish(consumer, _body())

    assert await _wait_until(lambda: len(consumer.processor.commands) >= 3)
    attempts = [command.attempt for command in consumer.processor.commands]
    assert attempts[:3] == [1, 2, 3]


async def test_copy_keeps_the_body_byte_for_byte(
    consumer: Consumer,
    management: Management,
) -> None:
    # Пересериализованная модель могла бы незаметно измениться между версиями
    # библиотеки, и повторная валидация дала бы другой результат.
    consumer.processor.error = StorageUnavailableError("хранилище недоступно")
    body = _body()

    await _publish(consumer, body, headers={HEADER_ATTEMPT: len(FAST_LADDER)})

    await _wait_for(management, consumer, DLQ_QUEUE, 1)
    assert management.peek(DLQ_QUEUE, vhost=consumer.vhost)["payload"].encode() == body


async def test_copy_drops_broker_death_headers(
    consumer: Consumer,
    management: Management,
) -> None:
    # Скопированный x-death уехал бы в копию обычным заголовком и врал бы про
    # число возвратов: брокер его при обычной публикации не создаёт.
    consumer.processor.error = StorageUnavailableError("хранилище недоступно")

    await _publish(
        consumer,
        _body(),
        headers={"x-death": "заглушка", HEADER_ATTEMPT: len(FAST_LADDER)},
    )

    await _wait_for(management, consumer, DLQ_QUEUE, 1)
    headers = management.peek(DLQ_QUEUE, vhost=consumer.vhost)["properties"]["headers"]
    assert "x-death" not in headers


async def test_exhausted_ladder_sends_the_message_to_the_dlq(
    consumer: Consumer,
    management: Management,
) -> None:
    consumer.processor.error = StorageUnavailableError("хранилище недоступно")

    await _publish(consumer, _body(), headers={HEADER_ATTEMPT: len(FAST_LADDER)})

    assert await _wait_for(management, consumer, DLQ_QUEUE, 1) == 1
    headers = management.peek(DLQ_QUEUE, vhost=consumer.vhost)["properties"]["headers"]
    assert headers[HEADER_ATTEMPTS_EXHAUSTED] is True


async def test_permanent_error_goes_to_the_dlq_without_retry(
    consumer: Consumer,
    management: Management,
) -> None:
    consumer.processor.error = CorruptedDocumentError("файл не читается")

    await _publish(consumer, _body())

    assert await _wait_for(management, consumer, DLQ_QUEUE, 1) == 1
    assert (
        management.fetch(retry_queue_name(FAST_LADDER[0][0]), vhost=consumer.vhost)
        == []
    )
    headers = management.peek(DLQ_QUEUE, vhost=consumer.vhost)["properties"]["headers"]
    assert headers[HEADER_ERROR_CODE] == CorruptedDocumentError.code


async def test_terminal_failure_of_the_use_case_goes_to_the_dlq(
    consumer: Consumer,
    management: Management,
) -> None:
    # Отказ уже зафиксирован транзакцией T4f: сообщение подтверждается, а копия
    # ложится в DLQ для разбора.
    consumer.processor.result = _failed(_command_stub())

    await _publish(consumer, _body())

    assert await _wait_for(management, consumer, DLQ_QUEUE, 1) == 1


async def test_concurrent_processing_consumes_an_attempt(
    consumer: Consumer,
    management: Management,
) -> None:
    # Иначе живой лиз зависшего воркера гонял бы сообщение по первой ступени
    # без предела: сообщение вечно живо, работа вечно не делается.
    consumer.processor.error = ConcurrentProcessingError("документ занят")

    await _publish(consumer, _body())

    assert await _wait_for(management, consumer, DLQ_QUEUE, 1) == 1
    assert len(consumer.processor.commands) == len(FAST_LADDER) + 1


async def test_unreadable_body_is_rejected_without_reaching_the_use_case(
    consumer: Consumer,
    management: Management,
) -> None:
    # Тела мы не понимаем, обогащать нечем: reject, брокерский путь в DLQ и ни
    # одного обращения к базе.
    await _publish(consumer, b"{ not json")

    assert await _wait_for(management, consumer, DLQ_QUEUE, 1) == 1
    assert consumer.processor.commands == []


async def test_message_failing_the_contract_is_rejected(
    consumer: Consumer,
    management: Management,
) -> None:
    await _publish(consumer, _body(object_key="../../etc/passwd"))

    assert await _wait_for(management, consumer, DLQ_QUEUE, 1) == 1
    assert consumer.processor.commands == []


async def test_short_delay_is_not_blocked_by_a_long_one(
    consumer: Consumer,
    management: Management,
) -> None:
    # Ради этого ступеней пять, а не одна очередь с TTL на сообщении: там
    # пятисекундная задержка за получасовой превратилась бы в получасовую.
    slow = retry_queue_name(FAST_LADDER[2][0])
    fast = retry_queue_name(FAST_LADDER[0][0])
    await consumer.broker.publish(
        b"slow",
        exchange=_retry_exchange(consumer),
        routing_key=f"{RK_RETRY_BASE}.{FAST_LADDER[2][0]}",
    )
    assert await _wait_for(management, consumer, slow, 1) == 1
    await consumer.broker.publish(
        b"fast",
        exchange=_retry_exchange(consumer),
        routing_key=f"{RK_RETRY_BASE}.{FAST_LADDER[0][0]}",
    )

    await asyncio.sleep(0.6)

    assert management.fetch(fast, vhost=consumer.vhost) == []
    assert len(management.fetch(slow, vhost=consumer.vhost)) == 1
    assert consumer.processor.commands == []


def _retry_exchange(consumer: Consumer) -> RabbitExchange:
    return consumer.topology.retry


def _command_stub() -> ProcessDocumentCommand:
    return ProcessDocumentCommand(
        event_id=EventId(uuid.uuid4()),
        document_id=DocumentId(DOCUMENT_ID),
        correlation_id=CorrelationId(str(uuid.uuid4())),
        object_ref=ObjectRef(
            bucket=DEFAULT_BUCKET, key=f"documents/{DOCUMENT_ID}/s.pdf"
        ),
        mime_type=MimeType(MimeType.PDF),
        occurred_at=datetime.now(UTC),
    )
