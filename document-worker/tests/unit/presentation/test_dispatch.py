"""Что подписчик делает с сообщением при каждом исходе.

Каждая ветка обязана завершиться подтверждением, отклонением или сознательным
пробросом: неподтверждённое сообщение висит в доставленных до закрытия канала,
а канал закрывается по таймауту вместе со всей пачкой.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from document_worker.application.dto.results import ProcessDocumentResult
from document_worker.application.errors import (
    ConcurrentProcessingError,
    CorruptedDocumentError,
)
from document_worker.domain.value_objects.enums import DocumentStatus
from document_worker.domain.value_objects.identifiers import DocumentId
from document_worker.presentation.messaging.contracts.commands import (
    ProcessDocumentRequestedV1,
)
from document_worker.presentation.messaging.headers import (
    HEADER_ATTEMPT,
    HEADER_ATTEMPTS_EXHAUSTED,
    HEADER_ERROR_CODE,
)
from document_worker.presentation.messaging.ports import (
    DocumentProcessor,
    MessageRetrier,
)
from document_worker.presentation.messaging.subscribers.process_document import dispatch

if TYPE_CHECKING:
    from document_worker.application.dto.commands import ProcessDocumentCommand

pytestmark = pytest.mark.unit

DOCUMENT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
DEFAULT_BUCKET = "documents"
MAX_RETRIES = 3
BODY = b'{"raw": true}'


@dataclass
class StubProcessor:
    result: ProcessDocumentResult | None = None
    error: BaseException | None = None
    commands: list[ProcessDocumentCommand] = field(default_factory=list)

    async def execute(self, command: ProcessDocumentCommand) -> ProcessDocumentResult:
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return self.result or ProcessDocumentResult(
            document_id=DocumentId(DOCUMENT_ID),
            status=DocumentStatus.PROCESSED,
            pages_total=1,
            chunks_total=0,
        )


@dataclass
class SpyRetrier:
    scheduled: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    dead_lettered: list[dict[str, Any]] = field(default_factory=list)
    bodies: list[bytes] = field(default_factory=list)

    async def schedule(
        self,
        body: bytes,
        headers: dict[str, Any],
        *,
        attempt: int,
    ) -> None:
        self.bodies.append(body)
        self.scheduled.append((attempt, headers))

    async def send_to_dlq(self, body: bytes, headers: dict[str, Any]) -> None:
        self.bodies.append(body)
        self.dead_lettered.append(headers)


@dataclass
class FakeMessage:
    headers: dict[str, Any] = field(default_factory=dict)
    body: bytes = BODY
    acked: int = 0
    rejected: int = 0

    async def ack(self) -> None:
        self.acked += 1

    async def reject(self) -> None:
        self.rejected += 1


def _payload() -> ProcessDocumentRequestedV1:
    return ProcessDocumentRequestedV1.model_validate(
        {
            "event_id": str(uuid.uuid4()),
            "document_id": str(DOCUMENT_ID),
            "object_key": f"documents/{DOCUMENT_ID}/source.pdf",
            "mime_type": "application/pdf",
            "occurred_at": datetime.now(UTC).isoformat(),
        }
    )


async def _dispatch(
    processor: StubProcessor,
    retrier: SpyRetrier,
    message: FakeMessage,
) -> None:
    await dispatch(
        _payload(),
        message,  # type: ignore[arg-type]
        processor=processor,
        retrier=retrier,
        default_bucket=DEFAULT_BUCKET,
        max_attempts=MAX_RETRIES + 1,
        max_retries=MAX_RETRIES,
    )


def test_real_implementations_satisfy_the_ports() -> None:
    assert isinstance(StubProcessor(), DocumentProcessor)
    assert isinstance(SpyRetrier(), MessageRetrier)


async def test_success_only_acknowledges() -> None:
    message = FakeMessage()
    retrier = SpyRetrier()

    await _dispatch(StubProcessor(), retrier, message)

    assert message.acked == 1
    assert not retrier.scheduled
    assert not retrier.dead_lettered


async def test_terminal_failure_goes_to_the_dlq_and_is_acknowledged() -> None:
    # Отказ уже зафиксирован транзакцией: повторять нечего, копия нужна на разбор.
    message = FakeMessage()
    retrier = SpyRetrier()
    processor = StubProcessor(
        result=ProcessDocumentResult(
            document_id=DocumentId(DOCUMENT_ID),
            status=DocumentStatus.FAILED,
            pages_total=0,
            chunks_total=0,
            failure_code="retries_exhausted",
        )
    )

    await _dispatch(processor, retrier, message)

    assert retrier.dead_lettered[0][HEADER_ERROR_CODE] == "retries_exhausted"
    assert message.acked == 1


async def test_permanent_error_goes_to_the_dlq_without_retry() -> None:
    message = FakeMessage()
    retrier = SpyRetrier()

    await _dispatch(
        StubProcessor(error=CorruptedDocumentError("файл не читается")),
        retrier,
        message,
    )

    assert not retrier.scheduled
    assert retrier.dead_lettered[0][HEADER_ERROR_CODE] == CorruptedDocumentError.code
    assert message.acked == 1


async def test_transient_error_schedules_the_next_step() -> None:
    message = FakeMessage(headers={HEADER_ATTEMPT: 1})
    retrier = SpyRetrier()

    await _dispatch(
        StubProcessor(error=ConcurrentProcessingError("документ занят")),
        retrier,
        message,
    )

    attempt, headers = retrier.scheduled[0]
    assert attempt == 2
    assert headers[HEADER_ATTEMPT] == 2
    assert message.acked == 1


async def test_unknown_error_is_treated_as_repeatable() -> None:
    # Незнакомая ошибка не повод хоронить документ: повтор безопаснее отказа.
    message = FakeMessage()
    retrier = SpyRetrier()

    await _dispatch(StubProcessor(error=RuntimeError("что-то новое")), retrier, message)

    assert retrier.scheduled
    assert message.acked == 1


async def test_exhausted_budget_goes_to_the_dlq() -> None:
    message = FakeMessage(headers={HEADER_ATTEMPT: MAX_RETRIES})
    retrier = SpyRetrier()

    await _dispatch(
        StubProcessor(error=ConcurrentProcessingError("документ занят")),
        retrier,
        message,
    )

    assert not retrier.scheduled
    assert retrier.dead_lettered[0][HEADER_ATTEMPTS_EXHAUSTED] is True


async def test_copies_carry_the_original_body() -> None:
    message = FakeMessage()
    retrier = SpyRetrier()

    await _dispatch(
        StubProcessor(error=ConcurrentProcessingError("документ занят")),
        retrier,
        message,
    )

    assert retrier.bodies == [BODY]


async def test_shutdown_leaves_the_message_untouched() -> None:
    # Остановку процесса подписчик не подтверждает: сообщение вернёт брокер.
    message = FakeMessage()
    retrier = SpyRetrier()

    with pytest.raises(asyncio.CancelledError):
        await _dispatch(StubProcessor(error=asyncio.CancelledError()), retrier, message)

    assert message.acked == 0
    assert message.rejected == 0
    assert not retrier.dead_lettered
