"""Результаты use cases и вспомогательные проекции для чтения."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from document_worker.domain.value_objects.enums import ExtractionMethod

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from document_worker.domain.value_objects.confidence import OcrConfidence
    from document_worker.domain.value_objects.enums import (
        DocumentStatus,
        PageFailureReason,
        PageStatus,
    )
    from document_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
        EventId,
        PageId,
    )
    from document_worker.domain.value_objects.paging import PageNumber
    from document_worker.domain.value_objects.versioning import PipelineVersion


class ClaimOutcome(StrEnum):
    """Исход попытки занять сообщение."""

    PROCEED = "proceed"
    RESUME = "resume"
    REJECT_CONCURRENT = "reject_concurrent"
    SKIP = "skip"


class MessageOutcome(StrEnum):
    """Чем закончилась обработка сообщения."""

    PROCESSED = "processed"
    PARTIALLY_PROCESSED = "partially_processed"
    FAILED = "failed"


class TerminalOutcome(StrEnum):
    """Записан ли терминальный результат или его уже записал другой воркер."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class MessageClaimDTO:
    """Заявка на обработку сообщения."""

    event_id: EventId
    document_id: DocumentId
    correlation_id: CorrelationId
    pipeline_version: PipelineVersion
    message_type: str
    lease_owner: str
    lease_expires_at: datetime
    claimed_at: datetime


@dataclass(frozen=True, slots=True)
class ClaimOutcomeDTO:
    """Что делать с сообщением после попытки его занять."""

    outcome: ClaimOutcome
    persisted_page_numbers: frozenset[int] = frozenset()
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class PageSummaryDTO:
    """Лёгкая проекция страницы: без текста, только метрики."""

    page_id: PageId
    page_number: PageNumber
    status: PageStatus
    method: ExtractionMethod
    confidence: OcrConfidence | None
    char_count: int
    illegible_char_count: int


@dataclass(frozen=True, slots=True)
class JobProgressDTO:
    """Приращение прогресса прогона.

    Приращение, а не итог: воркер, продолживший чужую работу, знает только свои
    страницы, и запись итогов затёрла бы чужие. Счётчики по способам, а не один
    суммарный: строка прогона хранит именно это разбиение, а восстановить его
    из суммы нельзя.
    """

    heartbeat_at: datetime
    pages_text_layer: int = 0
    pages_ocr: int = 0
    pages_hybrid: int = 0
    pages_failed: int = 0
    chunks_created: int = 0

    @classmethod
    def for_page(cls, method: ExtractionMethod, *, at: datetime) -> Self:
        """Приращение на одну прочитанную страницу."""
        return cls(
            heartbeat_at=at,
            pages_text_layer=int(method is ExtractionMethod.TEXT_LAYER),
            pages_ocr=int(method is ExtractionMethod.OCR),
            pages_hybrid=int(method is ExtractionMethod.HYBRID),
            pages_failed=int(method is ExtractionMethod.NONE),
        )


@dataclass(frozen=True, slots=True)
class OutboxEventDTO:
    """Готовое к записи в outbox событие."""

    event_id: UUID
    aggregate_id: UUID
    event_type: str
    routing_key: str
    payload: dict[str, object]
    correlation_id: CorrelationId
    occurred_at: datetime
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutboxRecordDTO:
    """Строка outbox, взятая в публикацию."""

    id: int
    event_id: UUID
    routing_key: str
    payload: dict[str, object]
    headers: dict[str, str]
    correlation_id: str | None
    occurred_at: datetime
    attempts: int


@dataclass(frozen=True, slots=True)
class PublishOutboxEventsResult:
    """Итог одного прохода публикатора."""

    fetched: int
    published: int
    failed: int


@dataclass(frozen=True, slots=True)
class ProcessDocumentPageResult:
    """Итог одной страницы."""

    number: PageNumber
    page_id: PageId
    status: PageStatus
    method: ExtractionMethod
    confidence: OcrConfidence | None
    char_count: int
    failure_reason: PageFailureReason | None
    persisted: bool


@dataclass(frozen=True, slots=True)
class CompleteDocumentProcessingResult:
    """Итог терминальной транзакции успеха."""

    terminal: TerminalOutcome
    status: DocumentStatus
    event_type: str | None
    events_enqueued: int
    pages_total: int
    pages_failed: int


@dataclass(frozen=True, slots=True)
class FailDocumentProcessingResult:
    """Итог терминальной транзакции отказа."""

    terminal: TerminalOutcome
    events_enqueued: int = 0


@dataclass(frozen=True, slots=True)
class CreateDocumentChunksResult:
    """Итог чанкования документа.

    Вставлено и всего — разные величины: на повторной доставке вставлено ноль,
    а чанки у документа есть, и в событие уходит именно общее число.
    """

    chunks_created: int
    chunks_total: int


@dataclass(frozen=True, slots=True)
class ProcessDocumentResult:
    """Итог обработки одного сообщения."""

    document_id: DocumentId
    status: DocumentStatus
    pages_total: int
    chunks_total: int
    # Страницы, прочитанные именно этим прогоном: на возобновлении их меньше,
    # чем в документе, и по разнице видно, что возобновление сработало.
    pages_processed: int = 0
    duplicate: bool = False
    # Код отказа сопровождает копию сообщения в очереди разбора: тело без
    # причины оператору ничего не объясняет.
    failure_code: str | None = None
