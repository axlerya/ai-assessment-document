"""Доменные события завершения обработки документа.

`event_id` не хранится, а выводится: он детерминирован по документу, версии
пайплайна и типу события, поэтому повторное завершение даёт тот же ключ и
дубль гасится уникальным ограничением outbox.

Валидация вынесена в функции, а не в цепочку super(): dataclass со slots
пересоздаёт класс, и безаргументный super() в потомке перестаёт работать.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, ClassVar, override

from document_worker.domain.errors import InvariantViolation
from document_worker.domain.value_objects.identifiers import EventId

if TYPE_CHECKING:
    from datetime import datetime

    from document_worker.domain.value_objects.confidence import OcrConfidence
    from document_worker.domain.value_objects.enums import ProcessingStage
    from document_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
    )
    from document_worker.domain.value_objects.versioning import PipelineVersion


def _validate_occurred_at(occurred_at: datetime) -> None:
    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
        raise InvariantViolation(
            "момент события обязан быть с указанием зоны",
            context={"occurred_at": occurred_at.isoformat()},
        )
    if occurred_at.utcoffset() != UTC.utcoffset(None):
        raise InvariantViolation(
            "момент события обязан быть в UTC",
            context={"occurred_at": occurred_at.isoformat()},
        )


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Общая часть терминальных событий документа."""

    document_id: DocumentId
    correlation_id: CorrelationId
    pipeline_version: PipelineVersion
    occurred_at: datetime

    event_type: ClassVar[str] = ""

    def __post_init__(self) -> None:
        """Требует момент события в UTC с указанием зоны."""
        _validate_occurred_at(self.occurred_at)

    @property
    def event_id(self) -> EventId:
        """Ключ дедупликации: один и тот же для повторного завершения."""
        return EventId.deterministic(
            document_id=self.document_id,
            pipeline_version=self.pipeline_version,
            event_type=type(self).event_type,
        )


@dataclass(frozen=True, slots=True)
class _PageCounters(DomainEvent):
    """Общие счётчики страниц двух успешных событий."""

    pages_total: int
    pages_text_layer: int
    pages_ocr: int
    pages_hybrid: int
    pages_failed: int
    chunks_total: int
    total_chars: int
    mean_ocr_confidence: OcrConfidence | None
    ocr_coverage: float
    processing_duration_ms: int


def _validate_counters(event: _PageCounters) -> None:
    counted = (
        event.pages_text_layer
        + event.pages_ocr
        + event.pages_hybrid
        + event.pages_failed
    )
    if counted != event.pages_total:
        raise InvariantViolation(
            "счётчики страниц не сходятся с их общим числом",
            context={"counted": counted, "total": event.pages_total},
        )


@dataclass(frozen=True, slots=True)
class DocumentProcessed(_PageCounters):
    """Все страницы прочитаны полностью, неразборчивых фрагментов нет."""

    event_type: ClassVar[str] = "document.processed"

    @override
    def __post_init__(self) -> None:
        """Запрещает непрочитанные страницы у полного успеха."""
        _validate_occurred_at(self.occurred_at)
        _validate_counters(self)
        if self.pages_failed:
            raise InvariantViolation(
                "полностью обработанный документ не имеет непрочитанных страниц",
                context={"pages_failed": self.pages_failed},
            )


@dataclass(frozen=True, slots=True)
class DocumentPartiallyProcessed(_PageCounters):
    """Результат пригоден, но помечен: часть страниц прочитана не полностью."""

    event_type: ClassVar[str] = "document.partially_processed"

    partially_illegible_page_numbers: tuple[int, ...]
    illegible_page_numbers: tuple[int, ...]
    failed_page_numbers: tuple[int, ...]
    illegible_char_ratio: float
    reasons: tuple[str, ...]

    @override
    def __post_init__(self) -> None:
        """Требует хотя бы одну проблемную страницу или причину."""
        _validate_occurred_at(self.occurred_at)
        _validate_counters(self)
        if not (
            self.partially_illegible_page_numbers
            or self.illegible_page_numbers
            or self.failed_page_numbers
            or self.reasons
        ):
            raise InvariantViolation(
                "частичная обработка без единой причины бессмысленна",
                context={"pages_total": self.pages_total},
            )


@dataclass(frozen=True, slots=True)
class DocumentProcessingFailed(DomainEvent):
    """Пригодного результата нет, документ индексировать нельзя.

    Признака «можно ли повторить» здесь нет: решение о retry и DLQ принимает
    application и в шину не транслирует.
    """

    event_type: ClassVar[str] = "document.processing.failed"

    error_code: str
    error_message: str
    stage: ProcessingStage
    attempt: int
    pages_total: int | None = None
    pages_persisted: int = 0
