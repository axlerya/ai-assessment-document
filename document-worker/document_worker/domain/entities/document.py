"""Агрегат документа.

Коллекций страниц и чанков здесь нет: страницы пишутся своими транзакциями,
загружать ради каждой весь агрегат нельзя. Принадлежность и уникальность
держат ограничения БД.

Фабрики создания тоже нет — строку documents создаёт сервис приёма файлов,
worker её только захватывает и дописывает свои колонки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from typing import TYPE_CHECKING

from document_worker.domain.constants import MAX_PAGES
from document_worker.domain.errors import (
    DocumentTooLarge,
    EmptyDocument,
    IncompletePageSet,
    InvalidStatusTransition,
    InvariantViolation,
)
from document_worker.domain.events import (
    DocumentPartiallyProcessed,
    DocumentProcessed,
    DocumentProcessingFailed,
)
from document_worker.domain.value_objects.enums import (
    CompletionOutcome,
    DocumentStatus,
)

if TYPE_CHECKING:
    from datetime import datetime

    from document_worker.domain.events import DomainEvent
    from document_worker.domain.value_objects.enums import ProcessingStage
    from document_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
    )
    from document_worker.domain.value_objects.paging import PageNumber
    from document_worker.domain.value_objects.quality import (
        DocumentQualityStats,
        DocumentStatusVerdict,
    )
    from document_worker.domain.value_objects.storage import SourceFile
    from document_worker.domain.value_objects.versioning import PipelineVersion

_MILLISECONDS_IN_SECOND = 1000


def _require_utc(moment: datetime, name: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() != UTC.utcoffset(None):
        raise InvariantViolation(
            f"{name} обязан быть в UTC с указанием зоны",
            context={name: moment.isoformat()},
        )


@dataclass(slots=True)
class Document:
    """Состояние документа и переходы между статусами."""

    id: DocumentId
    source: SourceFile
    status: DocumentStatus
    pipeline_version: PipelineVersion
    correlation_id: CorrelationId
    created_at: datetime
    updated_at: datetime
    page_count: int | None = None
    processing_started_at: datetime | None = None
    processed_at: datetime | None = None
    stats: DocumentQualityStats | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    failure_stage: ProcessingStage | None = None
    _events: list[DomainEvent] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Проверяет временные метки."""
        _require_utc(self.created_at, "created_at")
        _require_utc(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise InvariantViolation(
                "документ изменён раньше, чем создан",
                context={
                    "created_at": self.created_at.isoformat(),
                    "updated_at": self.updated_at.isoformat(),
                },
            )

    def start_processing(
        self,
        *,
        now: datetime,
        pipeline_version: PipelineVersion | None = None,
    ) -> None:
        """Берёт документ в обработку.

        Вернуться из терминального статуса можно только строго более новой
        версией пайплайна: это открывает новое пространство имён, а не
        переписывает прошлый результат.

        Raises:
            InvalidStatusTransition: Переход запрещён или версия не новее.
        """
        _require_utc(now, "now")
        target = pipeline_version or self.pipeline_version
        if self.status.is_terminal and not target.is_newer_than(self.pipeline_version):
            raise InvalidStatusTransition(
                "повторная обработка требует более новой версии пайплайна",
                context={"current": str(self.pipeline_version), "target": str(target)},
            )
        self.status.ensure_can_transition_to(DocumentStatus.PROCESSING)

        self.status = DocumentStatus.PROCESSING
        self.pipeline_version = target
        self.processing_started_at = now
        self.updated_at = now
        self.processed_at = None
        self.stats = None
        self.failure_code = None
        self.failure_message = None
        self.failure_stage = None

    def declare_page_count(self, value: int) -> None:
        """Объявляет число страниц документа.

        Raises:
            EmptyDocument: Ноль страниц.
            DocumentTooLarge: Больше допустимого предела.
            InvariantViolation: Повторное объявление другого числа.
        """
        if value <= 0:
            raise EmptyDocument(
                "документ без страниц",
                context={"document_id": str(self.id)},
            )
        if value > MAX_PAGES:
            raise DocumentTooLarge(actual_bytes=value, limit_bytes=MAX_PAGES)
        if self.page_count is not None and self.page_count != value:
            raise InvariantViolation(
                "число страниц уже объявлено другим значением",
                context={"declared": self.page_count, "value": value},
            )
        self.page_count = value

    def complete(
        self,
        verdict: DocumentStatusVerdict,
        *,
        chunks_total: int,
        now: datetime,
    ) -> CompletionOutcome:
        """Завершает обработку успехом или частичным успехом.

        Повтор на уже завершённом документе — no-op: гонка двух воркеров не
        должна портить полученный результат.

        Raises:
            IncompletePageSet: Число страниц не объявлено или не совпадает.
            InvalidStatusTransition: Документ не в обработке.
        """
        _require_utc(now, "now")
        if self.status.is_terminal:
            return CompletionOutcome.DUPLICATE
        self.status.ensure_can_transition_to(verdict.status)
        self._ensure_page_set_complete(verdict.stats)

        self.status = verdict.status
        self.stats = verdict.stats
        self.processed_at = now
        self.updated_at = now
        self._events.append(self._completion_event(verdict, chunks_total, now))
        return CompletionOutcome.APPLIED

    def fail(  # noqa: PLR0913 — все шесть значений уходят в событие отказа
        self,
        *,
        code: str,
        message: str,
        stage: ProcessingStage,
        now: datetime,
        attempt: int = 1,
        pages_persisted: int = 0,
    ) -> CompletionOutcome:
        """Завершает обработку отказом.

        Повтор на уже завершённом документе — no-op: именно этот путь
        превращал корректно обработанный документ в failed.

        Raises:
            InvalidStatusTransition: Переход в отказ запрещён.
        """
        _require_utc(now, "now")
        if self.status.is_terminal:
            return CompletionOutcome.DUPLICATE
        self.status.ensure_can_transition_to(DocumentStatus.FAILED)

        self.status = DocumentStatus.FAILED
        self.failure_code = code
        self.failure_message = message
        self.failure_stage = stage
        self.processed_at = now
        self.updated_at = now
        self._events.append(
            DocumentProcessingFailed(
                document_id=self.id,
                correlation_id=self.correlation_id,
                pipeline_version=self.pipeline_version,
                occurred_at=now,
                error_code=code,
                error_message=message,
                stage=stage,
                attempt=attempt,
                pages_total=self.page_count,
                pages_persisted=pages_persisted,
            )
        )
        return CompletionOutcome.APPLIED

    def pull_events(self) -> tuple[DomainEvent, ...]:
        """Забирает накопленные события и очищает буфер."""
        events = tuple(self._events)
        self._events.clear()
        return events

    def _ensure_page_set_complete(self, stats: DocumentQualityStats) -> None:
        if self.page_count is None:
            raise IncompletePageSet(
                "число страниц не объявлено",
                context={"document_id": str(self.id)},
            )
        if stats.pages_total != self.page_count:
            raise IncompletePageSet(
                "обработано не столько страниц, сколько объявлено",
                context={"declared": self.page_count, "processed": stats.pages_total},
            )

    def _duration_ms(self, now: datetime) -> int:
        if self.processing_started_at is None:
            return 0
        elapsed = now - self.processing_started_at
        return int(elapsed.total_seconds() * _MILLISECONDS_IN_SECOND)

    def _completion_event(
        self,
        verdict: DocumentStatusVerdict,
        chunks_total: int,
        now: datetime,
    ) -> DomainEvent:
        stats = verdict.stats
        if verdict.status is DocumentStatus.PROCESSED:
            return DocumentProcessed(
                document_id=self.id,
                correlation_id=self.correlation_id,
                pipeline_version=self.pipeline_version,
                occurred_at=now,
                pages_total=stats.pages_total,
                pages_text_layer=stats.pages_text_layer,
                pages_ocr=stats.pages_ocr,
                pages_hybrid=stats.pages_hybrid,
                pages_failed=stats.pages_failed,
                chunks_total=chunks_total,
                total_chars=stats.total_chars,
                mean_ocr_confidence=stats.mean_ocr_confidence,
                ocr_coverage=stats.ocr_coverage,
                processing_duration_ms=self._duration_ms(now),
            )
        return DocumentPartiallyProcessed(
            document_id=self.id,
            correlation_id=self.correlation_id,
            pipeline_version=self.pipeline_version,
            occurred_at=now,
            pages_total=stats.pages_total,
            pages_text_layer=stats.pages_text_layer,
            pages_ocr=stats.pages_ocr,
            pages_hybrid=stats.pages_hybrid,
            pages_failed=stats.pages_failed,
            chunks_total=chunks_total,
            total_chars=stats.total_chars,
            mean_ocr_confidence=stats.mean_ocr_confidence,
            ocr_coverage=stats.ocr_coverage,
            processing_duration_ms=self._duration_ms(now),
            partially_illegible_page_numbers=_numbers(
                verdict.partially_illegible_pages
            ),
            illegible_page_numbers=_numbers(verdict.illegible_pages),
            failed_page_numbers=_numbers(verdict.failed_pages),
            illegible_char_ratio=stats.illegible_char_ratio,
            reasons=verdict.reasons,
        )


def _numbers(pages: tuple[PageNumber, ...]) -> tuple[int, ...]:
    return tuple(int(page) for page in pages)
