"""Прогон обработки документа.

Признака «можно ли повторить» здесь нет: решение о retry и DLQ принимает
application по типу ошибки.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Final, Self

from document_worker.domain.errors import InvalidStatusTransition, InvariantViolation
from document_worker.domain.value_objects.enums import CompletionOutcome, JobStatus

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime, timedelta

    from document_worker.domain.value_objects.enums import (
        DocumentStatus,
        ProcessingStage,
    )
    from document_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
        EventId,
        JobId,
    )
    from document_worker.domain.value_objects.versioning import PipelineVersion

_MIN_ATTEMPT = 1

_TRANSITIONS: Final[Mapping[JobStatus, frozenset[JobStatus]]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
}

_TERMINAL_STATUSES: Final[frozenset[JobStatus]] = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED}
)


def _require_utc(moment: datetime, name: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() != UTC.utcoffset(None):
        raise InvariantViolation(
            f"{name} обязан быть в UTC с указанием зоны",
            context={name: moment.isoformat()},
        )


@dataclass(slots=True)
class ProcessingJob:
    """Одна попытка обработать документ конкретной версией пайплайна."""

    id: JobId
    document_id: DocumentId
    event_id: EventId
    correlation_id: CorrelationId
    pipeline_version: PipelineVersion
    status: JobStatus
    attempt: int
    scheduled_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    pages_total: int | None = None
    # Разбиение по способам, а не один счётчик: событие и строка прогона
    # требуют именно его, а вывести способ из числа готовых страниц нельзя.
    pages_text_layer: int = 0
    pages_ocr: int = 0
    pages_hybrid: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    result_status: DocumentStatus | None = None
    error_code: str | None = None
    error_message: str | None = None
    stage: ProcessingStage | None = None

    def __post_init__(self) -> None:
        """Проверяет номер попытки и отметку постановки в очередь."""
        _require_utc(self.scheduled_at, "scheduled_at")
        if self.attempt < _MIN_ATTEMPT:
            raise InvariantViolation(
                "номер попытки начинается с единицы",
                context={"attempt": self.attempt},
            )

    @classmethod
    def schedule(  # noqa: PLR0913 — прогон описывается всеми этими значениями
        cls,
        *,
        job_id: JobId,
        document_id: DocumentId,
        event_id: EventId,
        correlation_id: CorrelationId,
        pipeline_version: PipelineVersion,
        now: datetime,
        attempt: int = 1,
    ) -> Self:
        """Ставит прогон в очередь."""
        return cls(
            id=job_id,
            document_id=document_id,
            event_id=event_id,
            correlation_id=correlation_id,
            pipeline_version=pipeline_version,
            status=JobStatus.QUEUED,
            attempt=attempt,
            scheduled_at=now,
        )

    @property
    def is_terminal(self) -> bool:
        """Завершён ли прогон."""
        return self.status in _TERMINAL_STATUSES

    @property
    def pages_done(self) -> int:
        """Сколько страниц прочитано любым способом."""
        return self.pages_text_layer + self.pages_ocr + self.pages_hybrid

    def start(self, *, now: datetime) -> None:
        """Берёт прогон в работу.

        Raises:
            InvalidStatusTransition: Прогон уже начат или завершён.
            InvariantViolation: Момент старта раньше постановки в очередь.
        """
        _require_utc(now, "now")
        self._ensure_can_move_to(JobStatus.RUNNING)
        if now < self.scheduled_at:
            raise InvariantViolation(
                "прогон не может начаться раньше, чем был поставлен в очередь",
                context={"scheduled_at": self.scheduled_at.isoformat()},
            )
        self.status = JobStatus.RUNNING
        self.started_at = now

    def declare_pages(self, total: int) -> None:
        """Объявляет число страниц документа.

        Raises:
            InvariantViolation: Отрицательное или конфликтующее значение.
        """
        if total < 0:
            raise InvariantViolation(
                "число страниц отрицательно",
                context={"total": total},
            )
        if self.pages_total is not None and self.pages_total != total:
            raise InvariantViolation(
                "число страниц уже объявлено другим значением",
                context={"declared": self.pages_total, "total": total},
            )
        self.pages_total = total

    def record_pages(
        self,
        *,
        text_layer: int,
        ocr: int,
        hybrid: int,
        failed: int,
    ) -> None:
        """Ставит счётчики страниц по фактически сохранённым.

        Именно постановка, а не приращение: итог считается по строкам страниц,
        а не по тому, что успел насчитать прогресс живого прогона.

        Raises:
            InvariantViolation: Страниц больше, чем объявлено.
        """
        counted = text_layer + ocr + hybrid + failed
        if self.pages_total is not None and counted > self.pages_total:
            raise InvariantViolation(
                "учтено больше страниц, чем объявлено",
                context={"counted": counted, "total": self.pages_total},
            )
        self.pages_text_layer = text_layer
        self.pages_ocr = ocr
        self.pages_hybrid = hybrid
        self.pages_failed = failed

    def succeed(self, *, result: DocumentStatus, now: datetime) -> CompletionOutcome:
        """Завершает прогон успехом.

        Raises:
            InvalidStatusTransition: Прогон не в работе.
            InvariantViolation: Результат не является успешным.
        """
        _require_utc(now, "now")
        if self.is_terminal:
            return CompletionOutcome.DUPLICATE
        self._ensure_can_move_to(JobStatus.SUCCEEDED)
        if not result.is_successful:
            raise InvariantViolation(
                "успешный прогон не может дать неуспешный статус документа",
                context={"result": result.value},
            )
        self._finish(now)
        self.status = JobStatus.SUCCEEDED
        self.result_status = result
        return CompletionOutcome.APPLIED

    def fail(
        self,
        *,
        code: str,
        message: str,
        stage: ProcessingStage,
        now: datetime,
    ) -> CompletionOutcome:
        """Завершает прогон отказом.

        Raises:
            InvalidStatusTransition: Переход в отказ запрещён.
        """
        _require_utc(now, "now")
        if self.is_terminal:
            return CompletionOutcome.DUPLICATE
        self._ensure_can_move_to(JobStatus.FAILED)
        self._finish(now)
        self.status = JobStatus.FAILED
        self.error_code = code
        self.error_message = message
        self.stage = stage
        return CompletionOutcome.APPLIED

    def duration(self) -> timedelta | None:
        """Длительность прогона, если он начат и завершён."""
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def _ensure_can_move_to(self, target: JobStatus) -> None:
        if target not in _TRANSITIONS[self.status]:
            raise InvalidStatusTransition(
                f"переход прогона {self.status.value} -> {target.value} запрещён",
                context={"from": self.status.value, "to": target.value},
            )

    def _finish(self, now: datetime) -> None:
        if self.started_at is not None and now < self.started_at:
            raise InvariantViolation(
                "прогон не может завершиться раньше, чем начался",
                context={"started_at": self.started_at.isoformat()},
            )
        self.finished_at = now
