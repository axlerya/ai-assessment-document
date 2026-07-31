"""Общая часть терминальных транзакций T4 и T4f.

Статус документа, строка прогона, исходящее событие и отметка сообщения
пишутся вместе — это требование устава, и разносить их по транзакциям нельзя.
Ноль строк от guard-UPDATE означает «кто-то уже завершил» и ошибкой не
является: именно эта ветка не даёт пометить обработанный документ отказом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.events import to_outbox_event
from document_worker.domain.value_objects.enums import DocumentStatus, JobStatus

if TYPE_CHECKING:
    from datetime import datetime

    from document_worker.application.dto.results import MessageOutcome
    from document_worker.application.ports.unit_of_work import UnitOfWork
    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.processing_job import ProcessingJob
    from document_worker.domain.value_objects.identifiers import EventId


@dataclass(frozen=True, slots=True)
class TerminalWriteResult:
    """Записалось ли терминальное состояние и сколько событий встало в очередь."""

    applied: bool
    events_enqueued: int = 0


async def write_terminal_state(  # noqa: PLR0913 — транзакция пишет ровно эти пять вещей
    uow: UnitOfWork,
    *,
    document: Document,
    job: ProcessingJob | None,
    event_id: EventId,
    outcome: MessageOutcome,
    now: datetime,
) -> TerminalWriteResult:
    """Пишет терминальное состояние документа и всё, что идёт вместе с ним."""
    if not await uow.documents.finish(document, expected=DocumentStatus.PROCESSING):
        return TerminalWriteResult(applied=False)
    if job is not None:
        await uow.jobs.finish(job, expected=JobStatus.RUNNING)
    enqueued = await uow.outbox.enqueue(
        [to_outbox_event(event) for event in document.pull_events()]
    )
    await uow.messages.mark_completed(event_id, outcome=outcome, completed_at=now)
    return TerminalWriteResult(applied=True, events_enqueued=enqueued)
