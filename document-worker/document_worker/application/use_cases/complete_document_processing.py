"""Терминальная транзакция T4: чем закончилась обработка документа.

Отказ по вердикту и отказ по исключению — разные вещи. Здесь фиксируется
первый: все страницы обработаны, но пригодного текста в них не оказалось.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from document_worker.application.dto.results import (
    CompleteDocumentProcessingResult,
    MessageOutcome,
    TerminalOutcome,
)
from document_worker.application.errors import (
    DocumentNotFoundError,
    DomainInvariantViolationError,
    translate_domain_error,
)
from document_worker.application.services.terminal import write_terminal_state
from document_worker.domain.errors import DomainError
from document_worker.domain.events import (
    DocumentPartiallyProcessed,
    DocumentProcessed,
    DocumentProcessingFailed,
)
from document_worker.domain.value_objects.enums import DocumentStatus, ProcessingStage
from document_worker.domain.value_objects.quality import PageOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.commands import (
        CompleteDocumentProcessingCommand,
    )
    from document_worker.application.dto.results import PageSummaryDTO
    from document_worker.application.ports.system import Clock
    from document_worker.application.ports.unit_of_work import (
        UnitOfWork,
        UnitOfWorkFactory,
    )
    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.processing_job import ProcessingJob
    from document_worker.domain.policies.document_status import DocumentStatusPolicy
    from document_worker.domain.value_objects.quality import DocumentStatusVerdict

EVENT_TYPE_BY_STATUS: Final[Mapping[DocumentStatus, str]] = {
    DocumentStatus.PROCESSED: DocumentProcessed.event_type,
    DocumentStatus.PARTIALLY_PROCESSED: DocumentPartiallyProcessed.event_type,
    DocumentStatus.FAILED: DocumentProcessingFailed.event_type,
}

REASON_NO_USABLE_PAGES: Final[str] = "no_usable_pages"
NO_USABLE_TEXT: Final[str] = "пригодного текста в документе не оказалось"

OUTCOME_BY_STATUS: Final[Mapping[DocumentStatus, MessageOutcome]] = {
    DocumentStatus.PROCESSED: MessageOutcome.PROCESSED,
    DocumentStatus.PARTIALLY_PROCESSED: MessageOutcome.PARTIALLY_PROCESSED,
    DocumentStatus.FAILED: MessageOutcome.FAILED,
}


@dataclass(frozen=True, slots=True)
class CompleteDocumentProcessing:
    """Считает итоговый статус документа и фиксирует его одной транзакцией."""

    uow_factory: UnitOfWorkFactory
    status_policy: DocumentStatusPolicy
    clock: Clock
    config: ProcessingConfig

    async def execute(
        self,
        command: CompleteDocumentProcessingCommand,
    ) -> CompleteDocumentProcessingResult:
        """Завершает обработку документа.

        Raises:
            DocumentNotFoundError: Строку документа удалили во время обработки.
        """
        now = self.clock.now()
        async with self.uow_factory(
            statement_timeout_ms=self.config.tx.terminal_ms
        ) as uow:
            document = await uow.documents.acquire(command.document_id)
            if document is None:
                raise DocumentNotFoundError(
                    "документ исчез во время обработки",
                    context={"document_id": str(command.document_id)},
                )
            summaries = await uow.pages.list_summaries(
                document.id, self.config.pipeline_version
            )
            self._record_facts(document, command)
            verdict = self.status_policy.evaluate(
                [_outcome_of(summary) for summary in summaries],
                declared_page_count=document.page_count,
            )
            job = await self._job_of(uow, document)
            self._transition(document, job, verdict, command, now)
            written = await write_terminal_state(
                uow,
                document=document,
                job=job,
                event_id=command.event_id,
                outcome=OUTCOME_BY_STATUS[verdict.status],
                now=now,
            )
            if not written.applied:
                # Документ завершил кто-то другой: его результат не переписывают.
                return await self._duplicate(uow, document, command, now)
            await uow.commit()
            return CompleteDocumentProcessingResult(
                terminal=TerminalOutcome.APPLIED,
                status=verdict.status,
                event_type=EVENT_TYPE_BY_STATUS[verdict.status],
                events_enqueued=written.events_enqueued,
                pages_total=verdict.stats.pages_total,
                pages_failed=verdict.stats.pages_failed_status,
            )

    async def _job_of(self, uow: UnitOfWork, document: Document) -> ProcessingJob:
        job = await uow.jobs.get(document.id, self.config.pipeline_version)
        if job is None:
            raise DomainInvariantViolationError(
                "прогон обработки документа не найден",
                context={"document_id": str(document.id)},
            )
        return job

    def _record_facts(
        self,
        document: Document,
        command: CompleteDocumentProcessingCommand,
    ) -> None:
        try:
            document.record_source(
                size=command.source_size, checksum=command.source_checksum
            )
            document.declare_page_count(command.page_count)
        except DomainError as error:
            raise translate_domain_error(error) from error

    def _transition(
        self,
        document: Document,
        job: ProcessingJob,
        verdict: DocumentStatusVerdict,
        command: CompleteDocumentProcessingCommand,
        now: datetime,
    ) -> None:
        try:
            self._apply(document, job, verdict, command, now)
        except DomainError as error:
            raise translate_domain_error(error) from error

    def _apply(
        self,
        document: Document,
        job: ProcessingJob,
        verdict: DocumentStatusVerdict,
        command: CompleteDocumentProcessingCommand,
        now: datetime,
    ) -> None:
        # На уже завершённом документе все переходы ниже — no-op: так домен и
        # устроен, а разошедшееся состояние ловит guard-UPDATE.
        stats = verdict.stats
        job.declare_pages(command.page_count)
        job.record_pages(
            text_layer=stats.pages_text_layer,
            ocr=stats.pages_ocr,
            hybrid=stats.pages_hybrid,
            failed=stats.pages_failed,
        )
        if verdict.status.is_successful:
            document.complete(verdict, chunks_total=command.chunks_total, now=now)
            job.succeed(result=verdict.status, now=now)
            return
        # Страницы обработаны все, но читать в документе нечего: это отказ
        # с причиной от политики, а не сбой обработки.
        reason = verdict.reasons[0] if verdict.reasons else REASON_NO_USABLE_PAGES
        document.fail(
            code=reason,
            message=NO_USABLE_TEXT,
            stage=ProcessingStage.TEXT_EXTRACTION,
            now=now,
            pages_persisted=stats.pages_total,
        )
        job.fail(
            code=reason,
            message=NO_USABLE_TEXT,
            stage=ProcessingStage.TEXT_EXTRACTION,
            now=now,
        )

    async def _duplicate(
        self,
        uow: UnitOfWork,
        document: Document,
        command: CompleteDocumentProcessingCommand,
        now: datetime,
    ) -> CompleteDocumentProcessingResult:
        # Событие уже опубликовал тот, кто завершил документ, а вот своё
        # сообщение закрыть нужно: иначе его лиз держит документ занятым.
        await uow.messages.mark_completed(
            command.event_id,
            outcome=OUTCOME_BY_STATUS[document.status],
            completed_at=now,
        )
        await uow.commit()
        return CompleteDocumentProcessingResult(
            terminal=TerminalOutcome.DUPLICATE,
            status=document.status,
            event_type=None,
            events_enqueued=0,
            pages_total=document.page_count or 0,
            pages_failed=0,
        )


def _outcome_of(summary: PageSummaryDTO) -> PageOutcome:
    return PageOutcome(
        page_number=summary.page_number,
        status=summary.status,
        method=summary.method,
        confidence=summary.confidence,
        char_count=summary.char_count,
        illegible_char_count=summary.illegible_char_count,
    )
