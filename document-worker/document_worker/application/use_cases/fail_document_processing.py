"""Терминальная транзакция T4f: фиксация отказа обработки.

Вызывается только для неисправимых ошибок. Для временных — никогда: иначе
недоступность хранилища на минуту навсегда пометила бы документ отказом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.dto.results import (
    FailDocumentProcessingResult,
    MessageOutcome,
    TerminalOutcome,
)
from document_worker.application.errors import DocumentNotFoundError
from document_worker.application.services.terminal import write_terminal_state

if TYPE_CHECKING:
    from datetime import datetime

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.commands import FailDocumentProcessingCommand
    from document_worker.application.ports.system import Clock
    from document_worker.application.ports.unit_of_work import (
        UnitOfWork,
        UnitOfWorkFactory,
    )
    from document_worker.domain.entities.document import Document


@dataclass(frozen=True, slots=True)
class FailDocumentProcessing:
    """Записывает отказ документа, прогона и событие о нём одной транзакцией."""

    uow_factory: UnitOfWorkFactory
    clock: Clock
    config: ProcessingConfig

    async def execute(
        self,
        command: FailDocumentProcessingCommand,
    ) -> FailDocumentProcessingResult:
        """Фиксирует отказ обработки документа.

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
            if self._is_terminal(document):
                return await self._duplicate(uow, command, now)

            job = await uow.jobs.get(document.id, self.config.pipeline_version)
            document.fail(
                code=command.error_code,
                message=command.error_message,
                stage=command.stage,
                now=now,
                pages_persisted=command.pages_persisted,
            )
            if job is not None:
                job.fail(
                    code=command.error_code,
                    message=command.error_message,
                    stage=command.stage,
                    now=now,
                )
            written = await write_terminal_state(
                uow,
                document=document,
                job=job,
                event_id=command.event_id,
                outcome=MessageOutcome.FAILED,
                now=now,
            )
            if not written.applied:
                return await self._duplicate(uow, command, now)
            await uow.commit()
            return FailDocumentProcessingResult(
                terminal=TerminalOutcome.APPLIED,
                events_enqueued=written.events_enqueued,
            )

    def _is_terminal(self, document: Document) -> bool:
        return (
            document.status.is_terminal
            and document.pipeline_version == self.config.pipeline_version
        )

    async def _duplicate(
        self,
        uow: UnitOfWork,
        command: FailDocumentProcessingCommand,
        now: datetime,
    ) -> FailDocumentProcessingResult:
        # Документ уже завершён, и переписывать его результат нельзя — но своё
        # сообщение нужно закрыть, иначе его лиз держит документ занятым.
        await uow.messages.mark_completed(
            command.event_id,
            outcome=MessageOutcome.FAILED,
            completed_at=now,
        )
        await uow.commit()
        return FailDocumentProcessingResult(terminal=TerminalOutcome.DUPLICATE)
