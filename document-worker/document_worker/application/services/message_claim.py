"""Захват сообщения: транзакция T0 и снятие лиза.

Порядок внутри транзакции обязателен: сначала проверяется терминальность
документа, потом занимается сообщение. Обратный порядок оставлял бы в ветке
«уже обработан» запись `in_progress`, которую некому перевести в `completed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from document_worker.application.dto.results import ClaimOutcome, MessageClaimDTO
from document_worker.application.errors import (
    ConcurrentProcessingError,
    DocumentNotFoundError,
    InvalidCommandError,
)
from document_worker.domain.entities.processing_job import ProcessingJob
from document_worker.domain.value_objects.identifiers import JobId

if TYPE_CHECKING:
    from datetime import datetime

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.commands import ProcessDocumentCommand
    from document_worker.application.ports.system import Clock, IdGenerator
    from document_worker.application.ports.unit_of_work import (
        UnitOfWork,
        UnitOfWorkFactory,
    )
    from document_worker.domain.entities.document import Document

MESSAGE_TYPE = "document.process.requested"


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """Чем закончилась попытка занять сообщение."""

    outcome: ClaimOutcome
    document: Document
    # Прогон открыт вместе с захватом, и запрашивать его заново на каждой
    # странице значило бы ходить в базу за уже известным.
    job: ProcessingJob | None = None
    persisted_page_numbers: frozenset[int] = frozenset()
    attempts: int = 1

    @property
    def should_process(self) -> bool:
        """Нужно ли браться за работу."""
        return self.outcome in (ClaimOutcome.PROCEED, ClaimOutcome.RESUME)


@dataclass(frozen=True, slots=True)
class MessageClaimService:
    """Занимает сообщение и открывает прогон обработки."""

    uow_factory: UnitOfWorkFactory
    clock: Clock
    ids: IdGenerator
    config: ProcessingConfig

    async def claim(self, command: ProcessDocumentCommand) -> ClaimResult:
        """Занимает сообщение и переводит документ в обработку.

        Raises:
            DocumentNotFoundError: Строки документа ещё нет.
            InvalidCommandError: Команда указывает на чужой объект.
            ConcurrentProcessingError: Документ занят живым лизом.
        """
        now = self.clock.now()
        async with self.uow_factory(
            statement_timeout_ms=self.config.tx.claim_ms
        ) as uow:
            document = await self._acquire(uow, command)
            if self._is_already_done(document):
                return ClaimResult(outcome=ClaimOutcome.SKIP, document=document)

            claimed = await uow.messages.try_claim(self._claim_dto(command, now))
            if claimed.outcome is ClaimOutcome.REJECT_CONCURRENT:
                # Попытка расходуется намеренно: иначе живой лиз зависшего
                # воркера гонял бы сообщение по первой ступени без предела.
                raise ConcurrentProcessingError(
                    "документ уже обрабатывается другим воркером",
                    context={"document_id": str(command.document_id)},
                )
            if claimed.outcome is ClaimOutcome.SKIP:
                return ClaimResult(
                    outcome=ClaimOutcome.SKIP,
                    document=document,
                    attempts=claimed.attempts,
                )

            job = await self._start_processing(uow, command, now)
            await uow.commit()
            return ClaimResult(
                outcome=claimed.outcome,
                document=document,
                job=job,
                persisted_page_numbers=claimed.persisted_page_numbers,
                attempts=claimed.attempts,
            )

    async def release(self, command: ProcessDocumentCommand) -> None:
        """Просрочивает лиз, оставляя работу незавершённой.

        Следующая доставка получит RESUME немедленно, а не через таймаут лиза.
        """
        async with self.uow_factory(
            statement_timeout_ms=self.config.tx.release_ms
        ) as uow:
            await uow.messages.release(command.event_id, at=self.clock.now())
            await uow.commit()

    async def _acquire(
        self,
        uow: UnitOfWork,
        command: ProcessDocumentCommand,
    ) -> Document:
        document = await uow.documents.acquire(command.document_id)
        if document is None:
            # Строку создаёт сервис приёма файлов: сообщение обогнало его
            # коммит, и повтор через несколько секунд её увидит.
            raise DocumentNotFoundError(
                "строки документа ещё нет",
                context={"document_id": str(command.document_id)},
            )
        if document.source.ref != command.object_ref:
            raise InvalidCommandError(
                "команда указывает на другой объект хранилища",
                context={
                    "document_id": str(command.document_id),
                    "expected": document.source.ref.to_uri(),
                    "received": command.object_ref.to_uri(),
                },
            )
        if document.source.mime_type != command.mime_type:
            raise InvalidCommandError(
                "команда описывает не тот тип файла, что записан в документе",
                context={
                    "document_id": str(command.document_id),
                    "expected": document.source.mime_type.value,
                    "received": command.mime_type.value,
                },
            )
        return document

    def _is_already_done(self, document: Document) -> bool:
        return (
            document.status.is_terminal
            and document.pipeline_version == self.config.pipeline_version
        )

    def _claim_dto(
        self,
        command: ProcessDocumentCommand,
        now: datetime,
    ) -> MessageClaimDTO:
        return MessageClaimDTO(
            event_id=command.event_id,
            document_id=command.document_id,
            correlation_id=command.correlation_id,
            pipeline_version=self.config.pipeline_version,
            message_type=MESSAGE_TYPE,
            lease_owner=self.config.consumer_name,
            lease_expires_at=now + timedelta(seconds=self.config.claim_lease_s),
            claimed_at=now,
        )

    async def _start_processing(
        self,
        uow: UnitOfWork,
        command: ProcessDocumentCommand,
        now: datetime,
    ) -> ProcessingJob:
        await uow.documents.start_processing(
            command.document_id,
            pipeline_version=self.config.pipeline_version,
            at=now,
        )
        job = ProcessingJob.schedule(
            job_id=JobId(self.ids.new_uuid()),
            document_id=command.document_id,
            event_id=command.event_id,
            correlation_id=command.correlation_id,
            pipeline_version=self.config.pipeline_version,
            now=now,
        )
        # Прогон заводится сразу работающим: очередь между claim и работой
        # пуста по построению, а строка `queued` без исполнителя ничего не
        # значит и требовала бы второго перехода.
        job.start(now=now)
        return await uow.jobs.start(job)
