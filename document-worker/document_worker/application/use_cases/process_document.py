"""Оркестратор обработки документа: последовательность шагов и два except.

Фабрики единиц работы здесь нет намеренно — транзакции держат исполнители,
и открыть транзакцию поверх цикла по страницам оркестратору физически нечем.
Бизнес-правил тут тоже нет: числовые решения принимают доменные политики.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from document_worker.application.dto.commands import (
    CompleteDocumentProcessingCommand,
    CreateDocumentChunksCommand,
    ExtractDocumentTextCommand,
    FailDocumentProcessingCommand,
)
from document_worker.application.dto.results import ProcessDocumentResult
from document_worker.application.errors import (
    InvalidCommandError,
    PermanentError,
    ProcessingDeadlineExceededError,
    TransientError,
)
from document_worker.domain.value_objects.enums import DocumentStatus, ProcessingStage

# Лестница повторов исчерпана: держать документ в обработке дальше нечем.
RETRIES_EXHAUSTED: Final[str] = "retries_exhausted"

if TYPE_CHECKING:
    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.commands import ProcessDocumentCommand
    from document_worker.application.errors import ApplicationError
    from document_worker.application.ports.system import TempWorkspaceFactory
    from document_worker.application.services.message_claim import (
        ClaimResult,
        MessageClaimService,
    )
    from document_worker.application.services.page_runner import (
        PageSequenceRunner,
        PageSequenceRunResult,
    )
    from document_worker.application.services.source_loader import (
        LoadedSource,
        SourceDocumentLoader,
    )
    from document_worker.application.use_cases.complete_document_processing import (
        CompleteDocumentProcessing,
    )
    from document_worker.application.use_cases.create_document_chunks import (
        CreateDocumentChunks,
    )
    from document_worker.application.use_cases.extract_document_text import (
        ExtractDocumentText,
    )
    from document_worker.application.use_cases.fail_document_processing import (
        FailDocumentProcessing,
    )
    from document_worker.domain.value_objects.identifiers import JobId


@dataclass(frozen=True, slots=True)
class ProcessDocument:
    """Проводит документ от захвата сообщения до терминального события."""

    claim_service: MessageClaimService
    source_loader: SourceDocumentLoader
    extract_text: ExtractDocumentText
    page_runner: PageSequenceRunner
    create_chunks: CreateDocumentChunks
    complete: CompleteDocumentProcessing
    fail: FailDocumentProcessing
    workspaces: TempWorkspaceFactory
    config: ProcessingConfig

    async def execute(self, command: ProcessDocumentCommand) -> ProcessDocumentResult:
        """Обрабатывает одно сообщение целиком.

        Raises:
            ConcurrentProcessingError: Документ занят живым лизом.
            TransientError: Сбой, который имеет смысл повторить.
        """
        claim = await self.claim_service.claim(command)
        if not claim.should_process:
            return _skipped(claim)
        try:
            async with asyncio.timeout(self.config.document_timeout_s):
                return await self._process(command, claim)
        except PermanentError as error:
            # Результат зафиксирован, и повторять нечего: presentation
            # подтверждает сообщение и кладёт копию в DLQ для разбора.
            return await self._fail(command, claim, error)
        except TimeoutError as error:
            await self.claim_service.release(command)
            raise ProcessingDeadlineExceededError(
                "документ не уложился в отведённое время",
                context={"document_id": str(command.document_id)},
            ) from error
        except TransientError as error:
            if command.is_last_attempt:
                # Повторов больше не будет, и незакрытый документ висел бы в
                # обработке до вмешательства оператора.
                return await self._fail(command, claim, error, code=RETRIES_EXHAUSTED)
            # Документ остаётся в обработке: в failed его не переводит ничто.
            await self.claim_service.release(command)
            raise

    async def _process(
        self,
        command: ProcessDocumentCommand,
        claim: ClaimResult,
    ) -> ProcessDocumentResult:
        async with self.workspaces(prefix=f"doc-{command.document_id}") as workspace:
            source = await self.source_loader.load(claim.document, workspace=workspace)
            page_count, run = await self._read_pages(command, claim, source)
            chunks = await self.create_chunks.execute(
                CreateDocumentChunksCommand(
                    document_id=command.document_id,
                    correlation_id=command.correlation_id,
                    job_id=_job_id(claim),
                )
            )
            done = await self.complete.execute(
                CompleteDocumentProcessingCommand(
                    document_id=command.document_id,
                    correlation_id=command.correlation_id,
                    event_id=command.event_id,
                    job_id=_job_id(claim),
                    page_count=page_count,
                    chunks_total=chunks.chunks_total,
                    source_size=source.size,
                    source_checksum=source.checksum,
                )
            )
        return ProcessDocumentResult(
            document_id=command.document_id,
            status=done.status,
            pages_total=done.pages_total,
            chunks_total=chunks.chunks_total,
            pages_processed=run.processed,
            duplicate=done.event_type is None,
        )

    async def _read_pages(
        self,
        command: ProcessDocumentCommand,
        claim: ClaimResult,
        source: LoadedSource,
    ) -> tuple[int, PageSequenceRunResult]:
        async with self.extract_text.execute(
            ExtractDocumentTextCommand(
                document_id=command.document_id,
                correlation_id=command.correlation_id,
                source_path=source.path,
            )
        ) as extraction:
            run = await self.page_runner.run(
                extraction=extraction,
                document_id=command.document_id,
                correlation_id=command.correlation_id,
                job_id=_job_id(claim),
                persisted=claim.persisted_page_numbers,
            )
            return extraction.plan.page_count, run

    async def _fail(
        self,
        command: ProcessDocumentCommand,
        claim: ClaimResult,
        error: ApplicationError,
        *,
        code: str | None = None,
    ) -> ProcessDocumentResult:
        failure_code = code or type(error).code
        await self.fail.execute(
            FailDocumentProcessingCommand(
                document_id=command.document_id,
                correlation_id=command.correlation_id,
                event_id=command.event_id,
                job_id=_job_id(claim),
                error_code=failure_code,
                error_message=error.message,
                stage=ProcessingStage.TEXT_EXTRACTION,
                pages_persisted=len(claim.persisted_page_numbers),
            )
        )
        return ProcessDocumentResult(
            document_id=command.document_id,
            status=DocumentStatus.FAILED,
            pages_total=0,
            chunks_total=0,
            failure_code=failure_code,
        )


def _skipped(claim: ClaimResult) -> ProcessDocumentResult:
    return ProcessDocumentResult(
        document_id=claim.document.id,
        status=claim.document.status,
        pages_total=claim.document.page_count or 0,
        chunks_total=0,
        duplicate=True,
    )


def _job_id(claim: ClaimResult) -> JobId:
    if claim.job is None:  # pragma: no cover — работа без прогона не начинается
        raise InvalidCommandError("прогон обработки не открыт")
    return claim.job.id
