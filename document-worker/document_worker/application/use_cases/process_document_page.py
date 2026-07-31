"""Обработка одной страницы: чтение вне транзакции, запись внутри.

Страница читается секундами, а пишется миллисекундами, поэтому транзакция
открывается последней и держит только вставку строки и счётчик прогона.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from document_worker.application.dto.results import (
    JobProgressDTO,
    ProcessDocumentPageResult,
)
from document_worker.application.errors import (
    CorruptedPageImageError,
    PageLevelError,
    PageOcrTimeoutError,
    PageRenderError,
)
from document_worker.domain.entities.document_page import DocumentPage
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    PageFailureReason,
)
from document_worker.domain.value_objects.identifiers import PageId

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.commands import ProcessDocumentPageCommand
    from document_worker.application.ports.system import Clock, IdGenerator
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.normalization.normalizer import TextNormalizer

# Заглушка до подключения распознавания: страница не теряется, а фиксируется
# отказом, который снимется повторной обработкой новой версией пайплайна.
OCR_NOT_AVAILABLE: Final[str] = "распознавание ещё не подключено"

_FAILURE_REASONS: Final[Mapping[type[PageLevelError], PageFailureReason]] = {
    PageRenderError: PageFailureReason.RENDER_FAILED,
    PageOcrTimeoutError: PageFailureReason.TIMEOUT,
    CorruptedPageImageError: PageFailureReason.PAGE_CORRUPTED,
}


@dataclass(frozen=True, slots=True)
class ProcessDocumentPage:
    """Читает страницу выбранным способом и сохраняет её результат."""

    uow_factory: UnitOfWorkFactory
    normalizer: TextNormalizer
    ids: IdGenerator
    clock: Clock
    config: ProcessingConfig

    async def execute(
        self,
        command: ProcessDocumentPageCommand,
    ) -> ProcessDocumentPageResult:
        """Обрабатывает одну страницу и фиксирует её транзакцией T2ₙ."""
        now = self.clock.now()
        page = await self._read(command, now=now)
        async with self.uow_factory(statement_timeout_ms=self.config.tx.page_ms) as uow:
            persisted = await uow.pages.add(page)
            if persisted:
                # Счётчик двигается только вместе со строкой: повторная
                # доставка иначе насчитала бы страниц больше, чем в документе.
                await uow.jobs.record_progress(
                    command.job_id,
                    JobProgressDTO.for_page(page.method, at=now),
                )
            await uow.commit()
        return ProcessDocumentPageResult(
            number=page.number,
            page_id=page.id,
            status=page.status,
            method=page.method,
            confidence=page.confidence,
            char_count=page.char_count,
            failure_reason=page.failure.reason if page.failure else None,
            persisted=persisted,
        )

    async def _read(
        self,
        command: ProcessDocumentPageCommand,
        *,
        now: datetime,
    ) -> DocumentPage:
        try:
            return await self._extract(command, now=now)
        except PageLevelError as error:
            # Одна нечитаемая страница не отменяет документ: она сохраняется
            # отказом, и обработка идёт дальше.
            return self._failed(
                command,
                reason=_FAILURE_REASONS.get(
                    type(error), PageFailureReason.TEXT_EXTRACTION_FAILED
                ),
                message=error.message,
                now=now,
            )

    async def _extract(
        self,
        command: ProcessDocumentPageCommand,
        *,
        now: datetime,
    ) -> DocumentPage:
        if command.entry.method is not ExtractionMethod.TEXT_LAYER:
            return self._failed(
                command,
                reason=PageFailureReason.OCR_FAILED,
                message=OCR_NOT_AVAILABLE,
                now=now,
                recoverable=True,
            )
        page = await command.extraction.pdf.read_page_text(int(command.entry.number))
        normalized = self.normalizer.normalize(
            page.text, source=ExtractionMethod.TEXT_LAYER
        )
        return DocumentPage.from_text_layer(
            page_id=PageId(self.ids.new_uuid()),
            document_id=command.document_id,
            number=command.entry.number,
            pipeline_version=self.config.pipeline_version,
            content=normalized.content,
            now=now,
        )

    def _failed(
        self,
        command: ProcessDocumentPageCommand,
        *,
        reason: PageFailureReason,
        message: str,
        now: datetime,
        recoverable: bool = False,
    ) -> DocumentPage:
        return DocumentPage.failed(
            page_id=PageId(self.ids.new_uuid()),
            document_id=command.document_id,
            number=command.entry.number,
            pipeline_version=self.config.pipeline_version,
            reason=reason,
            message=message,
            now=now,
            recoverable=recoverable,
        )
