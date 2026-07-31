"""Построение плана извлечения. Транзакций не открывает и БД не трогает."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.dto.extraction import (
    DocumentExtraction,
    DocumentExtractionPlanDTO,
    PagePlanEntryDTO,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from document_worker.application.dto.commands import ExtractDocumentTextCommand
    from document_worker.application.ports.pdf import (
        PageRenderer,
        PdfDocumentReader,
        PdfHandle,
        PdfInspector,
    )
    from document_worker.domain.policies.text_layer_quality import (
        TextLayerProbe,
        TextLayerQualityPolicy,
    )


@dataclass(frozen=True, slots=True)
class ExtractDocumentText:
    """Решает, чем читать каждую страницу, и открывает нужные ресурсы."""

    inspector: PdfInspector
    reader: PdfDocumentReader
    renderer: PageRenderer
    policy: TextLayerQualityPolicy

    @contextlib.asynccontextmanager
    async def execute(
        self,
        command: ExtractDocumentTextCommand,
    ) -> AsyncIterator[DocumentExtraction]:
        """Строит план и держит открытыми документ и сессию рендера.

        Raises:
            UnsupportedMediaTypeError: Содержимое файла не PDF.
            CorruptedDocumentError: Документ не читается или пуст.
            EncryptedDocumentError: Документ защищён паролем.
            PageLimitExceededError: Страниц больше допустимого предела.
        """
        # Инспекция идёт первой: разбирать триста лишних страниц ради отказа
        # по их числу значит потратить память рабочего процесса впустую.
        inspection = await self.inspector.inspect(command.source_path)
        async with contextlib.AsyncExitStack() as stack:
            pdf = await stack.enter_async_context(self.reader.open(command.source_path))
            plan = await self._plan(command, pdf, page_count=inspection.page_count)
            session = (
                await stack.enter_async_context(
                    self.renderer.session(command.source_path)
                )
                if plan.needs_rendering
                else None
            )
            yield DocumentExtraction(plan=plan, pdf=pdf, renderer=session)

    async def _plan(
        self,
        command: ExtractDocumentTextCommand,
        pdf: PdfHandle,
        *,
        page_count: int,
    ) -> DocumentExtractionPlanDTO:
        probes: list[TextLayerProbe] = []
        for number in range(1, page_count + 1):
            page = await pdf.read_page_text(number)
            probes.append(page.probe)
        verdicts = self.policy.plan(probes).verdicts
        return DocumentExtractionPlanDTO(
            document_id=command.document_id,
            page_count=page_count,
            pages=tuple(
                PagePlanEntryDTO(
                    number=verdict.page_number,
                    method=verdict.decision,
                    reasons=verdict.reasons,
                )
                for verdict in verdicts
            ),
            text_layer_probe=await pdf.probe(),
        )
