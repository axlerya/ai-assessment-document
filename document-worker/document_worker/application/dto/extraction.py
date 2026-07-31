"""План извлечения текста и открытые на время обработки ресурсы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.value_objects.enums import ExtractionMethod

if TYPE_CHECKING:
    from document_worker.application.dto.pdf import TextLayerProbeDTO
    from document_worker.application.ports.pdf import PdfHandle, RenderSession
    from document_worker.domain.value_objects.identifiers import DocumentId
    from document_worker.domain.value_objects.paging import PageNumber


@dataclass(frozen=True, slots=True)
class PagePlanEntryDTO:
    """Чем читать одну страницу и почему именно так."""

    number: PageNumber
    method: ExtractionMethod
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentExtractionPlanDTO:
    """Способ чтения каждой страницы документа."""

    document_id: DocumentId
    page_count: int
    pages: tuple[PagePlanEntryDTO, ...]
    text_layer_probe: TextLayerProbeDTO

    @property
    def pages_text_layer(self) -> int:
        """Сколько страниц читается из текстового слоя."""
        return self._count(ExtractionMethod.TEXT_LAYER)

    @property
    def pages_ocr(self) -> int:
        """Сколько страниц уходит в распознавание."""
        return self._count(ExtractionMethod.OCR)

    @property
    def pages_hybrid(self) -> int:
        """Сколько страниц читается обоими способами сразу."""
        return self._count(ExtractionMethod.HYBRID)

    @property
    def needs_rendering(self) -> bool:
        """Нужна ли хоть одной странице картинка."""
        return any(entry.method.is_ocr_based for entry in self.pages)

    def _count(self, method: ExtractionMethod) -> int:
        return sum(1 for entry in self.pages if entry.method is method)


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    """План вместе с ресурсами, которые живут ровно столько же.

    Сессия рендера отсутствует, если распознавать нечего: открытая сессия
    держит документ в памяти рабочего процесса и ничего не даёт взамен.
    """

    plan: DocumentExtractionPlanDTO
    pdf: PdfHandle
    renderer: RenderSession | None
