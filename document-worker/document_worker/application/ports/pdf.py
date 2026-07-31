"""Порты работы с PDF.

Документ открывается один раз на всю обработку: повторный разбор файла на
каждой странице стоит дороже самой страницы.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from document_worker.application.dto.pdf import (
        PdfInspectionDTO,
        PdfPageTextDTO,
        RenderedPageDTO,
        TextLayerProbeDTO,
    )


@runtime_checkable
class PdfInspector(Protocol):
    """Что известно о документе до чтения содержимого."""

    async def inspect(self, path: Path) -> PdfInspectionDTO:
        """Читает число страниц, геометрию и признаки защиты."""
        ...


@runtime_checkable
class PdfHandle(Protocol):
    """Открытый документ."""

    async def read_page_text(self, number: int) -> PdfPageTextDTO:
        """Читает текстовый слой одной страницы."""
        ...

    async def probe(self) -> TextLayerProbeDTO:
        """Оценивает пригодность текстового слоя документа."""
        ...


@runtime_checkable
class PdfDocumentReader(Protocol):
    """Открывает документ на чтение текста."""

    def open(self, path: Path) -> AbstractAsyncContextManager[PdfHandle]:
        """Открывает документ и закрывает его при любом исходе."""
        ...


@runtime_checkable
class RenderSession(Protocol):
    """Открытый на рендер документ."""

    async def render(self, number: int, *, dpi: int) -> RenderedPageDTO:
        """Рендерит страницу, снижая разрешение при превышении предела."""
        ...


@runtime_checkable
class PageRenderer(Protocol):
    """Открывает документ на рендер страниц."""

    def session(self, path: Path) -> AbstractAsyncContextManager[RenderSession]:
        """Открывает сессию рендера и закрывает её при любом исходе."""
        ...
