"""Рендер страниц в PNG.

Запрошенный DPI снижается, если страница не помещается в предел по пикселям:
лист A0 при 400 DPI это гигабайт растра, и рабочий процесс на нём умрёт молча,
без ошибки, которую можно было бы записать в отказ.

Встроенное изображение as-is не извлекается: типовой скан лежит в CCITT G4 или
JBIG2, декодер есть не у всякой сборки, а падение на этом пути дороже
безусловного рендера.
"""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pypdfium2

from document_worker.application.dto.pdf import RenderedPageDTO
from document_worker.application.errors import PageRenderError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from PIL import Image

    from document_worker.infrastructure.cpu.executor import CpuPool

POINTS_PER_INCH: Final[int] = 72
MIN_DPI: Final[int] = 72


def render_page(path: str, number: int, dpi: int, max_pixels: int) -> RenderedPageDTO:
    """Рендерит страницу. Выполняется в рабочем процессе."""
    document = pypdfium2.PdfDocument(path)
    try:
        page = document[number - 1]
        effective = _fit_dpi(page.get_width(), page.get_height(), dpi, max_pixels)
        bitmap = page.render(scale=effective / POINTS_PER_INCH)
        image = bitmap.to_pil()
        return RenderedPageDTO(
            number=number,
            png=_encode(image),
            width_px=image.width,
            height_px=image.height,
            dpi=effective,
        )
    finally:
        document.close()


def _fit_dpi(width_pt: float, height_pt: float, dpi: int, max_pixels: int) -> int:
    pixels = (width_pt / POINTS_PER_INCH * dpi) * (height_pt / POINTS_PER_INCH * dpi)
    if pixels <= max_pixels:
        return dpi
    reduced = int(dpi * math.sqrt(max_pixels / pixels))
    return max(reduced, MIN_DPI)


def _encode(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    # Метаданные не пишутся: без них байты страницы воспроизводимы, и по их
    # хешу видно, что рендер не поехал между версиями.
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class PdfiumRenderSession:
    """Открытая на рендер сессия документа."""

    path: Path
    pool: CpuPool
    max_pixels: int

    async def render(self, number: int, *, dpi: int) -> RenderedPageDTO:
        """Рендерит страницу, снижая разрешение при превышении предела.

        Raises:
            PageRenderError: Страницу отрендерить не удалось.
        """
        try:
            return await self.pool.run(
                render_page, str(self.path), number, dpi, self.max_pixels
            )
        except (pypdfium2.PdfiumError, OSError) as error:
            # Отказ одной страницы не роняет документ: обработка продолжается,
            # а страница уходит в отказ со своей причиной.
            raise PageRenderError(
                "страницу не удалось отрендерить",
                page_number=number,
                context={"path": str(self.path), "dpi": dpi},
            ) from error


@dataclass(frozen=True, slots=True)
class PdfiumPageRenderer:
    """Рендер страниц поверх pypdfium2."""

    pool: CpuPool
    max_pixels: int

    @contextlib.asynccontextmanager
    async def session(self, path: Path) -> AsyncIterator[PdfiumRenderSession]:
        """Открывает сессию рендера документа."""
        yield PdfiumRenderSession(path=path, pool=self.pool, max_pixels=self.max_pixels)
