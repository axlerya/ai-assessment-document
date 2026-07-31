"""Рендер страниц: размер, воспроизводимость, предел по пикселям."""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from document_worker.application.errors import PageRenderError
from document_worker.application.ports.pdf import PageRenderer
from document_worker.infrastructure.pdf.pypdfium2_page_renderer import (
    POINTS_PER_INCH,
    PdfiumPageRenderer,
)
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.infrastructure.cpu.executor import CpuPool

pytestmark = pytest.mark.integration

MAX_PIXELS = 40_000_000
DPI = 300


@pytest.fixture
def renderer(cpu_pool: CpuPool) -> PdfiumPageRenderer:
    return PdfiumPageRenderer(pool=cpu_pool, max_pixels=MAX_PIXELS)


def test_renderer_satisfies_its_port(renderer: PdfiumPageRenderer) -> None:
    assert isinstance(renderer, PageRenderer)


async def test_render_produces_expected_pixel_size_for_300_dpi(
    renderer: PdfiumPageRenderer,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")
    expected = round(pdf_builder.PAGE_WIDTH / POINTS_PER_INCH * DPI)

    async with renderer.session(path) as session:
        page = await session.render(1, dpi=DPI)

    assert page.dpi == DPI
    assert abs(page.width_px - expected) <= 1
    assert _size_of(page.png) == (page.width_px, page.height_px)


async def test_render_output_is_a_png(
    renderer: PdfiumPageRenderer,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with renderer.session(path) as session:
        page = await session.render(1, dpi=150)

    assert page.png.startswith(b"\x89PNG\r\n\x1a\n")


async def test_render_is_deterministic_between_runs(
    renderer: PdfiumPageRenderer,
    tmp_path: Path,
) -> None:
    # По хешу видно, что рендер не поехал между версиями библиотеки.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with renderer.session(path) as session:
        first = await session.render(1, dpi=150)
        second = await session.render(1, dpi=150)

    assert hashlib.sha256(first.png).digest() == hashlib.sha256(second.png).digest()


async def test_dpi_is_reduced_when_pixel_limit_exceeded(
    cpu_pool: CpuPool,
    tmp_path: Path,
) -> None:
    # Лист при 400 DPI это гигабайт растра: рабочий процесс умрёт молча, без
    # ошибки, которую можно было бы записать в отказ.
    limit = 1_000_000
    renderer = PdfiumPageRenderer(pool=cpu_pool, max_pixels=limit)
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with renderer.session(path) as session:
        page = await session.render(1, dpi=600)

    assert page.dpi < 600
    assert page.width_px * page.height_px <= limit


async def test_dpi_never_falls_below_the_floor(
    cpu_pool: CpuPool,
    tmp_path: Path,
) -> None:
    # Ниже порога распознавать нечего, поэтому предел упирается в пол.
    renderer = PdfiumPageRenderer(pool=cpu_pool, max_pixels=1)
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with renderer.session(path) as session:
        page = await session.render(1, dpi=600)

    assert page.dpi == 72


async def test_render_of_scanned_page_returns_the_picture(
    renderer: PdfiumPageRenderer,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    async with renderer.session(path) as session:
        page = await session.render(1, dpi=150)

    assert _size_of(page.png) == (page.width_px, page.height_px)


async def test_ccitt_scan_is_rendered_instead_of_extracted(
    renderer: PdfiumPageRenderer,
    tmp_path: Path,
) -> None:
    # Встроенное изображение as-is не извлекается: декодер CCITT G4 есть не у
    # всякой сборки, а падение на этом пути дороже безусловного рендера.
    path = pdf_builder.make_ccitt_g4_scan_pdf(tmp_path / "doc.pdf")

    async with renderer.session(path) as session:
        page = await session.render(1, dpi=150)

    assert page.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert page.width_px > 0


async def test_page_that_does_not_exist_is_a_page_level_error(
    renderer: PdfiumPageRenderer,
    tmp_path: Path,
) -> None:
    # Сбой одной страницы не роняет документ: обработка продолжается.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with renderer.session(path) as session:
        with pytest.raises(PageRenderError):
            await session.render(99, dpi=150)


def _size_of(png: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(png)) as image:
        return image.width, image.height
