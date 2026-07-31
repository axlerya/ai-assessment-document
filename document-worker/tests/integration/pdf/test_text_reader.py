"""Чтение текстового слоя: слова, координаты и пригодность слоя.

Слой бывает формально непустым и при этом бесполезным — по этим признакам
и принимается решение отправить страницу в распознавание.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.application.errors import (
    CorruptedDocumentError,
    EncryptedDocumentError,
)
from document_worker.application.ports.pdf import PdfDocumentReader
from document_worker.infrastructure.pdf.pdfplumber_text_reader import (
    PdfPlumberDocumentReader,
)
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.infrastructure.cpu.executor import CpuPool

pytestmark = pytest.mark.integration


@pytest.fixture
def reader(cpu_pool: CpuPool) -> PdfPlumberDocumentReader:
    return PdfPlumberDocumentReader(pool=cpu_pool)


def test_reader_satisfies_its_port(reader: PdfPlumberDocumentReader) -> None:
    assert isinstance(reader, PdfDocumentReader)


async def test_reads_text_of_digital_pdf_without_loss(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        page = await handle.read_page_text(1)

    for line in pdf_builder.DEFAULT_LINES:
        assert line in page.text.replace("\n", " ")


async def test_words_carry_their_place_on_the_page(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        page = await handle.read_page_text(1)

    assert page.words
    assert {word.text for word in page.words} >= {"SUPPLY", "CONTRACT"}


async def test_word_boxes_are_normalized_to_unit_range(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    # Пиксели рендера и точки PDF несопоставимы, поэтому наружу уходят доли
    # страницы: DPI за границу адаптера не протекает.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        page = await handle.read_page_text(1)

    for word in page.words:
        assert 0.0 <= word.bbox.x0 < word.bbox.x1 <= 1.0
        assert 0.0 <= word.bbox.y0 < word.bbox.y1 <= 1.0


async def test_line_numbers_grow_down_the_page(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        page = await handle.read_page_text(1)

    by_line = {word.line_number for word in page.words}
    assert by_line == {1, 2, 3}


async def test_two_column_page_keeps_columns_apart(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_two_column_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        page = await handle.read_page_text(1)

    left = [word for word in page.words if word.text == "left"]
    right = [word for word in page.words if word.text == "right"]
    assert left
    assert right
    assert max(word.bbox.x1 for word in left) < min(word.bbox.x0 for word in right)


async def test_probe_reports_no_broken_fonts_for_clean_document(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        probe = await handle.probe()

    assert probe.fonts_without_tounicode == 0
    assert probe.unmapped_char_ratio == 0.0
    assert probe.char_count > 0


async def test_probe_detects_fonts_without_tounicode(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    # Символ из такого шрифта в текст не восстановить: слой формально есть,
    # а читать в нём нечего.
    path = pdf_builder.make_broken_tounicode_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        probe = await handle.probe()

    assert probe.fonts_without_tounicode == 1


async def test_probe_detects_glued_text_without_spaces(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_glued_text_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        probe = await handle.probe()

    assert probe.glued_word_ratio == 1.0


async def test_probe_of_scanned_document_reports_no_text(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        probe = await handle.probe()

    assert probe.char_count == 0


async def test_probe_of_page_without_resources_reports_nothing(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    # Пустая страница без словаря ресурсов встречается в сшитых документах.
    path = pdf_builder.make_blank_page_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        probe = await handle.probe()

    assert probe.fonts_without_tounicode == 0
    assert probe.char_count == 0


async def test_reader_rejects_a_page_that_does_not_exist(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with reader.open(path) as handle:
        with pytest.raises(IndexError):
            await handle.read_page_text(2)


async def test_encrypted_document_raises_permanent_error(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_encrypted_pdf(tmp_path / "doc.pdf")

    with pytest.raises(EncryptedDocumentError):
        await _open_and_close(reader, path)


async def test_unreadable_document_raises_permanent_error(
    reader: PdfPlumberDocumentReader,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_truncated_pdf(tmp_path / "doc.pdf")

    with pytest.raises(CorruptedDocumentError):
        await _open_and_close(reader, path)


async def _open_and_close(reader: PdfPlumberDocumentReader, path: Path) -> None:
    async with reader.open(path):
        pass
