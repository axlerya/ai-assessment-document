"""План извлечения: чем и как читать каждую страницу.

Use case ни одной цифры не решает сам — пороги живут в доменной политике,
а признаки страницы приходят из адаптера. Здесь проверяется, что решение
доходит до плана неискажённым и что сессия рендера открывается только тогда,
когда её есть кому использовать.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.application.dto.commands import ExtractDocumentTextCommand
from document_worker.application.errors import (
    CorruptedDocumentError,
    EncryptedDocumentError,
    PageLimitExceededError,
    UnsupportedMediaTypeError,
)
from document_worker.application.use_cases.extract_document_text import (
    ExtractDocumentText,
)
from document_worker.domain.policies.text_layer_quality import TextLayerQualityPolicy
from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.infrastructure.pdf.pdfplumber_text_reader import (
    PdfPlumberDocumentReader,
)
from document_worker.infrastructure.pdf.pikepdf_inspector import PikePdfInspector
from document_worker.infrastructure.pdf.pypdfium2_page_renderer import (
    PdfiumPageRenderer,
)
from tests.factories import new_correlation_id
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.domain.value_objects.identifiers import DocumentId
    from document_worker.infrastructure.cpu.executor import CpuPool

pytestmark = pytest.mark.integration

MAX_PAGES = 8
MAX_PIXELS = 4_000_000


@pytest.fixture
def extract(cpu_pool: CpuPool) -> ExtractDocumentText:
    """Use case поверх настоящих PDF-адаптеров."""
    return ExtractDocumentText(
        inspector=PikePdfInspector(pool=cpu_pool, max_pages=MAX_PAGES),
        reader=PdfPlumberDocumentReader(pool=cpu_pool),
        renderer=PdfiumPageRenderer(pool=cpu_pool, max_pixels=MAX_PIXELS),
        policy=TextLayerQualityPolicy(),
    )


@pytest.fixture
def read_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Пути, которые дошли до разбора текстового слоя."""
    calls: list[Path] = []
    original = PdfPlumberDocumentReader.open

    def spy(self: PdfPlumberDocumentReader, path: Path) -> object:
        calls.append(path)
        return original(self, path)

    monkeypatch.setattr(PdfPlumberDocumentReader, "open", spy)
    return calls


def command_for(document_id: DocumentId, path: Path) -> ExtractDocumentTextCommand:
    """Команда извлечения для указанного файла."""
    return ExtractDocumentTextCommand(
        document_id=document_id,
        correlation_id=new_correlation_id(),
        source_path=path,
    )


async def test_plan_covers_every_page_of_the_document(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=3)

    async with extract.execute(command_for(document_id, path)) as extraction:
        plan = extraction.plan

    assert plan.page_count == 3
    assert [int(entry.number) for entry in plan.pages] == [1, 2, 3]


async def test_clean_text_document_is_planned_for_its_text_layer(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=2)

    async with extract.execute(command_for(document_id, path)) as extraction:
        plan = extraction.plan

    assert plan.pages_text_layer == 2
    assert plan.pages_ocr == 0
    assert plan.pages_hybrid == 0


async def test_scanned_document_is_planned_for_recognition(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    async with extract.execute(command_for(document_id, path)) as extraction:
        plan = extraction.plan

    assert plan.pages_ocr == 1
    assert all(entry.method is ExtractionMethod.OCR for entry in plan.pages)


async def test_plan_keeps_the_reason_a_page_left_its_text_layer(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    # Без причины страница в отчёте выглядит отправленной на OCR произвольно.
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    async with extract.execute(command_for(document_id, path)) as extraction:
        plan = extraction.plan

    assert plan.pages[0].reasons


async def test_render_session_is_not_opened_for_a_pure_text_document(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    # Открытая сессия держит документ в памяти рабочего процесса, а рендерить
    # в текстовом документе нечего.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=2)

    async with extract.execute(command_for(document_id, path)) as extraction:
        assert extraction.renderer is None


async def test_render_session_is_opened_when_a_page_needs_recognition(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    async with extract.execute(command_for(document_id, path)) as extraction:
        assert extraction.renderer is not None


async def test_plan_reports_document_wide_signs_of_a_broken_layer(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    # Шрифт без `/ToUnicode` объясняет, почему страницы уехали на распознавание.
    path = pdf_builder.make_broken_tounicode_pdf(tmp_path / "doc.pdf")

    async with extract.execute(command_for(document_id, path)) as extraction:
        plan = extraction.plan

    assert plan.text_layer_probe.fonts_without_tounicode == 1


async def test_pages_of_the_document_are_readable_through_the_extraction(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with extract.execute(command_for(document_id, path)) as extraction:
        page = await extraction.pdf.read_page_text(1)

    assert pdf_builder.DEFAULT_LINES[0] in page.text.replace("\n", " ")


async def test_document_without_pages_is_corrupted(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_empty_pdf(tmp_path / "doc.pdf")

    with pytest.raises(CorruptedDocumentError):
        await _drain(extract, command_for(document_id, path))


async def test_file_that_is_not_a_pdf_is_rejected(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    # Заявленный тип проверен до скачивания, а содержимое — только здесь.
    path = pdf_builder.make_non_pdf_file(tmp_path / "doc.pdf")

    with pytest.raises(UnsupportedMediaTypeError):
        await _drain(extract, command_for(document_id, path))


async def test_document_above_the_page_limit_is_rejected(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=MAX_PAGES + 1)

    with pytest.raises(PageLimitExceededError):
        await _drain(extract, command_for(document_id, path))


async def test_encrypted_document_is_rejected(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_encrypted_pdf(tmp_path / "doc.pdf")

    with pytest.raises(EncryptedDocumentError):
        await _drain(extract, command_for(document_id, path))


async def test_page_limit_is_checked_before_the_document_is_parsed(
    extract: ExtractDocumentText,
    document_id: DocumentId,
    tmp_path: Path,
    read_calls: list[Path],
) -> None:
    # Разбор трёхсот лишних страниц ради отказа по их числу — потраченная
    # память рабочего процесса.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=MAX_PAGES + 1)

    with pytest.raises(PageLimitExceededError):
        await _drain(extract, command_for(document_id, path))

    assert read_calls == []


async def _drain(
    extract: ExtractDocumentText,
    command: ExtractDocumentTextCommand,
) -> None:
    async with extract.execute(command) as extraction:
        del extraction
