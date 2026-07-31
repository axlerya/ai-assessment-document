"""Инспекция PDF: страницы, шифрование, ремонт, сигнатура.

Всё, что здесь проверяется, решается до чтения содержимого — и определяет,
стоит ли вообще браться за документ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.application.errors import (
    CorruptedDocumentError,
    EncryptedDocumentError,
    PageLimitExceededError,
    UnsupportedMediaTypeError,
)
from document_worker.application.ports.pdf import PdfInspector
from document_worker.infrastructure.pdf.pikepdf_inspector import PikePdfInspector
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.infrastructure.cpu.executor import CpuPool

pytestmark = pytest.mark.integration

PAGE_LIMIT = 300


@pytest.fixture
def inspector(cpu_pool: CpuPool) -> PikePdfInspector:
    return PikePdfInspector(pool=cpu_pool, max_pages=PAGE_LIMIT)


def test_inspector_satisfies_its_port(inspector: PikePdfInspector) -> None:
    assert isinstance(inspector, PdfInspector)


async def test_page_count_of_generated_document(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=3)

    inspection = await inspector.inspect(path)

    assert inspection.page_count == 3
    assert len(inspection.pages) == 3


async def test_page_geometry_is_reported_in_points(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    inspection = await inspector.inspect(path)

    page = inspection.pages[0]
    assert (page.width_pt, page.height_pt) == (
        float(pdf_builder.PAGE_WIDTH),
        float(pdf_builder.PAGE_HEIGHT),
    )
    assert page.rotation == 0


async def test_encrypted_with_user_password_raises_permanent_error(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    # Пароля у сервиса нет и не будет: повтор ничего не изменит.
    path = pdf_builder.make_encrypted_pdf(tmp_path / "doc.pdf")

    with pytest.raises(EncryptedDocumentError):
        await inspector.inspect(path)


async def test_owner_only_encrypted_document_is_opened(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    # Владельческий пароль ограничивает печать и правку, а чтение разрешает.
    path = pdf_builder.make_owner_encrypted_pdf(tmp_path / "doc.pdf")

    inspection = await inspector.inspect(path)

    assert inspection.page_count == 1
    assert inspection.is_encrypted


async def test_corrupted_document_is_repaired_and_reported(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_corrupted_pdf(tmp_path / "doc.pdf")

    inspection = await inspector.inspect(path)

    assert inspection.page_count == 1
    assert inspection.was_repaired


async def test_unrepairable_document_raises_permanent_error(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_truncated_pdf(tmp_path / "doc.pdf")

    with pytest.raises(CorruptedDocumentError):
        await inspector.inspect(path)


async def test_zero_page_document_raises_permanent_error(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_empty_pdf(tmp_path / "doc.pdf")

    with pytest.raises(CorruptedDocumentError):
        await inspector.inspect(path)


async def test_page_count_above_limit_raises_permanent_error(
    cpu_pool: CpuPool,
    tmp_path: Path,
) -> None:
    inspector = PikePdfInspector(pool=cpu_pool, max_pages=2)
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=3)

    with pytest.raises(PageLimitExceededError):
        await inspector.inspect(path)


async def test_non_pdf_content_with_pdf_extension_is_rejected_by_signature(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    # Расширению верить нельзя: тип определяется содержимым.
    path = pdf_builder.make_non_pdf_file(tmp_path / "doc.pdf")

    with pytest.raises(UnsupportedMediaTypeError):
        await inspector.inspect(path)


async def test_missing_file_raises_permanent_error(
    inspector: PikePdfInspector,
    tmp_path: Path,
) -> None:
    with pytest.raises(CorruptedDocumentError):
        await inspector.inspect(tmp_path / "nowhere.pdf")
