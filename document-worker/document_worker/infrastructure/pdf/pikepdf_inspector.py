"""Инспекция PDF: страницы, шифрование, ремонт.

Всё решается до чтения содержимого. Документ, который не открылся или оказался
больше предела, дальше не идёт — незачем скачивать его страницы в память.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pikepdf

from document_worker.application.dto.pdf import PdfInspectionDTO, PdfPageGeometryDTO
from document_worker.application.errors import (
    CorruptedDocumentError,
    PageLimitExceededError,
    UnsupportedMediaTypeError,
)
from document_worker.infrastructure.pdf.errors import translate_pdf_error
from document_worker.infrastructure.pdf.magic import PDF_MIME_TYPE, is_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.infrastructure.cpu.executor import CpuPool

FULL_TURN_DEGREES = 360


def inspect_file(path: str) -> PdfInspectionDTO:
    """Разбирает документ. Выполняется в рабочем процессе, поэтому свободна."""
    repaired = False
    try:
        pdf = pikepdf.open(path, attempt_recovery=False)
    except pikepdf.PasswordError:
        raise
    except pikepdf.PdfError:
        # Сломана только таблица ссылок — qpdf пересканирует объекты.
        pdf = pikepdf.open(path, attempt_recovery=True)
        repaired = True

    with pdf:
        return PdfInspectionDTO(
            page_count=len(pdf.pages),
            is_encrypted=pdf.is_encrypted,
            was_repaired=repaired,
            pages=tuple(
                _geometry_of(page, number)
                for number, page in enumerate(pdf.pages, start=1)
            ),
        )


def _geometry_of(page: pikepdf.Page, number: int) -> PdfPageGeometryDTO:
    box = [float(value) for value in page.mediabox]
    rotation = int(page.obj.get("/Rotate", 0)) % FULL_TURN_DEGREES
    return PdfPageGeometryDTO(
        number=number,
        width_pt=box[2] - box[0],
        height_pt=box[3] - box[1],
        rotation=rotation,
    )


@dataclass(frozen=True, slots=True)
class PikePdfInspector:
    """Инспектор поверх pikepdf. Разбор идёт в отдельном процессе."""

    pool: CpuPool
    max_pages: int

    async def inspect(self, path: Path) -> PdfInspectionDTO:
        """Читает число страниц, геометрию и признаки защиты.

        Raises:
            UnsupportedMediaTypeError: Содержимое не PDF, что бы ни говорило имя.
            CorruptedDocumentError: Документ не читается или пуст.
            EncryptedDocumentError: Документ защищён паролем.
            PageLimitExceededError: Страниц больше допустимого.
        """
        # Проверки по файлу синхронны и стоят одного системного вызова.
        if not path.is_file():  # noqa: ASYNC240
            raise CorruptedDocumentError(
                "файла документа нет на диске",
                context={"path": str(path)},
            )
        if not is_pdf(path):
            raise UnsupportedMediaTypeError(
                "содержимое файла не является PDF",
                context={"path": str(path), "expected": PDF_MIME_TYPE},
            )
        inspection = await self._inspect(path)
        if inspection.page_count == 0:
            raise CorruptedDocumentError(
                "в документе нет ни одной страницы",
                context={"path": str(path)},
            )
        if inspection.page_count > self.max_pages:
            raise PageLimitExceededError(
                "страниц больше допустимого предела",
                context={"pages": inspection.page_count, "limit": self.max_pages},
            )
        return inspection

    async def _inspect(self, path: Path) -> PdfInspectionDTO:
        try:
            return await self.pool.run(inspect_file, str(path))
        except (pikepdf.PasswordError, pikepdf.PdfError, OSError) as error:
            raise translate_pdf_error(error, path=str(path)) from error
