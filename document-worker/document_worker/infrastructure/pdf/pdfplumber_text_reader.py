"""Чтение текстового слоя со словами и их координатами.

Документ разбирается один раз на всю обработку: pdfplumber строит дерево
объектов заново на каждое открытие, и по разу на страницу это дороже самого
чтения. Результат возвращается из рабочего процесса целиком — текст трёхсот
страниц это единицы мегабайт, а картинок в нём нет.
"""

from __future__ import annotations

import contextlib
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import pdfplumber
import pikepdf

from document_worker.application.dto.pdf import (
    PdfPageTextDTO,
    PdfWordDTO,
    TextLayerProbeDTO,
)
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.infrastructure.pdf.errors import translate_pdf_error

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from document_worker.infrastructure.cpu.executor import CpuPool

# Базовые четырнадцать шрифтов мапятся встроенными таблицами, `/ToUnicode`
# им не нужен, и считать их сломанными нельзя.
STANDARD_FONTS: Final[frozenset[str]] = frozenset(
    {
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-Oblique",
        "Helvetica-BoldOblique",
        "Courier",
        "Courier-Bold",
        "Courier-Oblique",
        "Courier-BoldOblique",
        "Symbol",
        "ZapfDingbats",
    }
)

# Слово длиннее этого в юридическом тексте не встречается: столько символов
# без пробела означает потерянные разделители, а не длинное слово.
GLUED_WORD_LENGTH: Final[int] = 25
UNMAPPED_CATEGORIES: Final[frozenset[str]] = frozenset({"Cc", "Cf", "Cn", "Co", "Cs"})
LINE_TOLERANCE_PT: Final[float] = 3.0


@dataclass(frozen=True, slots=True)
class DocumentText:
    """Разобранный документ: страницы и признаки качества слоя."""

    pages: tuple[PdfPageTextDTO, ...]
    probe: TextLayerProbeDTO


def read_document(path: str) -> DocumentText:
    """Разбирает документ целиком. Выполняется в рабочем процессе."""
    fonts_without_tounicode = _count_fonts_without_tounicode(path)
    with pdfplumber.open(path) as pdf:
        pages = tuple(
            _read_page(page, number) for number, page in enumerate(pdf.pages, start=1)
        )
    return DocumentText(pages=pages, probe=_probe(pages, fonts_without_tounicode))


def _count_fonts_without_tounicode(path: str) -> int:
    seen: set[str] = set()
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            for name, font in _fonts_of(page).items():
                base = str(font.get("/BaseFont", "")).lstrip("/").split("+")[-1]
                if base in STANDARD_FONTS or "/ToUnicode" in font:
                    continue
                seen.add(f"{name}:{base}")
    return len(seen)


def _fonts_of(page: pikepdf.Page) -> dict[str, pikepdf.Object]:
    # Словаря ресурсов у страницы может не быть вовсе: он наследуется от узла
    # Pages, а у пустой страницы шрифтов нет и там.
    resources = page.obj.get("/Resources", pikepdf.Dictionary())
    fonts = resources.get("/Font", pikepdf.Dictionary())
    return {str(name): font for name, font in fonts.items()}


def _read_page(page: Any, number: int) -> PdfPageTextDTO:
    width = float(page.width)
    height = float(page.height)
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    return PdfPageTextDTO(
        number=number,
        text=page.extract_text() or "",
        words=tuple(_word(word, width, height) for word in _numbered(words)),
    )


def _numbered(words: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    """Расставляет номера строк по вертикальной координате."""
    numbered: list[tuple[dict[str, Any], int]] = []
    line_number = 0
    previous_top: float | None = None
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        top = float(word["top"])
        if previous_top is None or abs(top - previous_top) > LINE_TOLERANCE_PT:
            line_number += 1
            previous_top = top
        numbered.append((word, line_number))
    return numbered


def _word(item: tuple[dict[str, Any], int], width: float, height: float) -> PdfWordDTO:
    word, line_number = item
    return PdfWordDTO(
        text=str(word["text"]),
        bbox=BoundingBox(
            _unit(float(word["x0"]), width),
            _unit(float(word["top"]), height),
            _unit(float(word["x1"]), width),
            _unit(float(word["bottom"]), height),
        ),
        line_number=line_number,
    )


def _unit(value: float, size: float) -> float:
    return min(max(value / size, 0.0), 1.0)


def _probe(
    pages: Sequence[PdfPageTextDTO], fonts_without_tounicode: int
) -> TextLayerProbeDTO:
    characters = "".join(page.text for page in pages)
    words = [word for page in pages for word in page.words]
    # Переводы строк и табуляции формально относятся к управляющим символам,
    # но это разметка текста, а не непрочитанные глифы.
    unmapped = sum(
        1
        for character in characters
        if not character.isspace()
        and unicodedata.category(character) in UNMAPPED_CATEGORIES
    )
    glued = sum(1 for word in words if len(word.text) > GLUED_WORD_LENGTH)
    return TextLayerProbeDTO(
        char_count=len(characters),
        unmapped_char_ratio=unmapped / len(characters) if characters else 0.0,
        glued_word_ratio=glued / len(words) if words else 0.0,
        fonts_without_tounicode=fonts_without_tounicode,
    )


@dataclass(frozen=True, slots=True)
class PdfPlumberHandle:
    """Открытый документ. Страницы уже прочитаны, обращения к диску нет."""

    document: DocumentText

    async def read_page_text(self, number: int) -> PdfPageTextDTO:
        """Читает текстовый слой одной страницы.

        Raises:
            IndexError: Страницы с таким номером в документе нет.
        """
        if not 1 <= number <= len(self.document.pages):
            msg = f"страницы {number} в документе нет"
            raise IndexError(msg)
        return self.document.pages[number - 1]

    async def probe(self) -> TextLayerProbeDTO:
        """Признаки пригодности текстового слоя."""
        return self.document.probe


@dataclass(frozen=True, slots=True)
class PdfPlumberDocumentReader:
    """Читатель текстового слоя поверх pdfplumber."""

    pool: CpuPool

    @contextlib.asynccontextmanager
    async def open(self, path: Path) -> AsyncIterator[PdfPlumberHandle]:
        """Открывает документ на чтение текста.

        Raises:
            CorruptedDocumentError: Документ не читается.
            EncryptedDocumentError: Документ защищён паролем.
        """
        try:
            document = await self.pool.run(read_document, str(path))
        except (pikepdf.PasswordError, pikepdf.PdfError, OSError) as error:
            raise translate_pdf_error(error, path=str(path)) from error
        yield PdfPlumberHandle(document=document)
