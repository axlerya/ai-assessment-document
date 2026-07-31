"""Данные, которыми PDF-адаптеры отвечают прикладному слою.

Координаты слов нормализованы к долям страницы: пиксели рендера и точки PDF
несопоставимы, а DPI не должен протекать за границу адаптера.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document_worker.domain.value_objects.geometry import BoundingBox


@dataclass(frozen=True, slots=True)
class PdfPageGeometryDTO:
    """Размеры и поворот страницы в точках PDF."""

    number: int
    width_pt: float
    height_pt: float
    rotation: int


@dataclass(frozen=True, slots=True)
class PdfInspectionDTO:
    """Что известно о документе до чтения его содержимого."""

    page_count: int
    is_encrypted: bool
    was_repaired: bool
    pages: tuple[PdfPageGeometryDTO, ...]


@dataclass(frozen=True, slots=True)
class PdfWordDTO:
    """Слово текстового слоя с его местом на странице."""

    text: str
    bbox: BoundingBox
    line_number: int


@dataclass(frozen=True, slots=True)
class PdfPageTextDTO:
    """Текст одной страницы вместе со словами."""

    number: int
    text: str
    words: tuple[PdfWordDTO, ...]


@dataclass(frozen=True, slots=True)
class TextLayerProbeDTO:
    """Признаки, по которым решают, годен ли текстовый слой.

    Слой бывает формально непустым и при этом бесполезным: символы из шрифта
    без `/ToUnicode` превращаются в мусор, а склеенный текст не режется на слова.
    """

    char_count: int
    unmapped_char_ratio: float
    glued_word_ratio: float
    fonts_without_tounicode: int


@dataclass(frozen=True, slots=True)
class RenderedPageDTO:
    """Отрендеренная страница в PNG."""

    number: int
    png: bytes
    width_px: int
    height_px: int
    dpi: int
