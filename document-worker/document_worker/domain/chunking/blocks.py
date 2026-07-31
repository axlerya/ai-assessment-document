"""Смысловые блоки страницы.

Блок не пересекает границу страницы по построению: склеивать нечего, потому
что чанк всё равно не имеет права выйти за страницу — иначе у него перестают
быть определёнными способ извлечения и средняя уверенность.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from document_worker.domain.chunking.line_classifier import LineKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from document_worker.domain.value_objects.text import TextSpan


class BlockKind(StrEnum):
    """Чем является блок для сборки чанков."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    REQUISITES = "requisites"
    SIGNATURE = "signature"

    @property
    def is_atomic(self) -> bool:
        """Нельзя ли резать блок и склеивать его с соседями."""
        return self in _ATOMIC_KINDS


@dataclass(frozen=True, slots=True)
class Block:
    """Непрерывный кусок текста страницы внутри одной секции."""

    span: TextSpan
    kind: BlockKind
    section_key: str
    heading_path: tuple[str, ...]

    @property
    def is_atomic(self) -> bool:
        """Таблицы, реквизиты и подписи не режутся и не склеиваются."""
        return self.kind.is_atomic


BLOCK_KIND_BY_LINE: Final[Mapping[LineKind, BlockKind]] = {
    LineKind.APPENDIX: BlockKind.HEADING,
    LineKind.HEADING_PART: BlockKind.HEADING,
    LineKind.HEADING_ARTICLE: BlockKind.HEADING,
    LineKind.HEADING_UPPER: BlockKind.HEADING,
    LineKind.TABLE_ROW: BlockKind.TABLE,
    LineKind.TABLE_SEPARATOR: BlockKind.TABLE,
    LineKind.REQUISITES: BlockKind.REQUISITES,
    LineKind.SIGNATURE: BlockKind.SIGNATURE,
    LineKind.CLAUSE: BlockKind.LIST_ITEM,
    LineKind.SUBCLAUSE_PAREN: BlockKind.LIST_ITEM,
    LineKind.SUBCLAUSE_LETTER: BlockKind.LIST_ITEM,
    LineKind.BULLET: BlockKind.LIST_ITEM,
    LineKind.TEXT: BlockKind.PARAGRAPH,
}

# Строки, которые разрывают блок и не входят ни в один спан: их символы просто
# не покрыты, поэтому смещения соседей не сдвигаются.
SEPARATING_KINDS: Final[frozenset[LineKind]] = frozenset(
    {LineKind.BLANK, LineKind.PAGE_ARTIFACT}
)

_ATOMIC_KINDS: Final[frozenset[BlockKind]] = frozenset(
    {BlockKind.TABLE, BlockKind.REQUISITES, BlockKind.SIGNATURE}
)
