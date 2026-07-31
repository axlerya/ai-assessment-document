"""Дерево секций документа.

Стек секций — единственное состояние, которое переносится через границу
страницы. Именно он связывает части секции, идущей с третьей страницы по
седьмую: все её чанки получают одинаковый heading_path, и downstream собирает
их одним запросом, не храня ссылок на соседей.

Одинаковый heading_path у двух разных секций (вторая «Статья 5») даёт разные
ключи с числовым суффиксом, но одну и ту же хлебную крошку: суффикс — деталь
дерева, а не то, что показывают оператору.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from document_worker.domain.chunking.line_classifier import LineKind
from document_worker.domain.chunking.structure_rules import MAX_HEADING_PATH_DEPTH

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from document_worker.domain.chunking.line_classifier import LayoutLine

ROOT_KEY: Final[str] = "document"

# Ключевые слова, открывающие секцию верхнего уровня; глава и подраздел из того
# же правила вкладываются в неё.
TOP_LEVEL_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"РАЗДЕЛ", "ЧАСТЬ", "Раздел", "Часть"}
)

RANK_ROOT: Final[int] = 0
RANK_APPENDIX: Final[int] = 1
RANK_PART: Final[int] = 2
RANK_CHAPTER: Final[int] = 3
RANK_ARTICLE: Final[int] = 4
RANK_UPPER: Final[int] = 5
# Пункт «1.2.3» глубже капс-заголовка ровно на число своих уровней.
RANK_CLAUSE_BASE: Final[int] = 5


class SectionKind(StrEnum):
    """Чем открыта секция."""

    DOCUMENT = "document"
    APPENDIX = "appendix"
    PART = "part"
    ARTICLE = "article"
    HEADING = "heading"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"


@dataclass(frozen=True, slots=True)
class SectionNode:
    """Секция, которой принадлежит строка."""

    key: str
    kind: SectionKind
    rank: int
    heading_path: tuple[str, ...]


ROOT_SECTION: Final[SectionNode] = SectionNode(
    key=ROOT_KEY, kind=SectionKind.DOCUMENT, rank=RANK_ROOT, heading_path=()
)

_SECTION_KINDS: Final[Mapping[LineKind, SectionKind]] = {
    LineKind.APPENDIX: SectionKind.APPENDIX,
    LineKind.HEADING_PART: SectionKind.PART,
    LineKind.HEADING_ARTICLE: SectionKind.ARTICLE,
    LineKind.HEADING_UPPER: SectionKind.HEADING,
    LineKind.CLAUSE: SectionKind.CLAUSE,
    LineKind.SUBCLAUSE_PAREN: SectionKind.SUBCLAUSE,
    LineKind.SUBCLAUSE_LETTER: SectionKind.SUBCLAUSE,
}

_FIXED_RANKS: Final[Mapping[LineKind, int]] = {
    LineKind.APPENDIX: RANK_APPENDIX,
    LineKind.HEADING_ARTICLE: RANK_ARTICLE,
    LineKind.HEADING_UPPER: RANK_UPPER,
}


class StructureDetector:
    """Собирает дерево секций по размеченным строкам всего документа."""

    def detect(self, lines: Sequence[LayoutLine]) -> tuple[SectionNode, ...]:
        """Возвращает секцию каждой строки в порядке следования строк."""
        stack: list[SectionNode] = [ROOT_SECTION]
        used_keys: dict[str, int] = {}
        sections: list[SectionNode] = []
        for line in lines:
            if line.kind.opens_section:
                _open_section(line, stack=stack, used_keys=used_keys)
            sections.append(stack[-1])
        return tuple(sections)


def _open_section(
    line: LayoutLine,
    *,
    stack: list[SectionNode],
    used_keys: dict[str, int],
) -> None:
    rank = _rank_of(line, stack[-1])
    # Приложение даёт ранг 1 и закрывает стек до корня: оно не может стать
    # потомком статьи основного текста.
    while len(stack) > 1 and stack[-1].rank >= rank:
        stack.pop()
    heading_path = _capped((*stack[-1].heading_path, _label_of(line)))
    stack.append(
        SectionNode(
            key=_unique_key(heading_path, used_keys),
            kind=_SECTION_KINDS[line.kind],
            rank=rank,
            heading_path=heading_path,
        )
    )


def _rank_of(line: LayoutLine, current: SectionNode) -> int:
    fixed = _FIXED_RANKS.get(line.kind)
    if fixed is not None:
        return fixed
    if line.kind is LineKind.HEADING_PART:
        # Приложение даёт ранг 1 и закрывает стек до корня.
        return RANK_PART if line.keyword in TOP_LEVEL_KEYWORDS else RANK_CHAPTER
    if line.kind is LineKind.CLAUSE:
        return RANK_CLAUSE_BASE + (line.number or "").count(".") + 1
    # Подпункт вложен в свой пункт, каким бы глубоким тот ни был; соседний
    # подпункт остаётся соседом, а не потомком предыдущего.
    if current.kind is SectionKind.SUBCLAUSE:
        return current.rank
    return current.rank + 1


def _label_of(line: LayoutLine) -> str:
    if line.kind is LineKind.HEADING_UPPER:
        return line.title or line.text.strip()
    if line.keyword is None:
        return line.number or line.text.strip()
    return f"{line.keyword} {line.number}" if line.number else line.keyword


def _capped(path: tuple[str, ...]) -> tuple[str, ...]:
    """Ограничивает глубину, сохраняя верхние и нижние уровни.

    Хлебная крошка на двенадцать уровней раздувает jsonb каждой строки чанка,
    а на вопрос «где это в документе» отвечает уже верхом и низом.
    """
    if len(path) <= MAX_HEADING_PATH_DEPTH:
        return path
    head = MAX_HEADING_PATH_DEPTH // 2
    return (*path[:head], *path[len(path) - (MAX_HEADING_PATH_DEPTH - head) :])


def _unique_key(heading_path: tuple[str, ...], used_keys: dict[str, int]) -> str:
    base = " > ".join(heading_path)
    seen = used_keys.get(base, 0) + 1
    used_keys[base] = seen
    return base if seen == 1 else f"{base}#{seen}"
