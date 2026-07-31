"""Сборка размеченных строк страницы в блоки."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.chunking.blocks import (
    BLOCK_KIND_BY_LINE,
    SEPARATING_KINDS,
    Block,
    BlockKind,
)
from document_worker.domain.chunking.line_classifier import LineKind
from document_worker.domain.value_objects.text import TextSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.chunking.line_classifier import LayoutLine
    from document_worker.domain.chunking.structure_detector import SectionNode


@dataclass(frozen=True, slots=True)
class _Group:
    """Строки, которые ещё собираются в один блок."""

    kind: BlockKind
    section: SectionNode
    lines: list[LayoutLine]


class ParagraphSplitter:
    """Режет строки одной страницы на блоки."""

    def split(
        self,
        lines: Sequence[LayoutLine],
        sections: Sequence[SectionNode],
        *,
        content: str,
    ) -> tuple[Block, ...]:
        """Собирает блоки; пустые и служебные строки в спаны не попадают."""
        blocks: list[Block] = []
        group: _Group | None = None
        for line, section in zip(lines, sections, strict=True):
            if line.kind in SEPARATING_KINDS:
                _flush(group, blocks, content)
                group = None
                continue
            if group is not None and not _continues(group, line, section):
                _flush(group, blocks, content)
                group = None
            if group is None:
                group = _Group(BLOCK_KIND_BY_LINE[line.kind], section, [line])
            else:
                group.lines.append(line)
        _flush(group, blocks, content)
        return tuple(blocks)


def _continues(group: _Group, line: LayoutLine, section: SectionNode) -> bool:
    """Продолжает ли строка текущий блок."""
    if section.key != group.section.key or group.kind is BlockKind.HEADING:
        return False
    if group.kind is BlockKind.LIST_ITEM:
        # Продолжение пункта — строка с бо́льшим отступом, а не следующий пункт.
        return line.kind is LineKind.TEXT and line.indent > group.lines[0].indent
    return BLOCK_KIND_BY_LINE[line.kind] is group.kind


def _flush(group: _Group | None, blocks: list[Block], content: str) -> None:
    if group is None:
        return
    # Пробельная строка всегда BLANK и в группу не попадает, поэтому обрезанный
    # спан группы непуст по построению.
    span = _trimmed(
        TextSpan(group.lines[0].span.start, group.lines[-1].span.end), content
    )
    blocks.append(
        Block(
            span=span,
            kind=group.kind,
            section_key=group.section.key,
            heading_path=group.section.heading_path,
        )
    )


def _trimmed(span: TextSpan, content: str) -> TextSpan:
    """Убирает пробельные края: текст чанка не начинается с перевода строки."""
    start, end = span.start, span.end
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return TextSpan(start, end)
