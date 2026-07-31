"""Сборка строк страницы в блоки."""

from __future__ import annotations

import pytest

from document_worker.domain.chunking.blocks import BlockKind
from document_worker.domain.chunking.line_classifier import LineClassifier
from document_worker.domain.chunking.paragraph_splitter import ParagraphSplitter
from document_worker.domain.chunking.structure_detector import StructureDetector
from tests.unit.domain.chunking.support import text_layer_page

pytestmark = pytest.mark.unit

CLASSIFIER = LineClassifier()
DETECTOR = StructureDetector()
SPLITTER = ParagraphSplitter()


def blocks_of(content: str) -> tuple[tuple[BlockKind, str, bool], ...]:
    """Вид блока, его текст и признак атомарности."""
    page = text_layer_page(content)
    lines = CLASSIFIER.classify_page(page)
    sections = DETECTOR.detect(lines)
    return tuple(
        (block.kind, block.span.slice_of(content), block.is_atomic)
        for block in SPLITTER.split(lines, sections, content=content)
    )


def test_blank_line_separates_paragraph_blocks() -> None:
    blocks = blocks_of("Первый абзац договора.\n\nВторой абзац договора.")

    assert [kind for kind, _, _ in blocks] == [
        BlockKind.PARAGRAPH,
        BlockKind.PARAGRAPH,
    ]


def test_consecutive_text_lines_form_one_paragraph() -> None:
    blocks = blocks_of("Первая строка абзаца\nвторая строка того же абзаца")

    assert len(blocks) == 1
    assert blocks[0][1] == "Первая строка абзаца\nвторая строка того же абзаца"


def test_page_artifact_line_is_excluded_from_any_block() -> None:
    # Колонтитул не сдвигает смещения соседей: его символы просто не покрыты.
    content = "Первый абзац договора.\n- 4 -\nВторой абзац договора."

    blocks = blocks_of(content)

    assert len(blocks) == 2
    assert all("- 4 -" not in text for _, text, _ in blocks)


def test_keeps_table_block_atomic_across_rows() -> None:
    blocks = blocks_of(
        "| Наименование | Цена |\n| Бумага | 120 |\n| Ручка | 40 |",
    )

    assert len(blocks) == 1
    assert blocks[0][0] is BlockKind.TABLE
    assert blocks[0][2] is True


def test_signature_block_is_atomic() -> None:
    blocks = blocks_of("Генеральный директор\n______________ /Иванов И.И./")

    assert blocks[0][0] is BlockKind.SIGNATURE
    assert blocks[0][2] is True


def test_requisites_block_is_atomic() -> None:
    blocks = blocks_of("ИНН 7701234567 КПП 770101001\nр/с 40702810400000012345")

    assert blocks[0][0] is BlockKind.REQUISITES
    assert blocks[0][2] is True


def test_list_item_absorbs_following_indented_text_lines() -> None:
    blocks = blocks_of("1.2. Текст пункта\n    продолжение того же пункта")

    assert len(blocks) == 1
    assert blocks[0][0] is BlockKind.LIST_ITEM


def test_heading_line_becomes_its_own_block() -> None:
    blocks = blocks_of("Статья 12. Ответственность\nИсполнитель отвечает.")

    assert [kind for kind, _, _ in blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
    ]


def test_block_span_excludes_leading_and_trailing_whitespace() -> None:
    blocks = blocks_of("   Абзац с отступом   ")

    assert blocks[0][1] == "Абзац с отступом"


def test_blocks_never_span_two_sections() -> None:
    blocks = blocks_of("1.1 Первый пункт\n1.2 Второй пункт")

    assert len(blocks) == 2
