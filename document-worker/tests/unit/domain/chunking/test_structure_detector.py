"""Дерево секций документа.

Стек секций — единственное состояние, которое переносится через границу
страницы: именно он связывает части секции, идущей с третьей по седьмую.
"""

from __future__ import annotations

import pytest

from document_worker.domain.chunking.line_classifier import LineClassifier
from document_worker.domain.chunking.structure_detector import StructureDetector
from document_worker.domain.chunking.structure_rules import MAX_HEADING_PATH_DEPTH
from tests.unit.domain.chunking.support import text_layer_page

pytestmark = pytest.mark.unit

CLASSIFIER = LineClassifier()
DETECTOR = StructureDetector()


def sections_of(*pages: str) -> tuple[list[str], list[tuple[str, ...]]]:
    """Тексты строк и путь секции каждой из них."""
    lines = [
        line
        for number, content in enumerate(pages, start=1)
        for line in CLASSIFIER.classify_page(text_layer_page(content, number=number))
    ]
    nodes = DETECTOR.detect(lines)
    return [line.text for line in lines], [node.heading_path for node in nodes]


def path_of(fragment: str, *pages: str) -> tuple[str, ...]:
    """Путь секции строки, содержащей фрагмент."""
    texts, paths = sections_of(*pages)
    index = next(i for i, text in enumerate(texts) if fragment in text)
    return paths[index]


def test_nests_clause_under_article_under_section() -> None:
    path = path_of(
        "12.3",
        "РАЗДЕЛ II. ПРЕДМЕТ ДОГОВОРА\n"
        "Статья 12. Ответственность сторон\n"
        "12.3 Исполнитель отвечает за просрочку",
    )

    assert path == ("РАЗДЕЛ II", "Статья 12", "12.3")
    assert len(path) == 3


def test_deeper_clause_number_increases_rank_not_replaces_parent() -> None:
    path = path_of(
        "12.3.1",
        "Статья 12. Ответственность\n12.3 Общее правило\n12.3.1 Частный случай",
    )

    assert path == ("Статья 12", "12.3", "12.3.1")


def test_sibling_clause_replaces_previous_clause() -> None:
    path = path_of(
        "12.4",
        "Статья 12. Ответственность\n12.3 Общее правило\n12.4 Исключение",
    )

    assert path == ("Статья 12", "12.4")


def test_appendix_resets_stack_to_root() -> None:
    # Приложение не может стать потомком статьи основного текста.
    path = path_of(
        "Статья 1.",
        "РАЗДЕЛ I. ОБЩИЕ ПОЛОЖЕНИЯ\n"
        "Статья 5. Порядок расчётов\n"
        "ПРИЛОЖЕНИЕ № 1\n"
        "Статья 1. Форма акта",
    )

    assert path == ("ПРИЛОЖЕНИЕ 1", "Статья 1")


def test_text_before_first_heading_goes_to_root_section() -> None:
    path = path_of(
        "Настоящий договор",
        "Настоящий договор заключён между сторонами\nРАЗДЕЛ I. ОБЩИЕ ПОЛОЖЕНИЯ",
    )

    assert path == ()


def test_duplicate_section_path_gets_numeric_suffix() -> None:
    # Суффикс — деталь дерева: оператору показывается хлебная крошка без него,
    # а разрыв чанков обеспечивает сравнение по ключу.
    lines = CLASSIFIER.classify_page(
        text_layer_page(
            "Статья 5. Расчёты\nПорядок оплаты\nСтатья 6. Сроки\nСтатья 5. Расчёты"
        )
    )
    nodes = DETECTOR.detect(lines)
    first, second = nodes[0], nodes[3]

    assert first.key != second.key
    assert second.key.endswith("#2")
    assert first.heading_path == ("Статья 5",)


def test_section_stack_survives_page_boundary() -> None:
    texts, paths = sections_of(
        "Статья 12. Ответственность\n12.3 Исполнитель отвечает",
        "за просрочку поставки товара",
    )

    assert paths[texts.index("за просрочку поставки товара")] == ("Статья 12", "12.3")


def test_heading_path_depth_is_capped() -> None:
    deep = (
        "ПРИЛОЖЕНИЕ № 1\n"
        "РАЗДЕЛ I. ОБЩЕЕ\n"
        "ГЛАВА 1. ЧАСТНОЕ\n"
        "Статья 1. Условия\n"
        "УСЛОВИЯ ПОСТАВКИ\n"
        "1 Первый уровень\n"
        "1.1 Второй уровень\n"
        "1.1.1 Третий уровень\n"
        "1.1.1.1 Четвёртый уровень\n"
        "1.1.1.1.1 Пятый уровень"
    )

    path = path_of("Пятый уровень", deep)

    assert len(path) == MAX_HEADING_PATH_DEPTH
    assert path[0] == "ПРИЛОЖЕНИЕ 1"
    assert path[-1] == "1.1.1.1.1"


def test_uppercase_heading_opens_section() -> None:
    path = path_of(
        "текст условия",
        "Статья 3. Порядок\nУСЛОВИЯ ПОСТАВКИ\nтекст условия ниже заголовка",
    )

    assert path == ("Статья 3", "УСЛОВИЯ ПОСТАВКИ")


def test_subclause_nests_under_its_clause() -> None:
    path = path_of(
        "а) поставка",
        "Статья 4. Обязанности\n4.1 Исполнитель обязан\nа) поставка товара",
    )

    assert path == ("Статья 4", "4.1", "а")


def test_sibling_subclauses_stay_siblings() -> None:
    path = path_of(
        "б) оплата",
        "Статья 4. Обязанности\n4.1 Стороны обязаны\nа) поставка товара\nб) оплата",
    )

    assert path == ("Статья 4", "4.1", "б")
