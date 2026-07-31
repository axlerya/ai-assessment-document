"""Классификация строк юридического документа.

Порядок правил здесь не вкусовой: реквизиты, распознанные как заголовки,
рассыпают подписной блок договора на десяток односстрочных секций.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.domain.chunking.line_classifier import LineClassifier, LineKind
from tests.unit.domain.chunking.support import ocr_page, text_layer_page

if TYPE_CHECKING:
    from document_worker.domain.chunking.line_classifier import LayoutLine

pytestmark = pytest.mark.unit

CLASSIFIER = LineClassifier()

# Обрамление даёт странице строчные буквы: без него доля капс-строк переваливает
# за порог и детекция заголовков отключается предпроходом (guard G6).
PREFIX = "Настоящий договор заключён между сторонами в городе Москве."
SUFFIX = "Стороны договорились о нижеследующем в полном объёме."


def kind_of(line: str) -> LineKind:
    """Вид строки, поставленной в середину обычной страницы."""
    page = text_layer_page(f"{PREFIX}\n{line}\n{SUFFIX}")
    return CLASSIFIER.classify_page(page)[1].kind


def line_of(line: str) -> LayoutLine:
    """Сама размеченная строка со всеми её полями."""
    page = text_layer_page(f"{PREFIX}\n{line}\n{SUFFIX}")
    return CLASSIFIER.classify_page(page)[1]


def test_detects_article_heading_with_number_and_title() -> None:
    line = line_of("Статья 12. Ответственность сторон")

    assert line.kind is LineKind.HEADING_ARTICLE
    assert line.number == "12"
    assert line.title == "Ответственность сторон"


def test_detects_section_heading_with_roman_number() -> None:
    line = line_of("РАЗДЕЛ II. ПРЕДМЕТ ДОГОВОРА")

    assert line.kind is LineKind.HEADING_PART
    assert line.number == "II"


def test_detects_numbered_clause_up_to_five_levels() -> None:
    line = line_of("12.3.4.5.6 Текст пункта")

    assert line.kind is LineKind.CLAUSE
    assert line.number == "12.3.4.5.6"


def test_detects_letter_subclause() -> None:
    line = line_of("а) поставка товара;")

    assert line.kind is LineKind.SUBCLAUSE_LETTER
    assert line.number == "а"


def test_detects_paren_subclause() -> None:
    assert kind_of("1) первое условие;") is LineKind.SUBCLAUSE_PAREN


def test_detects_appendix_heading_with_number_sign() -> None:
    line = line_of("Приложение № 2 к Договору")

    assert line.kind is LineKind.APPENDIX
    assert line.number == "2"


def test_year_at_line_start_is_not_treated_as_clause() -> None:
    assert kind_of("2024 Отчёт о выполнении работ") is LineKind.TEXT


def test_money_amount_at_line_start_is_not_treated_as_clause() -> None:
    assert kind_of("1 500 000 (Один миллион пятьсот тысяч) рублей") is LineKind.TEXT


def test_requisites_line_is_not_classified_as_uppercase_heading() -> None:
    # Строка без строчных букв матчится правилом капс-заголовка; при обратном
    # порядке блок реквизитов распался бы на цепочку секций.
    assert kind_of("ИНН 7701234567 КПП 770101001") is LineKind.REQUISITES


def test_bank_account_line_is_requisites() -> None:
    assert kind_of("р/с 40702810400000012345") is LineKind.REQUISITES
    assert kind_of("40702810400000012345") is LineKind.REQUISITES


def test_detects_city_and_russian_date_as_requisites() -> None:
    assert kind_of("г. Москва") is LineKind.REQUISITES
    assert kind_of("«15» марта 2024 г.") is LineKind.REQUISITES


def test_numbered_clause_wins_over_requisites() -> None:
    # Иначе пункт «1. ИНН заказчика...» перестаёт быть пунктом и выпадает
    # из дерева секций.
    assert kind_of("1. ИНН заказчика указывается в разделе 12") is LineKind.CLAUSE


def test_signature_line_with_underscores_is_signature() -> None:
    assert kind_of("______________ /Иванов И.И./") is LineKind.SIGNATURE


def test_detects_stamp_marker() -> None:
    assert kind_of("М.П.") is LineKind.SIGNATURE


def test_detects_role_line_as_signature() -> None:
    assert kind_of("Генеральный директор") is LineKind.SIGNATURE


def test_detects_bullet_item() -> None:
    assert kind_of("— поставка товара в срок") is LineKind.BULLET


def test_page_of_very_short_lines_is_classified() -> None:
    # Судить о регистре страницы не по чему: делить на ноль нельзя.
    page = text_layer_page("Да\nНет\nОк")

    assert [line.kind for line in CLASSIFIER.classify_page(page)] == [
        LineKind.TEXT,
        LineKind.TEXT,
        LineKind.TEXT,
    ]


def test_detects_pipe_table_row() -> None:
    assert kind_of("| Наименование | Цена |") is LineKind.TABLE_ROW


def test_detects_table_separator() -> None:
    assert kind_of("-----------------") is LineKind.TABLE_SEPARATOR


def test_gap_row_is_table_only_in_series_of_at_least_two() -> None:
    single = text_layer_page(f"{PREFIX}\nНаименование   Цена   Количество\n{SUFFIX}")
    series = text_layer_page(
        f"{PREFIX}\nНаименование   Цена   Количество\nБумага   120   40\n{SUFFIX}"
    )

    assert CLASSIFIER.classify_page(single)[1].kind is LineKind.TEXT
    assert [line.kind for line in CLASSIFIER.classify_page(series)[1:3]] == [
        LineKind.TABLE_ROW,
        LineKind.TABLE_ROW,
    ]


def test_gap_rows_at_the_end_of_page_still_form_a_table() -> None:
    page = text_layer_page(f"{PREFIX}\nБумага   120   40\nРучка   80   20")

    assert [line.kind for line in CLASSIFIER.classify_page(page)[1:]] == [
        LineKind.TABLE_ROW,
        LineKind.TABLE_ROW,
    ]


def test_row_with_other_column_count_starts_its_own_series() -> None:
    # Подписной блок под таблицей — не её продолжение: одиночная строка с
    # другим числом колонок таблицей не становится.
    page = text_layer_page(
        f"{PREFIX}\nБумага   120   40\nРучка   80   20\nа   б   в   г   д   е\n{SUFFIX}"
    )

    assert [line.kind for line in CLASSIFIER.classify_page(page)[1:4]] == [
        LineKind.TABLE_ROW,
        LineKind.TABLE_ROW,
        LineKind.TEXT,
    ]


def test_page_number_line_is_page_artifact() -> None:
    assert kind_of("- 4 -") is LineKind.PAGE_ARTIFACT
    assert kind_of("Страница 4 из 23") is LineKind.PAGE_ARTIFACT


def test_number_above_max_pages_is_not_page_artifact() -> None:
    assert kind_of("2024") is LineKind.TEXT


def test_blank_line_is_blank() -> None:
    assert kind_of("   ") is LineKind.BLANK


def test_detects_uppercase_heading() -> None:
    assert kind_of("ПРЕДМЕТ ДОГОВОРА") is LineKind.HEADING_UPPER


def test_rejects_uppercase_line_longer_than_max_heading_chars() -> None:
    assert kind_of("СТОРОНЫ ДОГОВОРИЛИСЬ " * 10) is LineKind.TEXT


def test_rejects_uppercase_paragraph_with_several_sentences() -> None:
    assert kind_of("ПЕРВОЕ УСЛОВИЕ. ВТОРОЕ УСЛОВИЕ. ТРЕТЬЕ") is LineKind.TEXT


def test_rejects_uppercase_heading_when_illegible_ratio_above_threshold() -> None:
    # OCR-мусор капсом не имеет права открыть секцию.
    content = f"{PREFIX}\nПРЕДМЕТ ДОГОВОРА\n{SUFFIX}"
    start = content.index("ПРЕДМЕТ")
    page = ocr_page(content, illegible=((start, start + 7, 0.3),))

    assert CLASSIFIER.classify_page(page)[1].kind is LineKind.TEXT


def test_disables_uppercase_detection_when_page_is_mostly_uppercase() -> None:
    # Типовой устав ООО иначе даёт 30–50 «секций» на первой странице.
    page = text_layer_page(
        "ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ\n"
        "УСТАВ\n"
        "ГОРОД МОСКВА\n"
        "утверждён общим собранием участников"
    )

    kinds = [line.kind for line in CLASSIFIER.classify_page(page)]

    assert kinds == [LineKind.TEXT, LineKind.TEXT, LineKind.TEXT, LineKind.TEXT]


def test_line_spans_are_page_relative_and_exclude_newline() -> None:
    page = text_layer_page("Первая строка\n\nВторая строка\nтретья")

    for line in CLASSIFIER.classify_page(page):
        assert line.span.slice_of(page.text.content) == line.text


def test_indent_counts_tab_as_four_spaces() -> None:
    page = text_layer_page(f"{PREFIX}\n\t1.2. Текст пункта\n{SUFFIX}")

    assert CLASSIFIER.classify_page(page)[1].indent == 4
