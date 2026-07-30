"""Тесты построчных правил: снятие переноса, склейка абзацев, омоглифы."""

from __future__ import annotations

import pytest

from document_worker.domain.normalization.rules import (
    DEHYPHENATE_LINE_BREAK,
    FOLD_HOMOGLYPHS,
    JOIN_SOFT_LINES,
)

pytestmark = pytest.mark.unit

DEHYPHENATION_CASES = [
    ("дого-\nвор", "договор"),
    ("по-\nпрежнему", "по-прежнему"),
    ("ответствен-\nность", "ответственность"),
    ("аренд-\nной платы", "арендной платы"),
    # Заглавная буква после переноса — это составное имя собственное.
    ("Санкт-\nПетербург", "Санкт-Петербург"),
    # Цифра после дефиса это диапазон или номер, а не перенос слова.
    ("п. 1-\n5", "п. 1-\n5"),
    # Дефис внутри строки не перенос.
    ("по-прежнему", "по-прежнему"),
    # Перед пустой строкой переноса не бывает.
    ("дого-\n\nвор", "дого-\n\nвор"),
    # Дефис после небуквенного символа.
    (") -\nвор", ") -\nвор"),
]


@pytest.mark.parametrize(("source", "expected"), DEHYPHENATION_CASES)
def test_dehyphenation_table(source: str, expected: str) -> None:
    text, _ = DEHYPHENATE_LINE_BREAK.apply(source)

    assert text == expected


def test_dehyphenation_updates_offset_map() -> None:
    text, mapping = DEHYPHENATE_LINE_BREAK.apply("дого-\nвор")

    assert text == "договор"
    assert mapping.project_offset(6) == 4


def test_hyphen_before_capital_keeps_hyphen_and_joins_lines() -> None:
    text, _ = DEHYPHENATE_LINE_BREAK.apply("Санкт-\nПетербург")

    assert text == "Санкт-Петербург"


def test_single_newline_inside_paragraph_becomes_space() -> None:
    text, _ = JOIN_SOFT_LINES.apply("первая строка\nвторая строка")

    assert text == "первая строка вторая строка"


def test_paragraph_break_survives() -> None:
    text, _ = JOIN_SOFT_LINES.apply("абзац один\n\nабзац два")

    assert text == "абзац один\n\nабзац два"


@pytest.mark.parametrize("terminator", [".", "!", "?", ":", ";"])
def test_sentence_end_prevents_joining(terminator: str) -> None:
    source = f"конец предложения{terminator}\nследующая строка"

    text, _ = JOIN_SOFT_LINES.apply(source)

    assert text == source


@pytest.mark.parametrize(
    "marker",
    [
        "— пункт",
        "• пункт",
        "- пункт",
        "1. пункт",
        "1) пункт",
        "1.2.3. пункт",
        "а) пункт",
    ],
)
def test_line_join_skips_list_markers(marker: str) -> None:
    source = f"вводная строка\n{marker}"

    text, _ = JOIN_SOFT_LINES.apply(source)

    assert text == source


def test_line_join_skips_numbered_items_in_sequence() -> None:
    source = "1. первый пункт\n2. второй пункт\n3. третий пункт"

    text, _ = JOIN_SOFT_LINES.apply(source)

    assert text == source


def test_line_join_updates_offset_map() -> None:
    text, mapping = JOIN_SOFT_LINES.apply("первая\nвторая")

    assert text == "первая вторая"
    assert mapping.project_offset(7) == 7


def test_homoglyph_is_replaced_in_dominant_cyrillic_token() -> None:
    # Латинские "c" и "o" внутри русского слова — ошибка распознавания.
    text, _ = FOLD_HOMOGLYPHS.apply("договор apeнды")

    assert "apeнды" not in text


def test_homoglyph_requires_dominant_script() -> None:
    # Токен целиком из омоглифов: алфавит определить нельзя, трогать нечего.
    for token in ("AC", "CO", "OPT"):
        text, _ = FOLD_HOMOGLYPHS.apply(token)
        assert text == token


def test_homoglyph_skips_tokens_with_digits() -> None:
    source = "AC123"

    text, _ = FOLD_HOMOGLYPHS.apply(source)

    assert text == source


def test_homoglyph_keeps_pure_latin_token() -> None:
    source = "contract"

    text, _ = FOLD_HOMOGLYPHS.apply(source)

    assert text == source


def test_homoglyph_keeps_pure_cyrillic_token() -> None:
    source = "договор"

    text, _ = FOLD_HOMOGLYPHS.apply(source)

    assert text == source


def test_homoglyph_rule_applies_to_ocr_only() -> None:
    assert FOLD_HOMOGLYPHS.applies_to_ocr_only


def test_homoglyph_replacement_keeps_length() -> None:
    source = "apeнды"

    text, mapping = FOLD_HOMOGLYPHS.apply(source)

    assert len(text) == len(source)
    assert mapping.project_offset(3) == 3


@pytest.mark.parametrize(
    "rule", [DEHYPHENATE_LINE_BREAK, JOIN_SOFT_LINES, FOLD_HOMOGLYPHS]
)
def test_rule_of_empty_text_returns_empty(rule: object) -> None:
    text, mapping = rule.apply("")  # type: ignore[attr-defined]

    assert text == ""
    assert mapping.source_length == 0
