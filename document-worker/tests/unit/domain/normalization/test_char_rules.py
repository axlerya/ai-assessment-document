"""Тесты посимвольных правил нормализации."""

from __future__ import annotations

import pytest

from document_worker.domain.normalization.rules import (
    COLLAPSE_SPACES,
    FOLD_LIGATURES,
    NFC_NORMALIZE,
    REMOVE_INVISIBLE,
    STRIP_CONTROL,
    UNIFY_DASHES,
)
from document_worker.domain.value_objects.text import TextSpan

pytestmark = pytest.mark.unit

# Коды, а не литералы: литеральные управляющие символы в исходнике
# запрещены линтером и незаметно портятся при автоправках.
INVISIBLE_CHARS = [
    chr(code)
    for code in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x200E, 0x200F, 0x00AD)
]
SOFT_HYPHEN = chr(0x00AD)
NBSP = chr(0x00A0)
SPECIAL_SPACES = [chr(code) for code in (0x00A0, 0x2007, 0x202F, 0x2009)]
HYPHEN_LIKE_DASHES = [chr(code) for code in (0x2010, 0x2011, 0x2212)]


def test_nfc_composes_decomposed_letter() -> None:
    text, mapping = NFC_NORMALIZE.apply("é")

    assert text == "é"
    assert mapping.project_offset(2) == 1


@pytest.mark.parametrize("symbol", ["№", "½", "²", "㎡", "©", "§"])
def test_nfc_keeps_compatibility_symbols(symbol: str) -> None:
    text, _ = NFC_NORMALIZE.apply(f"пункт {symbol} 3")

    assert symbol in text


def test_nfc_maps_singleton_to_its_canonical_form() -> None:
    # Знак ангстрема имеет каноническую форму — латинскую Å.
    text, mapping = NFC_NORMALIZE.apply(chr(0x212B))

    assert text == chr(0x00C5)
    assert mapping.target_length == 1


def test_nfc_refuses_to_expand_text() -> None:
    # Этот символ разворачивается в два; расширять текст правило не вправе.
    source = chr(0x0344)

    text, _ = NFC_NORMALIZE.apply(source)

    assert text == source


def test_nfc_keeps_plain_text_untouched() -> None:
    source = "договор аренды"

    text, mapping = NFC_NORMALIZE.apply(source)

    assert text == source
    assert mapping.target_length == len(source)


@pytest.mark.parametrize(
    ("ligature", "expanded"),
    [("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl")],
)
def test_ligature_is_expanded(ligature: str, expanded: str) -> None:
    text, _ = FOLD_LIGATURES.apply(f"a{ligature}b")

    assert text == f"a{expanded}b"


def test_ligature_expansion_updates_offset_map() -> None:
    text, mapping = FOLD_LIGATURES.apply("aﬁb")

    assert text == "afib"
    assert mapping.project_offset(2) == 3
    assert mapping.project_span(TextSpan(2, 3)) == TextSpan(3, 4)


@pytest.mark.parametrize("letter", ["æ", "œ", "ß"])
def test_alphabet_letters_are_not_treated_as_ligatures(letter: str) -> None:
    text, _ = FOLD_LIGATURES.apply(letter)

    assert text == letter


@pytest.mark.parametrize("invisible", INVISIBLE_CHARS)
def test_invisible_char_is_removed(invisible: str) -> None:
    text, _ = REMOVE_INVISIBLE.apply(f"дого{invisible}вор")

    assert text == "договор"


def test_soft_hyphen_is_removed() -> None:
    text, mapping = REMOVE_INVISIBLE.apply(f"дого{SOFT_HYPHEN}вор")

    assert text == "договор"
    assert mapping.target_length == 7


def test_nbsp_survives_invisible_rule() -> None:
    text, _ = REMOVE_INVISIBLE.apply(f"п.{NBSP}1")

    assert text == f"п.{NBSP}1"


@pytest.mark.parametrize("symbol", ["§", "№", "©"])
def test_visible_special_symbols_survive(symbol: str) -> None:
    text, _ = REMOVE_INVISIBLE.apply(symbol)

    assert text == symbol


def test_carriage_return_pair_becomes_single_newline() -> None:
    text, mapping = STRIP_CONTROL.apply("первая\r\nвторая")

    assert text == "первая\nвторая"
    assert mapping.project_offset(8) == 7


def test_lone_carriage_return_becomes_newline() -> None:
    text, _ = STRIP_CONTROL.apply("первая\rвторая")

    assert text == "первая\nвторая"


def test_tab_becomes_space() -> None:
    text, _ = STRIP_CONTROL.apply("а\tб")

    assert text == "а б"


def test_other_control_chars_are_removed() -> None:
    text, _ = STRIP_CONTROL.apply("а\x00\x07б")

    assert text == "аб"


def test_newline_survives_control_rule() -> None:
    text, _ = STRIP_CONTROL.apply("а\nб")

    assert text == "а\nб"


@pytest.mark.parametrize("dash", HYPHEN_LIKE_DASHES)
def test_hyphen_like_dash_becomes_plain_hyphen(dash: str) -> None:
    text, _ = UNIFY_DASHES.apply(f"п{dash}1")

    assert text == "п-1"


@pytest.mark.parametrize("dash", ["–", "—"])
def test_en_and_em_dash_are_untouched(dash: str) -> None:
    text, _ = UNIFY_DASHES.apply(f"истец {dash} ООО")

    assert text == f"истец {dash} ООО"


@pytest.mark.parametrize("quote", ["«", "»", "“", "”", "„", '"'])
def test_quotes_are_untouched(quote: str) -> None:
    text, _ = UNIFY_DASHES.apply(f"{quote}Ромашка{quote}")

    assert text == f"{quote}Ромашка{quote}"


@pytest.mark.parametrize("space", SPECIAL_SPACES)
def test_special_space_becomes_regular(space: str) -> None:
    text, _ = COLLAPSE_SPACES.apply(f"п.{space}1")

    assert text == "п. 1"


def test_space_run_collapses_to_one() -> None:
    text, mapping = COLLAPSE_SPACES.apply("а     б")

    assert text == "а б"
    assert mapping.project_offset(6) == 2


def test_trailing_spaces_before_newline_are_removed() -> None:
    text, _ = COLLAPSE_SPACES.apply("строка   \nдалее")

    assert text == "строка\nдалее"


def test_leading_indentation_is_preserved() -> None:
    # Отступ использует детектор структуры при чанковании.
    text, _ = COLLAPSE_SPACES.apply("абзац\n    пункт")

    assert text == "абзац\n    пункт"


def test_newline_survives_space_collapse() -> None:
    text, _ = COLLAPSE_SPACES.apply("а\n\nб")

    assert text == "а\n\nб"


@pytest.mark.parametrize(
    "rule",
    [
        NFC_NORMALIZE,
        FOLD_LIGATURES,
        REMOVE_INVISIBLE,
        STRIP_CONTROL,
        UNIFY_DASHES,
        COLLAPSE_SPACES,
    ],
)
def test_rule_of_empty_text_returns_empty(rule: object) -> None:
    text, mapping = rule.apply("")  # type: ignore[attr-defined]

    assert text == ""
    assert mapping.source_length == 0
