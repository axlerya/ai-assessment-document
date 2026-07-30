"""Тесты конвейера нормализации."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from document_worker.domain.normalization.normalizer import TextNormalizer
from document_worker.domain.normalization.rules import RULES
from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.domain.value_objects.text import TextSpan

pytestmark = pytest.mark.unit

NORMALIZER = TextNormalizer()

LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz"
PRINTABLE = st.text(alphabet=LETTERS + "0123456789 .,;:()-\n\t", max_size=120)


def _normalize(text: str, method: ExtractionMethod = ExtractionMethod.OCR) -> str:
    return NORMALIZER.normalize(text, source=method).content


def test_rules_are_applied_in_declared_order() -> None:
    assert [rule.name for rule in RULES] == [
        "nfc_normalize",
        "fold_ligatures",
        "remove_invisible",
        "strip_control",
        "unify_dashes",
        "dehyphenate_line_break",
        "join_soft_lines",
        "fold_homoglyphs",
        "collapse_spaces",
    ]


def test_pipeline_cleans_realistic_page() -> None:
    source = "Договор\r\nаренды\tнежилого   поме-\nщения\n\nп. 1­2"

    assert _normalize(source) == "Договор аренды нежилого помещения\n\nп. 12"


def test_pipeline_output_is_deterministic() -> None:
    source = "Договор\r\nаренды   поме-\nщения"

    assert _normalize(source) == _normalize(source)


def test_homoglyph_rule_is_skipped_for_text_layer() -> None:
    source = "apeнды"

    assert _normalize(source, ExtractionMethod.TEXT_LAYER) == source
    assert _normalize(source, ExtractionMethod.OCR) != source


def test_offset_map_survives_whole_pipeline() -> None:
    source = "Договор\r\nаренды"

    result = NORMALIZER.normalize(source, source=ExtractionMethod.OCR)

    assert result.content == "Договор аренды"
    assert result.offsets.project_span(TextSpan(9, 15)) == TextSpan(8, 14)


def test_span_destroyed_by_normalization_projects_to_none() -> None:
    source = "а\u200b\u200bб"

    result = NORMALIZER.normalize(source, source=ExtractionMethod.OCR)

    assert result.offsets.project_span(TextSpan(1, 3)) is None


def test_empty_text_passes_through() -> None:
    result = NORMALIZER.normalize("", source=ExtractionMethod.OCR)

    assert result.content == ""
    assert result.offsets.source_length == 0


@given(PRINTABLE)
def test_pipeline_is_idempotent(text: str) -> None:
    once = _normalize(text)

    assert _normalize(once) == once


@given(PRINTABLE)
def test_digit_sequence_is_preserved(text: str) -> None:
    digits = [char for char in text if char.isdigit()]

    assert [char for char in _normalize(text) if char.isdigit()] == digits


@given(PRINTABLE)
def test_letter_count_never_increases(text: str) -> None:
    # Лигатур в алфавите стратегии нет, поэтому единственный разворачивающий
    # текст путь исключён: букв может стать только меньше.
    source_letters = sum(1 for char in text if char.isalpha())

    assert sum(1 for char in _normalize(text) if char.isalpha()) <= source_letters


@given(PRINTABLE)
def test_offset_map_covers_whole_source(text: str) -> None:
    result = NORMALIZER.normalize(text, source=ExtractionMethod.OCR)

    assert result.offsets.source_length == len(text)
    assert result.offsets.target_length == len(result.content)
