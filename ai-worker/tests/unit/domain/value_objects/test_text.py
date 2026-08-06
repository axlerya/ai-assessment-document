"""Диапазон цитаты: он и есть проверяемость источника."""

from __future__ import annotations

import pytest

from ai_worker.domain.errors import InvalidTextSpan
from ai_worker.domain.value_objects.text import QuoteSpan

pytestmark = pytest.mark.unit

CHUNK_TEXT = "Договор № 12/АБ от 3 марта 2024 года заключён между сторонами."


def test_span_knows_its_length() -> None:
    assert QuoteSpan(start=0, end=7).length == 7


def test_span_cuts_exactly_its_range() -> None:
    span = QuoteSpan(start=0, end=17)

    assert span.slice_of(CHUNK_TEXT) == "Договор № 12/АБ о"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (5, 5),  # пустая цитата ничего не подтверждает
        (7, 3),  # перевёрнутый диапазон
        (-1, 4),
        (0, -1),
    ],
)
def test_degenerate_ranges_are_rejected(start: int, end: int) -> None:
    with pytest.raises(InvalidTextSpan):
        QuoteSpan(start=start, end=end)


def test_span_beyond_the_text_is_not_a_citation() -> None:
    # Смещения приходят от модели: диапазон за пределами чанка означает
    # придуманную цитату, а не срез источника.
    span = QuoteSpan(start=0, end=len(CHUNK_TEXT) + 1)

    with pytest.raises(InvalidTextSpan):
        span.slice_of(CHUNK_TEXT)


def test_span_is_measured_in_code_points_not_bytes() -> None:
    # Кириллица в UTF-8 занимает два байта на символ: считать в байтах значит
    # промахиваться цитатой на каждом русском документе.
    span = QuoteSpan(start=0, end=7)

    assert span.slice_of(CHUNK_TEXT) == "Договор"
    assert span.length == len("Договор")


def test_span_matches_the_text_it_claims_to_cut() -> None:
    span = QuoteSpan(start=0, end=7)

    assert span.matches(CHUNK_TEXT, quote="Договор")
    assert not span.matches(CHUNK_TEXT, quote="Соглашение")


def test_matching_a_quote_outside_the_text_is_false_not_an_error() -> None:
    # Проверка обязана дать вердикт, а не исключение: несовпавшая цитата —
    # штатный исход верификации, её утверждение просто не публикуется.
    span = QuoteSpan(start=0, end=len(CHUNK_TEXT) + 5)

    assert not span.matches(CHUNK_TEXT, quote="что угодно")


def test_spans_with_the_same_bounds_are_the_same_value() -> None:
    assert QuoteSpan(start=2, end=9) == QuoteSpan(start=2, end=9)
