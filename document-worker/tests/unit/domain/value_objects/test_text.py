"""Тесты текстовых диапазонов, распознанных слов и неразборчивых фрагментов."""

from __future__ import annotations

import pytest

from document_worker.domain.constants import MAX_ILLEGIBLE_CONFIDENCE
from document_worker.domain.errors import (
    InvalidIllegibleSpan,
    InvalidRecognizedWord,
    InvalidTextSpan,
)
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import IllegibleReason
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.domain.value_objects.text import (
    IllegibleSpan,
    RecognizedWord,
    TextSpan,
)

pytestmark = pytest.mark.unit

TECHNICAL_REASONS = [
    IllegibleReason.PAGE_RENDER_FAILED,
    IllegibleReason.OCR_FAILED,
    IllegibleReason.PAGE_TIMEOUT,
]


def test_text_span_rejects_start_greater_than_end() -> None:
    with pytest.raises(InvalidTextSpan):
        TextSpan(10, 5)


def test_text_span_rejects_negative_start() -> None:
    with pytest.raises(InvalidTextSpan):
        TextSpan(-1, 5)


def test_text_span_allows_zero_length_span() -> None:
    span = TextSpan(7, 7)

    assert span.length == 0
    assert span.is_empty


def test_text_span_length_and_slice() -> None:
    span = TextSpan(2, 5)

    assert span.length == 3
    assert span.slice_of("абвгдежз") == "вгд"


def test_text_span_overlaps_false_for_touching_spans() -> None:
    assert not TextSpan(0, 5).overlaps(TextSpan(5, 10))


def test_text_span_overlaps_true_for_intersecting_spans() -> None:
    assert TextSpan(0, 6).overlaps(TextSpan(5, 10))


def test_empty_span_overlaps_nothing() -> None:
    assert not TextSpan(3, 3).overlaps(TextSpan(0, 10))
    assert not TextSpan(0, 10).overlaps(TextSpan(3, 3))


def test_text_span_contains_inner_span() -> None:
    assert TextSpan(0, 10).contains(TextSpan(2, 5))
    assert not TextSpan(2, 5).contains(TextSpan(0, 10))


def test_text_span_shift_moves_both_bounds() -> None:
    assert TextSpan(2, 5).shift(3) == TextSpan(5, 8)


def test_text_span_shift_below_zero_raises() -> None:
    with pytest.raises(InvalidTextSpan):
        TextSpan(2, 5).shift(-3)


def test_text_span_within_checks_upper_bound() -> None:
    assert TextSpan(0, 5).within(5)
    assert not TextSpan(0, 5).within(4)


def test_recognized_word_span_length_must_match_text_length() -> None:
    with pytest.raises(InvalidRecognizedWord):
        RecognizedWord(text="абв", confidence=OcrConfidence(0.9), span=TextSpan(0, 5))


def test_recognized_word_accepts_matching_span() -> None:
    word = RecognizedWord(
        text="абв",
        confidence=OcrConfidence(0.9),
        span=TextSpan(0, 3),
    )

    assert word.text == "абв"


def test_recognized_word_with_empty_text_requires_zero_confidence() -> None:
    with pytest.raises(InvalidRecognizedWord):
        RecognizedWord(text="", confidence=OcrConfidence(0.5), span=TextSpan(0, 0))


def test_recognized_word_rejects_line_number_below_one() -> None:
    with pytest.raises(InvalidRecognizedWord):
        RecognizedWord(
            text="абв",
            confidence=OcrConfidence(0.9),
            span=TextSpan(0, 3),
            line_number=0,
        )


def test_illegible_span_rejects_confidence_above_threshold() -> None:
    with pytest.raises(InvalidIllegibleSpan):
        IllegibleSpan(
            span=TextSpan(0, 3),
            confidence=OcrConfidence(MAX_ILLEGIBLE_CONFIDENCE + 0.01),
            reason=IllegibleReason.LOW_OCR_CONFIDENCE,
            raw_text="абв",
        )


def test_illegible_span_accepts_confidence_on_threshold() -> None:
    span = IllegibleSpan(
        span=TextSpan(0, 3),
        confidence=OcrConfidence(MAX_ILLEGIBLE_CONFIDENCE),
        reason=IllegibleReason.LOW_OCR_CONFIDENCE,
        raw_text="абв",
    )

    assert span.reason is IllegibleReason.LOW_OCR_CONFIDENCE


def test_no_text_recognized_allows_empty_raw_text_and_zero_length() -> None:
    span = IllegibleSpan(
        span=TextSpan(0, 0),
        confidence=OcrConfidence.ZERO,
        reason=IllegibleReason.NO_TEXT_RECOGNIZED,
        raw_text="",
    )

    assert span.span.is_empty
    assert span.raw_text == ""


def test_no_text_recognized_rejects_non_empty_raw_text() -> None:
    with pytest.raises(InvalidIllegibleSpan):
        IllegibleSpan(
            span=TextSpan(0, 3),
            confidence=OcrConfidence.ZERO,
            reason=IllegibleReason.NO_TEXT_RECOGNIZED,
            raw_text="абв",
        )


def test_no_text_recognized_rejects_non_zero_confidence() -> None:
    with pytest.raises(InvalidIllegibleSpan):
        IllegibleSpan(
            span=TextSpan(0, 0),
            confidence=OcrConfidence(0.3),
            reason=IllegibleReason.NO_TEXT_RECOGNIZED,
            raw_text="",
        )


@pytest.mark.parametrize("reason", TECHNICAL_REASONS)
def test_technical_reason_requires_empty_zero_length_span(
    reason: IllegibleReason,
) -> None:
    span = IllegibleSpan(
        span=TextSpan(0, 0),
        confidence=OcrConfidence.ZERO,
        reason=reason,
        raw_text="",
    )

    assert span.reason.is_technical


@pytest.mark.parametrize("reason", TECHNICAL_REASONS)
def test_technical_reason_rejects_non_empty_span(reason: IllegibleReason) -> None:
    with pytest.raises(InvalidIllegibleSpan):
        IllegibleSpan(
            span=TextSpan(0, 3),
            confidence=OcrConfidence.ZERO,
            reason=reason,
            raw_text="абв",
        )


def test_illegible_span_rejects_line_number_below_one() -> None:
    with pytest.raises(InvalidIllegibleSpan):
        IllegibleSpan(
            span=TextSpan(0, 3),
            confidence=OcrConfidence(0.3),
            reason=IllegibleReason.LOW_OCR_CONFIDENCE,
            raw_text="абв",
            line_number=0,
        )


def test_illegible_span_keeps_bbox_and_line_number() -> None:
    box = BoundingBox(0.1, 0.1, 0.2, 0.2)

    span = IllegibleSpan(
        span=TextSpan(0, 3),
        confidence=OcrConfidence(0.3),
        reason=IllegibleReason.HANDWRITING,
        raw_text="абв",
        bbox=box,
        line_number=14,
    )

    assert span.bbox == box
    assert span.line_number == 14


def test_illegible_span_has_no_rendered_marker_field() -> None:
    span = IllegibleSpan(
        span=TextSpan(0, 3),
        confidence=OcrConfidence(0.3),
        reason=IllegibleReason.LOW_OCR_CONFIDENCE,
        raw_text="абв",
    )

    assert not hasattr(span, "marker"), "маркер рендерится на выдаче, а не хранится"
