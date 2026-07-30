"""Тесты канонического текста страницы."""

from __future__ import annotations

import pytest

from document_worker.domain.constants import MAX_PAGE_TEXT_LENGTH
from document_worker.domain.errors import (
    FabricatedTextDetected,
    InvalidTextSpan,
    InvariantViolation,
)
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ExtractionMethod, IllegibleReason
from document_worker.domain.value_objects.recognized_text import RecognizedText
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan

pytestmark = pytest.mark.unit


def _span(start: int, end: int, raw: str) -> IllegibleSpan:
    return IllegibleSpan(
        span=TextSpan(start, end),
        confidence=OcrConfidence(0.3),
        reason=IllegibleReason.LOW_OCR_CONFIDENCE,
        raw_text=raw,
    )


def test_cannot_be_created_without_confidence_argument() -> None:
    with pytest.raises(TypeError):
        RecognizedText(content="текст", method=ExtractionMethod.OCR)  # type: ignore[call-arg]


def test_text_layer_requires_none_confidence() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="текст",
            method=ExtractionMethod.TEXT_LAYER,
            confidence=OcrConfidence(1.0),
        )


def test_text_layer_accepts_none_confidence() -> None:
    text = RecognizedText(
        content="текст",
        method=ExtractionMethod.TEXT_LAYER,
        confidence=None,
    )

    assert text.confidence is None


@pytest.mark.parametrize("method", [ExtractionMethod.OCR, ExtractionMethod.HYBRID])
def test_ocr_method_requires_non_none_confidence(method: ExtractionMethod) -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(content="текст", method=method, confidence=None)


def test_none_method_requires_none_confidence() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="",
            method=ExtractionMethod.NONE,
            confidence=OcrConfidence.ZERO,
        )


def test_none_method_requires_empty_content() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="текст",
            method=ExtractionMethod.NONE,
            confidence=None,
        )


def test_text_layer_rejects_illegible_spans() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="текст",
            method=ExtractionMethod.TEXT_LAYER,
            confidence=None,
            illegible_spans=(_span(0, 5, "текст"),),
        )


def test_rejects_content_above_page_limit() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="я" * (MAX_PAGE_TEXT_LENGTH + 1),
            method=ExtractionMethod.OCR,
            confidence=OcrConfidence(0.9),
        )


def test_rejects_span_outside_content() -> None:
    with pytest.raises(InvalidTextSpan):
        RecognizedText(
            content="абв",
            method=ExtractionMethod.OCR,
            confidence=OcrConfidence(0.9),
            illegible_spans=(_span(2, 10, "в"),),
        )


def test_rejects_overlapping_illegible_spans() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="абвгде",
            method=ExtractionMethod.OCR,
            confidence=OcrConfidence(0.9),
            illegible_spans=(_span(0, 4, "абвг"), _span(3, 6, "где")),
        )


def test_accepts_touching_spans() -> None:
    text = RecognizedText(
        content="абвгде",
        method=ExtractionMethod.OCR,
        confidence=OcrConfidence(0.9),
        illegible_spans=(_span(0, 3, "абв"), _span(3, 6, "где")),
    )

    assert text.illegible_char_count == 6


def test_rejects_unsorted_spans() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="абвгде",
            method=ExtractionMethod.OCR,
            confidence=OcrConfidence(0.9),
            illegible_spans=(_span(3, 6, "где"), _span(0, 3, "абв")),
        )


def test_rejects_span_raw_text_differing_from_content() -> None:
    with pytest.raises(FabricatedTextDetected):
        RecognizedText(
            content="абвгде",
            method=ExtractionMethod.OCR,
            confidence=OcrConfidence(0.9),
            illegible_spans=(_span(0, 3, "ЖЗИ"),),
        )


def test_rejects_content_with_rendered_marker() -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText(
            content="начало [НЕРАЗБОРЧИВО: строка 1, confidence=0.31] конец",
            method=ExtractionMethod.OCR,
            confidence=OcrConfidence(0.9),
        )


def test_allows_zero_length_span_for_no_text_recognized() -> None:
    text = RecognizedText.nothing_recognized(
        method=ExtractionMethod.OCR,
        reason=IllegibleReason.NO_TEXT_RECOGNIZED,
    )

    assert text.content == ""
    assert text.confidence == OcrConfidence.ZERO
    assert len(text.illegible_spans) == 1
    assert text.illegible_spans[0].span.is_empty


def test_not_extracted_page_has_no_confidence_and_no_spans() -> None:
    text = RecognizedText.not_extracted()

    assert text.method is ExtractionMethod.NONE
    assert text.confidence is None
    assert text.illegible_spans == ()


@pytest.mark.parametrize("method", [ExtractionMethod.TEXT_LAYER, ExtractionMethod.NONE])
def test_nothing_recognized_rejects_methods_without_confidence(
    method: ExtractionMethod,
) -> None:
    with pytest.raises(InvariantViolation):
        RecognizedText.nothing_recognized(
            method=method,
            reason=IllegibleReason.NO_TEXT_RECOGNIZED,
        )


def test_char_count_counts_code_points() -> None:
    text = RecognizedText(
        content="абв",
        method=ExtractionMethod.OCR,
        confidence=OcrConfidence(0.9),
    )

    assert text.char_count == 3


def test_illegible_ratio_is_share_of_content() -> None:
    text = RecognizedText(
        content="абвгде",
        method=ExtractionMethod.OCR,
        confidence=OcrConfidence(0.9),
        illegible_spans=(_span(0, 3, "абв"),),
    )

    assert text.has_illegible
    assert text.illegible_ratio == pytest.approx(0.5)


def test_illegible_ratio_is_zero_for_clean_text() -> None:
    text = RecognizedText(
        content="абв",
        method=ExtractionMethod.OCR,
        confidence=OcrConfidence(0.9),
    )

    assert not text.has_illegible
    assert text.illegible_ratio == 0.0


def test_illegible_ratio_of_empty_content_does_not_divide_by_zero() -> None:
    text = RecognizedText.nothing_recognized(
        method=ExtractionMethod.OCR,
        reason=IllegibleReason.NO_TEXT_RECOGNIZED,
    )

    assert text.illegible_ratio == 0.0


def test_is_immutable() -> None:
    text = RecognizedText(
        content="абв",
        method=ExtractionMethod.OCR,
        confidence=OcrConfidence(0.9),
    )

    with pytest.raises(AttributeError):
        text.content = "другое"  # type: ignore[misc]


def test_has_no_api_for_substituting_text() -> None:
    forbidden = {"corrected_text", "restored_text", "guess", "with_content", "fix"}

    assert not forbidden & set(dir(RecognizedText))
