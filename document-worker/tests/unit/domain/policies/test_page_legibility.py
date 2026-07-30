"""Тесты политики читаемости страницы."""

from __future__ import annotations

import pytest

from document_worker.domain.policies.page_legibility import PageLegibilityPolicy
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    IllegibleReason,
    PageStatus,
)
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.domain.value_objects.text import RecognizedWord, TextSpan

pytestmark = pytest.mark.unit

POLICY = PageLegibilityPolicy()


def _words(content: str, confidences: dict[str, float]) -> list[RecognizedWord]:
    """Режет текст по пробелам и вешает на слова заданную уверенность."""
    words: list[RecognizedWord] = []
    offset = 0
    for token in content.split(" "):
        start = content.index(token, offset)
        words.append(
            RecognizedWord(
                text=token,
                confidence=OcrConfidence(confidences.get(token, 0.95)),
                span=TextSpan(start, start + len(token)),
            )
        )
        offset = start + len(token)
    return words


def test_text_layer_page_is_extracted_without_spans() -> None:
    verdict = POLICY.evaluate(
        method=ExtractionMethod.TEXT_LAYER,
        words=[],
        content="договор аренды",
    )

    assert verdict.status is PageStatus.EXTRACTED
    assert verdict.illegible_spans == ()
    assert verdict.illegible_ratio == 0.0


def test_empty_word_list_returns_illegible_with_no_text_recognized() -> None:
    verdict = POLICY.evaluate(method=ExtractionMethod.OCR, words=[], content="")

    assert verdict.status is PageStatus.ILLEGIBLE
    assert len(verdict.illegible_spans) == 1
    span = verdict.illegible_spans[0]
    assert span.reason is IllegibleReason.NO_TEXT_RECOGNIZED
    assert span.span.is_empty
    assert span.raw_text == ""


def test_sparse_page_is_illegible_with_whole_page_span() -> None:
    content = "а б в"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {}),
        content=content,
    )

    assert verdict.status is PageStatus.ILLEGIBLE
    assert verdict.illegible_spans[0].reason is IllegibleReason.IMAGE_TOO_NOISY
    assert verdict.illegible_spans[0].span == TextSpan(0, len(content))
    assert "sparse_text" in verdict.warnings


def test_all_words_above_threshold_gives_extracted_without_spans() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {}),
        content=content,
    )

    assert verdict.status is PageStatus.EXTRACTED
    assert verdict.illegible_spans == ()


def test_single_word_below_threshold_makes_page_partially_illegible() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {"нежилого": 0.2}),
        content=content,
    )

    assert verdict.status is PageStatus.PARTIALLY_ILLEGIBLE
    assert len(verdict.illegible_spans) == 1
    assert verdict.illegible_spans[0].raw_text == "нежилого"


def test_ratio_above_threshold_makes_page_illegible() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"
    low = dict.fromkeys(["договор", "аренды", "нежилого", "помещения"], 0.2)

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, low),
        content=content,
    )

    assert verdict.status is PageStatus.ILLEGIBLE


def test_low_mean_confidence_makes_page_illegible() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"
    low = dict.fromkeys(content.split(" "), 0.45)

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, low),
        content=content,
    )

    assert verdict.status is PageStatus.ILLEGIBLE


def test_mean_confidence_is_weighted_by_word_length() -> None:
    content = "ответственность и"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {"ответственность": 0.9, "и": 0.1}),
        content=content,
    )

    # Простое среднее дало бы 0.5; вес по длине слова тянет к длинному слову.
    assert verdict.mean_confidence.value > 0.8


def test_adjacent_low_confidence_words_merge_into_single_span() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {"аренды": 0.2, "нежилого": 0.2}),
        content=content,
    )

    assert len(verdict.illegible_spans) == 1
    assert verdict.illegible_spans[0].raw_text == "аренды нежилого"


def test_words_separated_by_legible_text_produce_two_spans() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {"договор": 0.2, "помещения": 0.2}),
        content=content,
    )

    assert len(verdict.illegible_spans) == 2


def test_span_raw_text_equals_original_fragment() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {"помещения": 0.2}),
        content=content,
    )

    span = verdict.illegible_spans[0]
    assert span.raw_text == content[span.span.start : span.span.end]


def test_span_bbox_is_union_of_word_boxes() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"
    words = _words(content, {"аренды": 0.2, "нежилого": 0.2})
    boxed = [
        RecognizedWord(
            text=word.text,
            confidence=word.confidence,
            span=word.span,
            bbox=BoundingBox(0.1 * index + 0.1, 0.1, 0.1 * index + 0.15, 0.2),
            line_number=1,
        )
        for index, word in enumerate(words)
    ]

    verdict = POLICY.evaluate(method=ExtractionMethod.OCR, words=boxed, content=content)

    span = verdict.illegible_spans[0]
    assert span.bbox == BoundingBox(0.2, 0.1, 0.35, 0.2)
    assert span.line_number == 1


def test_policy_never_returns_extracted_with_non_empty_spans() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {"аренды": 0.2}),
        content=content,
    )

    assert verdict.status is not PageStatus.EXTRACTED


def test_verdict_is_deterministic_for_same_words() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"
    words = _words(content, {"аренды": 0.2})

    first = POLICY.evaluate(method=ExtractionMethod.OCR, words=words, content=content)
    second = POLICY.evaluate(method=ExtractionMethod.OCR, words=words, content=content)

    assert first == second


def test_warnings_report_degradation() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, {"нежилого": 0.2}),
        content=content,
    )

    assert "heavily_degraded" in verdict.warnings
    assert "critical_words_present" in verdict.warnings


def test_low_mean_confidence_produces_warning() -> None:
    content = "договор аренды нежилого помещения между сторонами настоящего"
    lowered = dict.fromkeys(content.split(" "), 0.7)

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, lowered),
        content=content,
    )

    assert "low_mean_confidence" in verdict.warnings


def test_many_fragments_produce_warning() -> None:
    tokens = [f"слово{index:02d}" for index in range(24)]
    content = " ".join(tokens)
    low = {token: 0.2 for index, token in enumerate(tokens) if index % 2 == 0}

    verdict = POLICY.evaluate(
        method=ExtractionMethod.OCR,
        words=_words(content, low),
        content=content,
    )

    assert "many_illegible_fragments" in verdict.warnings
