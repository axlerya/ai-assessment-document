"""Тесты снимков качества документа и вердиктов политик."""

from __future__ import annotations

import pytest

from document_worker.domain.errors import InvariantViolation
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    DocumentStatus,
    ExtractionMethod,
    PageStatus,
)
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.quality import (
    DocumentQualityStats,
    DocumentStatusVerdict,
    PageOutcome,
)

pytestmark = pytest.mark.unit


def _outcome(  # noqa: PLR0913 — фабрика теста повторяет поля PageOutcome
    number: int,
    status: PageStatus,
    method: ExtractionMethod,
    *,
    confidence: float | None = None,
    chars: int = 100,
    illegible_chars: int = 0,
) -> PageOutcome:
    return PageOutcome(
        page_number=PageNumber(number),
        status=status,
        method=method,
        confidence=None if confidence is None else OcrConfidence(confidence),
        char_count=chars,
        illegible_char_count=illegible_chars,
    )


def _text_layer(number: int) -> PageOutcome:
    return _outcome(number, PageStatus.EXTRACTED, ExtractionMethod.TEXT_LAYER)


def _ocr(number: int, *, confidence: float = 0.9) -> PageOutcome:
    return _outcome(
        number, PageStatus.EXTRACTED, ExtractionMethod.OCR, confidence=confidence
    )


def _failed(number: int) -> PageOutcome:
    return _outcome(
        number, PageStatus.FAILED, ExtractionMethod.NONE, chars=0, illegible_chars=0
    )


def test_page_outcome_with_text_layer_requires_no_confidence() -> None:
    with pytest.raises(InvariantViolation):
        _outcome(1, PageStatus.EXTRACTED, ExtractionMethod.TEXT_LAYER, confidence=1.0)


def test_page_outcome_with_ocr_requires_confidence() -> None:
    with pytest.raises(InvariantViolation):
        _outcome(1, PageStatus.EXTRACTED, ExtractionMethod.OCR)


def test_page_outcome_rejects_illegible_chars_above_total() -> None:
    with pytest.raises(InvariantViolation):
        _outcome(
            1,
            PageStatus.PARTIALLY_ILLEGIBLE,
            ExtractionMethod.OCR,
            confidence=0.5,
            chars=10,
            illegible_chars=11,
        )


def test_failed_page_must_use_none_method() -> None:
    with pytest.raises(InvariantViolation):
        _outcome(1, PageStatus.FAILED, ExtractionMethod.OCR, confidence=0.5)


def test_none_method_requires_failed_status() -> None:
    with pytest.raises(InvariantViolation):
        _outcome(1, PageStatus.EXTRACTED, ExtractionMethod.NONE, chars=0)


def test_stats_counters_sum_to_total_including_hybrid() -> None:
    outcomes = [
        _text_layer(1),
        _ocr(2),
        _outcome(3, PageStatus.EXTRACTED, ExtractionMethod.HYBRID, confidence=0.8),
        _failed(4),
    ]

    stats = DocumentQualityStats.from_outcomes(outcomes)

    assert stats.pages_total == 4
    assert stats.pages_text_layer == 1
    assert stats.pages_ocr == 1
    assert stats.pages_hybrid == 1
    assert stats.pages_failed == 1
    assert (
        stats.pages_text_layer
        + stats.pages_ocr
        + stats.pages_hybrid
        + stats.pages_failed
        == stats.pages_total
    )


def test_stats_status_counters_sum_to_total() -> None:
    outcomes = [
        _text_layer(1),
        _outcome(
            2,
            PageStatus.PARTIALLY_ILLEGIBLE,
            ExtractionMethod.OCR,
            confidence=0.5,
            illegible_chars=10,
        ),
        _outcome(
            3,
            PageStatus.ILLEGIBLE,
            ExtractionMethod.OCR,
            confidence=0.2,
            illegible_chars=100,
        ),
        _failed(4),
    ]

    stats = DocumentQualityStats.from_outcomes(outcomes)

    assert (
        stats.pages_extracted
        + stats.pages_partially_illegible
        + stats.pages_illegible
        + stats.pages_failed
        == stats.pages_total
    )


def test_stats_reject_counters_that_do_not_sum_to_total() -> None:
    with pytest.raises(InvariantViolation):
        DocumentQualityStats(
            pages_total=4,
            pages_text_layer=1,
            pages_ocr=1,
            pages_hybrid=1,
            pages_failed=0,
            pages_extracted=3,
            pages_partially_illegible=0,
            pages_illegible=0,
            pages_failed_status=0,
            total_chars=300,
            illegible_chars=0,
            mean_ocr_confidence=OcrConfidence(0.9),
        )


def test_page_outcome_rejects_negative_char_count() -> None:
    with pytest.raises(InvariantViolation):
        _outcome(1, PageStatus.EXTRACTED, ExtractionMethod.TEXT_LAYER, chars=-1)


def test_stats_reject_status_counters_that_do_not_sum_to_total() -> None:
    with pytest.raises(InvariantViolation):
        DocumentQualityStats(
            pages_total=2,
            pages_text_layer=2,
            pages_ocr=0,
            pages_hybrid=0,
            pages_failed=0,
            pages_extracted=1,
            pages_partially_illegible=0,
            pages_illegible=0,
            pages_failed_status=0,
            total_chars=200,
            illegible_chars=0,
            mean_ocr_confidence=None,
        )


def test_illegible_char_ratio_is_share_of_all_chars() -> None:
    outcomes = [
        _outcome(
            1,
            PageStatus.PARTIALLY_ILLEGIBLE,
            ExtractionMethod.OCR,
            confidence=0.5,
            chars=100,
            illegible_chars=25,
        ),
    ]

    stats = DocumentQualityStats.from_outcomes(outcomes)

    assert stats.illegible_char_ratio == pytest.approx(0.25)


def test_failed_verdict_allows_problem_pages() -> None:
    stats = DocumentQualityStats.from_outcomes([_failed(1)])

    verdict = DocumentStatusVerdict(
        status=DocumentStatus.FAILED,
        stats=stats,
        reasons=("страница не прочитана",),
        problem_pages=(PageNumber(1),),
    )

    assert verdict.status is DocumentStatus.FAILED


def test_mean_confidence_ignores_text_layer_pages() -> None:
    outcomes = [_text_layer(1), _ocr(2, confidence=0.6)]

    stats = DocumentQualityStats.from_outcomes(outcomes)

    assert stats.mean_ocr_confidence == OcrConfidence(0.6)


def test_mean_confidence_is_none_without_ocr_pages() -> None:
    stats = DocumentQualityStats.from_outcomes([_text_layer(1), _text_layer(2)])

    assert stats.mean_ocr_confidence is None


def test_mean_confidence_is_weighted_by_char_count() -> None:
    outcomes = [
        _outcome(
            1, PageStatus.EXTRACTED, ExtractionMethod.OCR, confidence=1.0, chars=10
        ),
        _outcome(
            2, PageStatus.EXTRACTED, ExtractionMethod.OCR, confidence=0.0, chars=90
        ),
    ]

    stats = DocumentQualityStats.from_outcomes(outcomes)

    assert stats.mean_ocr_confidence == OcrConfidence(0.1)


def test_ocr_coverage_counts_ocr_and_hybrid() -> None:
    outcomes = [
        _text_layer(1),
        _ocr(2),
        _outcome(3, PageStatus.EXTRACTED, ExtractionMethod.HYBRID, confidence=0.8),
        _failed(4),
    ]

    stats = DocumentQualityStats.from_outcomes(outcomes)

    assert stats.ocr_coverage == pytest.approx(0.5)


def test_usable_and_failed_page_ratios() -> None:
    outcomes = [_text_layer(1), _text_layer(2), _text_layer(3), _failed(4)]

    stats = DocumentQualityStats.from_outcomes(outcomes)

    assert stats.usable_page_ratio == pytest.approx(0.75)
    assert stats.failed_page_ratio == pytest.approx(0.25)


def test_illegible_char_ratio_of_document_without_text_is_zero() -> None:
    stats = DocumentQualityStats.from_outcomes([_failed(1)])

    assert stats.illegible_char_ratio == 0.0


def test_from_outcomes_rejects_empty_list() -> None:
    with pytest.raises(InvariantViolation):
        DocumentQualityStats.from_outcomes([])


def test_from_outcomes_rejects_duplicate_page_numbers() -> None:
    with pytest.raises(InvariantViolation):
        DocumentQualityStats.from_outcomes([_text_layer(1), _text_layer(1)])


def test_document_status_verdict_requires_terminal_status() -> None:
    stats = DocumentQualityStats.from_outcomes([_text_layer(1)])

    with pytest.raises(InvariantViolation):
        DocumentStatusVerdict(
            status=DocumentStatus.PROCESSING,
            stats=stats,
            reasons=(),
            problem_pages=(),
        )


def test_partially_processed_verdict_requires_problem_pages() -> None:
    stats = DocumentQualityStats.from_outcomes([_text_layer(1), _failed(2)])

    with pytest.raises(InvariantViolation):
        DocumentStatusVerdict(
            status=DocumentStatus.PARTIALLY_PROCESSED,
            stats=stats,
            reasons=("одна страница не прочитана",),
            problem_pages=(),
        )


def test_processed_verdict_forbids_problem_pages() -> None:
    stats = DocumentQualityStats.from_outcomes([_text_layer(1)])

    with pytest.raises(InvariantViolation):
        DocumentStatusVerdict(
            status=DocumentStatus.PROCESSED,
            stats=stats,
            reasons=(),
            problem_pages=(PageNumber(1),),
        )
