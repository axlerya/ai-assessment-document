"""Тесты политики итогового статуса документа."""

from __future__ import annotations

import pytest

from document_worker.domain.policies.document_status import DocumentStatusPolicy
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    DocumentStatus,
    ExtractionMethod,
    PageStatus,
)
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.quality import PageOutcome

pytestmark = pytest.mark.unit

POLICY = DocumentStatusPolicy()

ALL_PAGE_STATUSES = list(PageStatus)


def _outcome(
    number: int,
    status: PageStatus,
    *,
    method: ExtractionMethod | None = None,
    confidence: float | None = None,
    chars: int = 500,
) -> PageOutcome:
    if status is PageStatus.FAILED:
        return PageOutcome(
            page_number=PageNumber(number),
            status=status,
            method=ExtractionMethod.NONE,
            confidence=None,
            char_count=0,
            illegible_char_count=0,
        )
    chosen = method or (
        ExtractionMethod.OCR if confidence is not None else ExtractionMethod.TEXT_LAYER
    )
    illegible = 0 if status is PageStatus.EXTRACTED else chars // 2
    return PageOutcome(
        page_number=PageNumber(number),
        status=status,
        method=chosen,
        confidence=None
        if chosen is ExtractionMethod.TEXT_LAYER
        else OcrConfidence(confidence if confidence is not None else 0.9),
        char_count=chars,
        illegible_char_count=illegible,
    )


def _clean(count: int) -> list[PageOutcome]:
    return [_outcome(number, PageStatus.EXTRACTED) for number in range(1, count + 1)]


def test_returns_processed_when_all_pages_extracted() -> None:
    verdict = POLICY.evaluate(_clean(10), declared_page_count=10)

    assert verdict.status is DocumentStatus.PROCESSED
    assert verdict.problem_pages == ()


def test_returns_failed_for_empty_outcome_list() -> None:
    verdict = POLICY.evaluate([], declared_page_count=0)

    assert verdict.status is DocumentStatus.FAILED
    assert "empty_document" in verdict.reasons


def test_returns_failed_when_page_count_does_not_match_declared() -> None:
    verdict = POLICY.evaluate(_clean(9), declared_page_count=10)

    assert verdict.status is DocumentStatus.FAILED
    assert "incomplete_page_set" in verdict.reasons


def test_returns_failed_when_half_of_pages_failed() -> None:
    outcomes = _clean(5) + [
        _outcome(number, PageStatus.FAILED) for number in range(6, 11)
    ]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.status is DocumentStatus.FAILED
    assert "too_many_failed_pages" in verdict.reasons


def test_returns_failed_when_too_few_usable_pages() -> None:
    outcomes = _clean(2) + [
        _outcome(number, PageStatus.ILLEGIBLE) for number in range(3, 11)
    ]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.status is DocumentStatus.FAILED
    assert "too_few_usable_pages" in verdict.reasons


def test_returns_failed_when_document_has_almost_no_text() -> None:
    outcomes = [
        _outcome(number, PageStatus.EXTRACTED, chars=10) for number in range(1, 6)
    ]

    verdict = POLICY.evaluate(outcomes, declared_page_count=5)

    assert verdict.status is DocumentStatus.FAILED
    assert "no_extractable_text" in verdict.reasons


def test_returns_partially_processed_for_one_illegible_page_out_of_ten() -> None:
    outcomes = [*_clean(9), _outcome(10, PageStatus.ILLEGIBLE)]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.status is DocumentStatus.PARTIALLY_PROCESSED
    assert verdict.illegible_pages == (PageNumber(10),)


def test_returns_partially_processed_for_one_failed_page_out_of_ten() -> None:
    outcomes = [*_clean(9), _outcome(10, PageStatus.FAILED)]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.status is DocumentStatus.PARTIALLY_PROCESSED
    assert verdict.failed_pages == (PageNumber(10),)
    assert "failed_pages" in verdict.reasons


def test_partially_illegible_page_lands_in_its_own_category() -> None:
    outcomes = [*_clean(9), _outcome(10, PageStatus.PARTIALLY_ILLEGIBLE)]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.partially_illegible_pages == (PageNumber(10),)
    assert verdict.status is DocumentStatus.PARTIALLY_PROCESSED


@pytest.mark.parametrize(
    "status", [PageStatus.PARTIALLY_ILLEGIBLE, PageStatus.ILLEGIBLE, PageStatus.FAILED]
)
def test_never_returns_processed_when_any_page_is_not_extracted(
    status: PageStatus,
) -> None:
    outcomes = [*_clean(9), _outcome(10, status)]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.status is not DocumentStatus.PROCESSED


def test_low_mean_confidence_downgrades_to_partially_processed() -> None:
    outcomes = [
        _outcome(number, PageStatus.EXTRACTED, confidence=0.5)
        for number in range(1, 11)
    ]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.status is DocumentStatus.PARTIALLY_PROCESSED
    assert "low_mean_confidence" in verdict.reasons


def test_document_without_ocr_pages_is_not_downgraded_by_confidence() -> None:
    verdict = POLICY.evaluate(_clean(10), declared_page_count=10)

    assert verdict.stats.mean_ocr_confidence is None
    assert verdict.status is DocumentStatus.PROCESSED


def test_mean_confidence_ignores_text_layer_pages() -> None:
    outcomes = [*_clean(9), _outcome(10, PageStatus.EXTRACTED, confidence=0.8)]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.stats.mean_ocr_confidence == OcrConfidence(0.8)


def test_failed_pages_do_not_dilute_mean_confidence() -> None:
    outcomes = [
        _outcome(1, PageStatus.EXTRACTED, confidence=0.9),
        _outcome(2, PageStatus.FAILED),
    ]

    verdict = POLICY.evaluate(outcomes, declared_page_count=2)

    assert verdict.stats.mean_ocr_confidence == OcrConfidence(0.9)


def test_problem_pages_collect_all_non_extracted_page_numbers() -> None:
    outcomes = [
        _outcome(1, PageStatus.EXTRACTED),
        _outcome(2, PageStatus.PARTIALLY_ILLEGIBLE),
        _outcome(3, PageStatus.ILLEGIBLE),
        _outcome(4, PageStatus.FAILED),
        _outcome(5, PageStatus.EXTRACTED),
    ]

    verdict = POLICY.evaluate(outcomes, declared_page_count=5)

    assert verdict.problem_pages == (PageNumber(2), PageNumber(3), PageNumber(4))


def test_rules_are_evaluated_in_declared_order() -> None:
    # Документ разом нарушает правила 2, 3 и 4; побеждает первое по порядку.
    outcomes = [_outcome(number, PageStatus.FAILED) for number in range(1, 6)]

    verdict = POLICY.evaluate(outcomes, declared_page_count=10)

    assert verdict.reasons == ("incomplete_page_set",)


def test_declared_page_count_of_none_is_failure() -> None:
    verdict = POLICY.evaluate(_clean(3), declared_page_count=None)

    assert verdict.status is DocumentStatus.FAILED
    assert "empty_document" in verdict.reasons
