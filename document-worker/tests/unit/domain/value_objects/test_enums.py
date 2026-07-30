"""Тесты словарей домена: статусы, способы извлечения, причины неразборчивости."""

from __future__ import annotations

import pytest

from document_worker.domain.errors import InvalidStatusTransition
from document_worker.domain.value_objects.enums import (
    DocumentStatus,
    ExtractionMethod,
    IllegibleReason,
    JobStatus,
    PageStatus,
    ProcessingStage,
)

pytestmark = pytest.mark.unit

# Значения — контракт с колонками varchar + CHECK в схеме БД.
DOCUMENT_STATUSES = frozenset(
    {"pending", "processing", "processed", "partially_processed", "failed"}
)
PAGE_STATUSES = frozenset({"extracted", "partially_illegible", "illegible", "failed"})
EXTRACTION_METHODS = frozenset({"text_layer", "ocr", "hybrid", "none"})
JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed"})

ALLOWED_TRANSITIONS = [
    (DocumentStatus.PENDING, DocumentStatus.PROCESSING),
    (DocumentStatus.PENDING, DocumentStatus.FAILED),
    (DocumentStatus.PROCESSING, DocumentStatus.PROCESSED),
    (DocumentStatus.PROCESSING, DocumentStatus.PARTIALLY_PROCESSED),
    (DocumentStatus.PROCESSING, DocumentStatus.FAILED),
    # Повторная обработка более новой версией пайплайна.
    (DocumentStatus.PROCESSED, DocumentStatus.PROCESSING),
    (DocumentStatus.PARTIALLY_PROCESSED, DocumentStatus.PROCESSING),
    (DocumentStatus.FAILED, DocumentStatus.PROCESSING),
]

FORBIDDEN_TRANSITIONS = [
    (DocumentStatus.PENDING, DocumentStatus.PROCESSED),
    (DocumentStatus.PENDING, DocumentStatus.PARTIALLY_PROCESSED),
    (DocumentStatus.PENDING, DocumentStatus.PENDING),
    (DocumentStatus.PROCESSING, DocumentStatus.PENDING),
    (DocumentStatus.PROCESSING, DocumentStatus.PROCESSING),
    (DocumentStatus.PROCESSED, DocumentStatus.FAILED),
    (DocumentStatus.PROCESSED, DocumentStatus.PARTIALLY_PROCESSED),
    (DocumentStatus.FAILED, DocumentStatus.PROCESSED),
    (DocumentStatus.PARTIALLY_PROCESSED, DocumentStatus.PROCESSED),
]


def test_document_status_vocabulary_matches_dictionary() -> None:
    assert {status.value for status in DocumentStatus} == DOCUMENT_STATUSES


def test_page_status_vocabulary_matches_dictionary() -> None:
    assert {status.value for status in PageStatus} == PAGE_STATUSES


def test_extraction_method_vocabulary_matches_dictionary() -> None:
    assert {method.value for method in ExtractionMethod} == EXTRACTION_METHODS


def test_job_status_vocabulary_matches_dictionary() -> None:
    assert {status.value for status in JobStatus} == JOB_STATUSES


def test_extraction_method_has_no_mixed_synonym() -> None:
    assert "mixed" not in {method.value for method in ExtractionMethod}, (
        "словарь способов извлечения единственный, синонимов у hybrid нет"
    )


@pytest.mark.parametrize(("source", "target"), ALLOWED_TRANSITIONS)
def test_document_status_transition_table_allows_declared_transitions(
    source: DocumentStatus,
    target: DocumentStatus,
) -> None:
    assert source.can_transition_to(target)
    source.ensure_can_transition_to(target)


@pytest.mark.parametrize(("source", "target"), FORBIDDEN_TRANSITIONS)
def test_document_status_transition_table_rejects_other_transitions(
    source: DocumentStatus,
    target: DocumentStatus,
) -> None:
    assert not source.can_transition_to(target)
    with pytest.raises(InvalidStatusTransition):
        source.ensure_can_transition_to(target)


def test_terminal_statuses_are_exactly_three() -> None:
    terminal = {status for status in DocumentStatus if status.is_terminal}

    assert terminal == {
        DocumentStatus.PROCESSED,
        DocumentStatus.PARTIALLY_PROCESSED,
        DocumentStatus.FAILED,
    }


def test_successful_statuses_exclude_failed() -> None:
    successful = {status for status in DocumentStatus if status.is_successful}

    assert successful == {
        DocumentStatus.PROCESSED,
        DocumentStatus.PARTIALLY_PROCESSED,
    }


def test_partially_illegible_page_is_usable_but_not_fully_read() -> None:
    assert PageStatus.PARTIALLY_ILLEGIBLE.is_usable
    assert not PageStatus.PARTIALLY_ILLEGIBLE.is_fully_read
    assert PageStatus.EXTRACTED.is_fully_read


@pytest.mark.parametrize(
    "status",
    [PageStatus.ILLEGIBLE, PageStatus.FAILED],
)
def test_illegible_and_failed_pages_are_not_usable(status: PageStatus) -> None:
    assert not status.is_usable


def test_only_ocr_and_hybrid_are_ocr_based() -> None:
    ocr_based = {method for method in ExtractionMethod if method.is_ocr_based}

    assert ocr_based == {ExtractionMethod.OCR, ExtractionMethod.HYBRID}


def test_only_none_method_yields_no_text() -> None:
    without_text = {method for method in ExtractionMethod if not method.yields_text}

    assert without_text == {ExtractionMethod.NONE}


def test_technical_illegible_reasons_are_exactly_three() -> None:
    technical = {reason for reason in IllegibleReason if reason.is_technical}

    assert technical == {
        IllegibleReason.PAGE_RENDER_FAILED,
        IllegibleReason.OCR_FAILED,
        IllegibleReason.PAGE_TIMEOUT,
    }


def test_handwriting_is_a_content_reason_not_a_technical_one() -> None:
    assert not IllegibleReason.HANDWRITING.is_technical, (
        "рукопись помечается неразборчивой, это не сбой обработки"
    )


def test_processing_stage_covers_whole_pipeline() -> None:
    assert {stage.value for stage in ProcessingStage} == {
        "download",
        "validation",
        "inspection",
        "text_extraction",
        "rendering",
        "ocr",
        "normalization",
        "chunking",
        "persistence",
        "publishing",
    }
