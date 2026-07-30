"""Тесты доменных событий."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from document_worker.domain.errors import InvariantViolation
from document_worker.domain.events import (
    DocumentPartiallyProcessed,
    DocumentProcessed,
    DocumentProcessingFailed,
)
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ProcessingStage
from document_worker.domain.value_objects.identifiers import (
    CorrelationId,
    DocumentId,
    EventId,
)
from document_worker.domain.value_objects.versioning import PipelineVersion

if TYPE_CHECKING:
    from document_worker.domain.events import DomainEvent

pytestmark = pytest.mark.unit

DOCUMENT_ID = DocumentId(uuid.UUID("11111111-1111-5111-9111-111111111111"))
CORRELATION_ID = CorrelationId("req-2026-07-30")
PIPELINE_VERSION = PipelineVersion(1, 0, 0)
OCCURRED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

ROUTING_KEYS = {
    DocumentProcessed: "document.processed",
    DocumentPartiallyProcessed: "document.partially_processed",
    DocumentProcessingFailed: "document.processing.failed",
}


COMMON: dict[str, Any] = {
    "document_id": DOCUMENT_ID,
    "correlation_id": CORRELATION_ID,
    "pipeline_version": PIPELINE_VERSION,
    "occurred_at": OCCURRED_AT,
}


def _processed(**overrides: Any) -> DocumentProcessed:
    fields: dict[str, Any] = {
        **COMMON,
        "pages_total": 4,
        "pages_text_layer": 2,
        "pages_ocr": 1,
        "pages_hybrid": 1,
        "pages_failed": 0,
        "chunks_total": 12,
        "total_chars": 4000,
        "mean_ocr_confidence": OcrConfidence(0.9),
        "ocr_coverage": 0.5,
        "processing_duration_ms": 1500,
    }
    fields.update(overrides)
    return DocumentProcessed(**fields)


def _partially(**overrides: Any) -> DocumentPartiallyProcessed:
    fields: dict[str, Any] = {
        **COMMON,
        "pages_total": 4,
        "pages_text_layer": 2,
        "pages_ocr": 1,
        "pages_hybrid": 0,
        "pages_failed": 1,
        "chunks_total": 9,
        "total_chars": 3000,
        "mean_ocr_confidence": OcrConfidence(0.7),
        "ocr_coverage": 0.25,
        "processing_duration_ms": 2500,
        "partially_illegible_page_numbers": (),
        "illegible_page_numbers": (),
        "failed_page_numbers": (4,),
        "illegible_char_ratio": 0.0,
        "reasons": ("одна страница не прочитана",),
    }
    fields.update(overrides)
    return DocumentPartiallyProcessed(**fields)


def _failed(**overrides: Any) -> DocumentProcessingFailed:
    fields: dict[str, Any] = {
        **COMMON,
        "error_code": "corrupted_document",
        "error_message": "не удалось открыть PDF",
        "stage": ProcessingStage.INSPECTION,
        "attempt": 1,
    }
    fields.update(overrides)
    return DocumentProcessingFailed(**fields)


@pytest.mark.parametrize(("event_type", "routing_key"), ROUTING_KEYS.items())
def test_routing_keys_match_contract(
    event_type: type[DomainEvent], routing_key: str
) -> None:
    assert event_type.event_type == routing_key


@pytest.mark.parametrize("build", [_processed, _partially, _failed])
def test_event_rejects_naive_datetime(build: Any) -> None:
    with pytest.raises(InvariantViolation):
        build(occurred_at=datetime(2026, 7, 30, 12, 0))  # noqa: DTZ001


@pytest.mark.parametrize("build", [_processed, _partially, _failed])
def test_event_rejects_non_utc_timezone(build: Any) -> None:
    tehran = datetime(2026, 7, 30, 12, 0, tzinfo=UTC).astimezone()
    if tehran.utcoffset() == timedelta(0):
        pytest.skip("локальная зона совпадает с UTC")

    with pytest.raises(InvariantViolation):
        build(occurred_at=tehran.replace(tzinfo=tehran.tzinfo))


def test_page_counters_sum_to_total_including_hybrid() -> None:
    event = _processed(pages_text_layer=1, pages_ocr=1, pages_hybrid=2, pages_failed=0)

    assert (
        event.pages_text_layer
        + event.pages_ocr
        + event.pages_hybrid
        + event.pages_failed
        == event.pages_total
    )


@pytest.mark.parametrize("build", [_processed, _partially])
def test_event_rejects_counters_that_do_not_sum(build: Any) -> None:
    with pytest.raises(InvariantViolation):
        build(pages_text_layer=1, pages_ocr=0, pages_hybrid=0, pages_failed=0)


def test_processed_event_forbids_failed_pages() -> None:
    with pytest.raises(InvariantViolation):
        _processed(pages_text_layer=2, pages_ocr=1, pages_hybrid=0, pages_failed=1)


def test_partially_processed_event_requires_non_empty_problem_pages() -> None:
    with pytest.raises(InvariantViolation):
        _partially(
            pages_text_layer=2,
            pages_ocr=2,
            pages_hybrid=0,
            pages_failed=0,
            partially_illegible_page_numbers=(),
            illegible_page_numbers=(),
            failed_page_numbers=(),
        )


@pytest.mark.parametrize(
    "problem_pages",
    [
        {"partially_illegible_page_numbers": (2,)},
        {"illegible_page_numbers": (3,)},
        {"failed_page_numbers": (4,)},
    ],
)
def test_partially_processed_accepts_any_kind_of_problem_page(
    problem_pages: dict[str, tuple[int, ...]],
) -> None:
    no_problem_pages: dict[str, tuple[int, ...]] = {
        "partially_illegible_page_numbers": (),
        "illegible_page_numbers": (),
        "failed_page_numbers": (),
    }

    event = _partially(
        pages_text_layer=3,
        pages_ocr=1,
        pages_hybrid=0,
        pages_failed=0,
        **{**no_problem_pages, **problem_pages},
    )

    assert event.event_type == "document.partially_processed"


def test_event_id_is_deterministic_across_two_completions() -> None:
    assert _processed().event_id == _processed().event_id


def test_event_id_differs_between_event_types() -> None:
    assert _processed().event_id != _partially().event_id


def test_event_id_matches_deterministic_formula() -> None:
    expected = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processed",
    )

    assert _processed().event_id == expected


def test_failed_event_carries_error_details() -> None:
    event = _failed()

    assert event.error_code == "corrupted_document"
    assert event.stage is ProcessingStage.INSPECTION


def test_failed_event_has_no_retryability_flag() -> None:
    forbidden = {"retryable", "is_retryable", "recoverable", "transient"}

    assert not forbidden & set(dir(DocumentProcessingFailed))


def test_event_is_immutable() -> None:
    event = _processed()

    with pytest.raises(AttributeError):
        event.pages_total = 10  # type: ignore[misc]
