"""Тесты агрегата документа."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from document_worker.domain.entities.document import Document
from document_worker.domain.errors import (
    DocumentTooLarge,
    EmptyDocument,
    IncompletePageSet,
    InvalidStatusTransition,
    InvariantViolation,
)
from document_worker.domain.events import (
    DocumentPartiallyProcessed,
    DocumentProcessed,
    DocumentProcessingFailed,
)
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    CompletionOutcome,
    DocumentStatus,
    ExtractionMethod,
    PageStatus,
    ProcessingStage,
)
from document_worker.domain.value_objects.identifiers import CorrelationId, DocumentId
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.quality import (
    DocumentQualityStats,
    DocumentStatusVerdict,
    PageOutcome,
)
from document_worker.domain.value_objects.storage import (
    FileSize,
    MimeType,
    ObjectRef,
    SourceFile,
)
from document_worker.domain.value_objects.versioning import PipelineVersion

pytestmark = pytest.mark.unit

DOCUMENT_ID = DocumentId(uuid.UUID("11111111-1111-5111-9111-111111111111"))
CORRELATION_ID = CorrelationId("req-2026-07-30")
VERSION = PipelineVersion(1, 0, 0)
NEWER_VERSION = PipelineVersion(2, 0, 0)
CREATED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
STARTED_AT = CREATED_AT + timedelta(seconds=1)
FINISHED_AT = CREATED_AT + timedelta(seconds=5)

SOURCE = SourceFile(
    ref=ObjectRef(bucket="documents", key="a/source.pdf"),
    mime_type=MimeType("application/pdf"),
    size=FileSize(2048),
)


def _document(status: DocumentStatus = DocumentStatus.PENDING) -> Document:
    return Document(
        id=DOCUMENT_ID,
        source=SOURCE,
        status=status,
        pipeline_version=VERSION,
        correlation_id=CORRELATION_ID,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _outcome(number: int, status: PageStatus, method: ExtractionMethod) -> PageOutcome:
    confidence = None if method is ExtractionMethod.NONE else OcrConfidence(0.9)
    if method is ExtractionMethod.TEXT_LAYER:
        confidence = None
    return PageOutcome(
        page_number=PageNumber(number),
        status=status,
        method=method,
        confidence=confidence,
        char_count=0 if method is ExtractionMethod.NONE else 100,
        illegible_char_count=0,
    )


def _stats(*outcomes: PageOutcome) -> DocumentQualityStats:
    return DocumentQualityStats.from_outcomes(outcomes)


def _success_verdict(pages: int = 2) -> DocumentStatusVerdict:
    outcomes = [
        _outcome(number, PageStatus.EXTRACTED, ExtractionMethod.TEXT_LAYER)
        for number in range(1, pages + 1)
    ]
    return DocumentStatusVerdict(
        status=DocumentStatus.PROCESSED,
        stats=_stats(*outcomes),
        reasons=(),
    )


def _partial_verdict() -> DocumentStatusVerdict:
    outcomes = [
        _outcome(1, PageStatus.EXTRACTED, ExtractionMethod.TEXT_LAYER),
        _outcome(2, PageStatus.FAILED, ExtractionMethod.NONE),
    ]
    return DocumentStatusVerdict(
        status=DocumentStatus.PARTIALLY_PROCESSED,
        stats=_stats(*outcomes),
        reasons=("страница 2 не прочитана",),
        failed_pages=(PageNumber(2),),
    )


def _processing(pages: int = 2) -> Document:
    document = _document()
    document.start_processing(now=STARTED_AT)
    document.declare_page_count(pages)
    return document


def test_document_has_no_register_factory() -> None:
    assert not hasattr(Document, "register"), "строку documents создаёт api-service"


@pytest.mark.parametrize("attribute", ["pages", "chunks", "_pages", "_chunks"])
def test_document_aggregate_has_no_page_collection(attribute: str) -> None:
    assert attribute not in set(dir(_document()))


def test_start_processing_moves_status_to_processing() -> None:
    document = _document()

    document.start_processing(now=STARTED_AT)

    assert document.status is DocumentStatus.PROCESSING
    assert document.processing_started_at == STARTED_AT


def test_start_processing_twice_raises_invalid_status_transition() -> None:
    document = _document()
    document.start_processing(now=STARTED_AT)

    with pytest.raises(InvalidStatusTransition):
        document.start_processing(now=STARTED_AT)


def test_start_processing_on_terminal_without_newer_version_raises() -> None:
    document = _processing()
    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)

    with pytest.raises(InvalidStatusTransition):
        document.start_processing(now=FINISHED_AT, pipeline_version=VERSION)


def test_start_processing_on_terminal_with_newer_version_is_allowed() -> None:
    document = _processing()
    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)

    document.start_processing(now=FINISHED_AT, pipeline_version=NEWER_VERSION)

    assert document.status is DocumentStatus.PROCESSING
    assert document.pipeline_version == NEWER_VERSION


def test_start_processing_resets_previous_result() -> None:
    document = _processing()
    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)
    document.pull_events()

    document.start_processing(now=FINISHED_AT, pipeline_version=NEWER_VERSION)

    assert document.processed_at is None
    assert document.stats is None
    assert document.failure_code is None


def test_declare_page_count_zero_raises_empty_document() -> None:
    document = _document()
    document.start_processing(now=STARTED_AT)

    with pytest.raises(EmptyDocument):
        document.declare_page_count(0)


def test_declare_page_count_above_limit_raises_document_too_large() -> None:
    document = _document()
    document.start_processing(now=STARTED_AT)

    with pytest.raises(DocumentTooLarge):
        document.declare_page_count(301)


def test_declare_page_count_conflicting_value_raises() -> None:
    document = _processing(pages=2)

    with pytest.raises(InvariantViolation):
        document.declare_page_count(3)


def test_declare_page_count_same_value_twice_is_allowed() -> None:
    document = _processing(pages=2)

    document.declare_page_count(2)

    assert document.page_count == 2


def test_complete_from_processing_sets_terminal_status_and_emits_one_event() -> None:
    document = _processing()

    outcome = document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)

    assert outcome is CompletionOutcome.APPLIED
    assert document.status is DocumentStatus.PROCESSED
    assert document.processed_at == FINISHED_AT
    events = document.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], DocumentProcessed)


def test_complete_emits_partially_processed_event_for_partial_verdict() -> None:
    document = _processing()

    document.complete(_partial_verdict(), chunks_total=2, now=FINISHED_AT)

    events = document.pull_events()
    assert isinstance(events[0], DocumentPartiallyProcessed)
    assert events[0].failed_page_numbers == (2,)


def test_complete_carries_processing_duration() -> None:
    document = _processing()

    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)

    event = document.pull_events()[0]
    assert isinstance(event, DocumentProcessed)
    assert event.processing_duration_ms == 4000


def test_duration_is_zero_when_start_time_is_unknown() -> None:
    # Документ, восстановленный из БД без отметки старта.
    document = _document(DocumentStatus.PROCESSING)
    document.declare_page_count(2)

    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)

    event = document.pull_events()[0]
    assert isinstance(event, DocumentProcessed)
    assert event.processing_duration_ms == 0


def test_complete_requires_declared_page_count() -> None:
    document = _document()
    document.start_processing(now=STARTED_AT)

    with pytest.raises(IncompletePageSet):
        document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)


def test_complete_requires_stats_matching_page_count() -> None:
    document = _processing(pages=3)

    with pytest.raises(IncompletePageSet):
        document.complete(_success_verdict(pages=2), chunks_total=4, now=FINISHED_AT)


def test_complete_on_already_terminal_document_is_idempotent_no_op() -> None:
    document = _processing()
    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)
    document.pull_events()

    outcome = document.complete(
        _success_verdict(), chunks_total=4, now=FINISHED_AT + timedelta(seconds=10)
    )

    assert outcome is CompletionOutcome.DUPLICATE
    assert document.processed_at == FINISHED_AT
    assert document.pull_events() == ()


def test_complete_from_pending_raises_invalid_status_transition() -> None:
    document = _document()

    with pytest.raises(InvalidStatusTransition):
        document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)


def test_fail_from_processing_sets_failed_and_emits_event() -> None:
    document = _processing()

    outcome = document.fail(
        code="corrupted_document",
        message="не удалось открыть PDF",
        stage=ProcessingStage.INSPECTION,
        now=FINISHED_AT,
    )

    assert outcome is CompletionOutcome.APPLIED
    assert document.status is DocumentStatus.FAILED
    assert document.failure_code == "corrupted_document"
    assert document.failure_stage is ProcessingStage.INSPECTION
    assert isinstance(document.pull_events()[0], DocumentProcessingFailed)


def test_fail_from_pending_is_allowed() -> None:
    document = _document()

    document.fail(
        code="unsupported_format",
        message="не PDF",
        stage=ProcessingStage.VALIDATION,
        now=FINISHED_AT,
    )

    assert document.status is DocumentStatus.FAILED


def test_fail_on_already_processed_document_is_duplicate_no_op() -> None:
    document = _processing()
    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)
    document.pull_events()

    outcome = document.fail(
        code="whatever",
        message="второй воркер догнал первого",
        stage=ProcessingStage.PERSISTENCE,
        now=FINISHED_AT + timedelta(seconds=10),
    )

    assert outcome is CompletionOutcome.DUPLICATE
    assert document.status is DocumentStatus.PROCESSED
    assert document.failure_code is None
    assert document.pull_events() == ()


def test_pull_events_clears_internal_buffer() -> None:
    document = _processing()
    document.complete(_success_verdict(), chunks_total=4, now=FINISHED_AT)

    assert len(document.pull_events()) == 1
    assert document.pull_events() == ()


def test_naive_datetime_argument_raises_invariant_violation() -> None:
    document = _document()

    with pytest.raises(InvariantViolation):
        document.start_processing(now=datetime(2026, 7, 30, 12, 0))  # noqa: DTZ001


def test_updated_at_before_created_at_raises() -> None:
    with pytest.raises(InvariantViolation):
        Document(
            id=DOCUMENT_ID,
            source=SOURCE,
            status=DocumentStatus.PENDING,
            pipeline_version=VERSION,
            correlation_id=CORRELATION_ID,
            created_at=CREATED_AT,
            updated_at=CREATED_AT - timedelta(seconds=1),
        )


def test_created_at_must_be_timezone_aware() -> None:
    with pytest.raises(InvariantViolation):
        Document(
            id=DOCUMENT_ID,
            source=SOURCE,
            status=DocumentStatus.PENDING,
            pipeline_version=VERSION,
            correlation_id=CORRELATION_ID,
            created_at=datetime(2026, 7, 30, 12, 0),  # noqa: DTZ001
            updated_at=CREATED_AT,
        )
