"""Тесты прогона обработки документа."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from document_worker.domain.entities.processing_job import ProcessingJob
from document_worker.domain.errors import InvalidStatusTransition, InvariantViolation
from document_worker.domain.value_objects.enums import (
    CompletionOutcome,
    DocumentStatus,
    JobStatus,
    ProcessingStage,
)
from document_worker.domain.value_objects.identifiers import (
    CorrelationId,
    DocumentId,
    EventId,
    JobId,
)
from document_worker.domain.value_objects.versioning import PipelineVersion

pytestmark = pytest.mark.unit

JOB_ID = JobId(uuid.UUID("66666666-6666-5666-9666-666666666666"))
DOCUMENT_ID = DocumentId(uuid.UUID("11111111-1111-5111-9111-111111111111"))
EVENT_ID = EventId(uuid.UUID("77777777-7777-5777-9777-777777777777"))
CORRELATION_ID = CorrelationId("req-2026-07-30")
VERSION = PipelineVersion(1, 0, 0)

SCHEDULED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
STARTED_AT = SCHEDULED_AT + timedelta(seconds=1)
FINISHED_AT = SCHEDULED_AT + timedelta(seconds=6)


def _scheduled(attempt: int = 1) -> ProcessingJob:
    return ProcessingJob.schedule(
        job_id=JOB_ID,
        document_id=DOCUMENT_ID,
        event_id=EVENT_ID,
        correlation_id=CORRELATION_ID,
        pipeline_version=VERSION,
        now=SCHEDULED_AT,
        attempt=attempt,
    )


def _running(pages: int | None = 2) -> ProcessingJob:
    job = _scheduled()
    job.start(now=STARTED_AT)
    if pages is not None:
        job.declare_pages(pages)
    return job


def test_schedule_creates_queued_job() -> None:
    job = _scheduled()

    assert job.status is JobStatus.QUEUED
    assert job.attempt == 1
    assert job.started_at is None
    assert not job.is_terminal


def test_schedule_rejects_attempt_below_one() -> None:
    with pytest.raises(InvariantViolation):
        _scheduled(attempt=0)


def test_job_has_no_retryability_flag() -> None:
    forbidden = {"retryable", "is_retryable", "recoverable", "transient"}

    assert not forbidden & set(dir(ProcessingJob))


def test_start_moves_job_to_running() -> None:
    job = _scheduled()

    job.start(now=STARTED_AT)

    assert job.status is JobStatus.RUNNING
    assert job.started_at == STARTED_AT


def test_start_twice_raises_invalid_status_transition() -> None:
    job = _running()

    with pytest.raises(InvalidStatusTransition):
        job.start(now=STARTED_AT)


def test_start_before_scheduling_moment_raises() -> None:
    job = _scheduled()

    with pytest.raises(InvariantViolation):
        job.start(now=SCHEDULED_AT - timedelta(seconds=1))


def test_succeed_from_queued_raises_invalid_status_transition() -> None:
    job = _scheduled()

    with pytest.raises(InvalidStatusTransition):
        job.succeed(result=DocumentStatus.PROCESSED, now=FINISHED_AT)


def test_succeed_rejects_non_successful_result() -> None:
    job = _running()

    with pytest.raises(InvariantViolation):
        job.succeed(result=DocumentStatus.FAILED, now=FINISHED_AT)


def test_succeed_sets_terminal_state() -> None:
    job = _running()

    outcome = job.succeed(result=DocumentStatus.PROCESSED, now=FINISHED_AT)

    assert outcome is CompletionOutcome.APPLIED
    assert job.status is JobStatus.SUCCEEDED
    assert job.result_status is DocumentStatus.PROCESSED
    assert job.finished_at == FINISHED_AT
    assert job.is_terminal


def test_succeed_twice_is_duplicate_no_op() -> None:
    job = _running()
    job.succeed(result=DocumentStatus.PROCESSED, now=FINISHED_AT)

    outcome = job.succeed(
        result=DocumentStatus.PARTIALLY_PROCESSED,
        now=FINISHED_AT + timedelta(seconds=5),
    )

    assert outcome is CompletionOutcome.DUPLICATE
    assert job.result_status is DocumentStatus.PROCESSED
    assert job.finished_at == FINISHED_AT


def test_fail_on_succeeded_job_is_duplicate_no_op() -> None:
    job = _running()
    job.succeed(result=DocumentStatus.PROCESSED, now=FINISHED_AT)

    outcome = job.fail(
        code="ocr_failed",
        message="движок упал",
        stage=ProcessingStage.OCR,
        now=FINISHED_AT + timedelta(seconds=5),
    )

    assert outcome is CompletionOutcome.DUPLICATE
    assert job.status is JobStatus.SUCCEEDED
    assert job.error_code is None


def test_fail_sets_error_details() -> None:
    job = _running()

    outcome = job.fail(
        code="ocr_failed",
        message="движок упал",
        stage=ProcessingStage.OCR,
        now=FINISHED_AT,
    )

    assert outcome is CompletionOutcome.APPLIED
    assert job.status is JobStatus.FAILED
    assert job.error_code == "ocr_failed"
    assert job.stage is ProcessingStage.OCR


def test_fail_from_queued_is_allowed() -> None:
    job = _scheduled()

    job.fail(
        code="document_not_found",
        message="объекта нет в хранилище",
        stage=ProcessingStage.DOWNLOAD,
        now=FINISHED_AT,
    )

    assert job.status is JobStatus.FAILED


def test_finished_at_is_not_before_started_at() -> None:
    job = _running()

    with pytest.raises(InvariantViolation):
        job.succeed(
            result=DocumentStatus.PROCESSED, now=STARTED_AT - timedelta(seconds=1)
        )


def test_declare_pages_rejects_negative_total() -> None:
    job = _running(pages=None)

    with pytest.raises(InvariantViolation):
        job.declare_pages(-1)


def test_declare_pages_conflicting_value_raises() -> None:
    job = _running(pages=2)

    with pytest.raises(InvariantViolation):
        job.declare_pages(3)


def test_record_pages_splits_counters_by_extraction_method() -> None:
    job = _running(pages=None)
    job.declare_pages(4)

    job.record_pages(text_layer=1, ocr=1, hybrid=1, failed=1)

    assert job.pages_text_layer == 1
    assert job.pages_ocr == 1
    assert job.pages_hybrid == 1
    assert job.pages_failed == 1
    assert job.pages_done == 3


def test_record_pages_replaces_the_running_tally() -> None:
    # Итог считается по сохранённым страницам, а не по тому, что успел
    # насчитать прогресс: воркер мог продолжить чужую работу.
    job = _running()
    job.record_pages(text_layer=1, ocr=0, hybrid=0, failed=0)

    job.record_pages(text_layer=2, ocr=0, hybrid=0, failed=0)

    assert job.pages_text_layer == 2


def test_record_pages_beyond_declared_total_raises() -> None:
    job = _running(pages=None)
    job.declare_pages(1)

    with pytest.raises(InvariantViolation):
        job.record_pages(text_layer=2, ocr=0, hybrid=0, failed=0)


def test_duration_is_none_until_finished() -> None:
    job = _running()

    assert job.duration() is None


def test_duration_covers_processing_time() -> None:
    job = _running()
    job.succeed(result=DocumentStatus.PROCESSED, now=FINISHED_AT)

    assert job.duration() == timedelta(seconds=5)


def test_scheduled_at_must_be_timezone_aware() -> None:
    with pytest.raises(InvariantViolation):
        ProcessingJob.schedule(
            job_id=JOB_ID,
            document_id=DOCUMENT_ID,
            event_id=EVENT_ID,
            correlation_id=CORRELATION_ID,
            pipeline_version=VERSION,
            now=datetime(2026, 7, 30, 12, 0),  # noqa: DTZ001
        )
