"""Репозиторий прогонов: заведение, прогресс и терминальный переход."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from document_worker.application.dto.results import JobProgressDTO
from document_worker.domain.value_objects.enums import JobStatus, ProcessingStage
from document_worker.domain.value_objects.versioning import PipelineVersion
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.repositories.jobs import (
    SqlAlchemyProcessingJobRepository,
)
from tests.factories import NOW, PIPELINE_VERSION, make_document, make_job

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.domain.entities.document import Document

pytestmark = pytest.mark.integration

LATER = NOW + timedelta(minutes=5)
NEWER_VERSION = PipelineVersion(2, 0, 0)


async def _document(session: AsyncSession) -> Document:
    document = make_document()
    session.add(document_to_row(document))
    await session.flush()
    return document


async def test_start_creates_the_job(session: AsyncSession) -> None:
    document = await _document(session)
    repository = SqlAlchemyProcessingJobRepository(session)
    job = make_job(document, status=JobStatus.QUEUED)

    started = await repository.start(job)

    assert started.id == job.id
    stored = await repository.get(document.id, PIPELINE_VERSION)
    assert stored == replace(job, result_status=None)


async def test_start_returns_the_existing_job_on_conflict(
    session: AsyncSession,
) -> None:
    # Два воркера получили дубликат сообщения и оба вставляют прогон: второй
    # обязан увидеть первый, а не упасть.
    document = await _document(session)
    repository = SqlAlchemyProcessingJobRepository(session)
    first = await repository.start(make_job(document, status=JobStatus.QUEUED))

    second = await repository.start(make_job(document, status=JobStatus.QUEUED))

    assert second.id == first.id


async def test_get_returns_none_for_other_pipeline_version(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyProcessingJobRepository(session)
    await repository.start(make_job(document, status=JobStatus.QUEUED))

    assert await repository.get(document.id, NEWER_VERSION) is None


async def test_record_progress_writes_counters_and_heartbeat(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyProcessingJobRepository(session)
    job = await repository.start(make_job(document, status=JobStatus.QUEUED))

    await repository.record_progress(
        job.id,
        JobProgressDTO(
            pages_text_layer=1,
            pages_ocr=0,
            pages_hybrid=0,
            pages_failed=1,
            chunks_created=9,
            heartbeat_at=LATER,
        ),
    )

    stored = await repository.get(document.id, PIPELINE_VERSION)
    assert stored is not None
    assert (stored.pages_text_layer, stored.pages_ocr, stored.pages_hybrid) == (1, 0, 0)
    assert stored.pages_failed == 1
    assert stored.chunks_created == 9


async def test_finish_writes_terminal_status(session: AsyncSession) -> None:
    document = await _document(session)
    repository = SqlAlchemyProcessingJobRepository(session)
    job = await repository.start(make_job(document, status=JobStatus.QUEUED))
    finished = replace(
        job,
        status=JobStatus.SUCCEEDED,
        started_at=NOW,
        finished_at=LATER,
        pages_total=2,
        pages_text_layer=1,
        pages_ocr=1,
        pages_hybrid=0,
        pages_failed=0,
    )

    applied = await repository.finish(finished, expected=JobStatus.QUEUED)

    assert applied
    stored = await repository.get(document.id, PIPELINE_VERSION)
    assert stored is not None
    assert stored.status is JobStatus.SUCCEEDED
    assert stored.finished_at == LATER


async def test_finish_with_wrong_expected_status_changes_nothing(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyProcessingJobRepository(session)
    job = await repository.start(make_job(document, status=JobStatus.QUEUED))
    failed = replace(
        job,
        status=JobStatus.FAILED,
        finished_at=LATER,
        error_code="timeout",
        error_message="не уложились",
        stage=ProcessingStage.OCR,
    )

    applied = await repository.finish(failed, expected=JobStatus.RUNNING)

    assert not applied
    stored = await repository.get(document.id, PIPELINE_VERSION)
    assert stored is not None
    assert stored.status is JobStatus.QUEUED


async def test_finish_writes_failure_triple(session: AsyncSession) -> None:
    document = await _document(session)
    repository = SqlAlchemyProcessingJobRepository(session)
    job = await repository.start(make_job(document, status=JobStatus.QUEUED))
    failed = replace(
        job,
        status=JobStatus.FAILED,
        finished_at=LATER,
        error_code="corrupted_document",
        error_message="файл не читается",
        stage=ProcessingStage.VALIDATION,
    )

    await repository.finish(failed, expected=JobStatus.QUEUED)

    stored = await repository.get(document.id, PIPELINE_VERSION)
    assert stored is not None
    assert stored.stage is ProcessingStage.VALIDATION
    assert stored.error_code == "corrupted_document"
