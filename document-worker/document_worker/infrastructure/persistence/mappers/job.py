"""Прогон ↔ строка processing_jobs."""

from __future__ import annotations

from document_worker.domain.entities.processing_job import ProcessingJob
from document_worker.domain.value_objects.enums import JobStatus, ProcessingStage
from document_worker.domain.value_objects.identifiers import (
    CorrelationId,
    DocumentId,
    EventId,
    JobId,
)
from document_worker.domain.value_objects.versioning import PipelineVersion
from document_worker.infrastructure.persistence.models.job import ProcessingJobRow


def job_to_row(job: ProcessingJob) -> ProcessingJobRow:
    """Собирает строку прогона."""
    return ProcessingJobRow(
        id=job.id.value,
        document_id=job.document_id.value,
        pipeline_version=str(job.pipeline_version),
        status=job.status.value,
        attempt=job.attempt,
        trigger_event_id=job.event_id.value,
        correlation_id=str(job.correlation_id),
        pages_total=job.pages_total,
        pages_text_layer=job.pages_text_layer,
        pages_ocr=job.pages_ocr,
        pages_hybrid=job.pages_hybrid,
        pages_failed=job.pages_failed,
        chunks_created=job.chunks_created,
        failure_code=job.error_code,
        failure_stage=job.stage.value if job.stage else None,
        failure_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.scheduled_at,
        updated_at=job.finished_at or job.started_at or job.scheduled_at,
    )


def apply_job_to_row(job: ProcessingJob, row: ProcessingJobRow) -> None:
    """Переносит в существующую строку изменяемые колонки прогона."""
    row.status = job.status.value
    row.attempt = job.attempt
    row.pages_total = job.pages_total
    row.pages_text_layer = job.pages_text_layer
    row.pages_ocr = job.pages_ocr
    row.pages_hybrid = job.pages_hybrid
    row.pages_failed = job.pages_failed
    row.chunks_created = job.chunks_created
    row.failure_code = job.error_code
    row.failure_stage = job.stage.value if job.stage else None
    row.failure_message = job.error_message
    row.started_at = job.started_at
    row.finished_at = job.finished_at
    row.updated_at = job.finished_at or job.started_at or job.scheduled_at


def job_to_domain(row: ProcessingJobRow) -> ProcessingJob:
    """Восстанавливает прогон из строки.

    Статус документа-результата не восстанавливается: его владелец — строка
    documents, дублировать его в прогоне нечем и незачем.
    """
    return ProcessingJob(
        id=JobId(row.id),
        document_id=DocumentId(row.document_id),
        event_id=EventId(row.trigger_event_id),
        correlation_id=CorrelationId(row.correlation_id),
        pipeline_version=PipelineVersion.parse(row.pipeline_version),
        status=JobStatus(row.status),
        attempt=row.attempt,
        scheduled_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        pages_total=row.pages_total,
        pages_text_layer=row.pages_text_layer,
        pages_ocr=row.pages_ocr,
        pages_hybrid=row.pages_hybrid,
        pages_failed=row.pages_failed,
        chunks_created=row.chunks_created,
        error_code=row.failure_code,
        error_message=row.failure_message,
        stage=ProcessingStage(row.failure_stage) if row.failure_stage else None,
    )
