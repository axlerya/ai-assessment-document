"""Доступ к прогонам обработки."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from document_worker.infrastructure.persistence.mappers.job import (
    job_to_domain,
    job_to_values,
)
from document_worker.infrastructure.persistence.models.job import ProcessingJobRow
from document_worker.infrastructure.persistence.repositories.base import (
    SqlAlchemyRepository,
)

if TYPE_CHECKING:
    from document_worker.application.dto.results import JobProgressDTO
    from document_worker.domain.entities.processing_job import ProcessingJob
    from document_worker.domain.value_objects.enums import JobStatus
    from document_worker.domain.value_objects.identifiers import DocumentId, JobId
    from document_worker.domain.value_objects.versioning import PipelineVersion

JOB_CONSTRAINT = "uq__processing_jobs__document__version"


class SqlAlchemyProcessingJobRepository(SqlAlchemyRepository):
    """Прогоны: заведение с гашением дубля, прогресс и терминальный переход."""

    async def get(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> ProcessingJob | None:
        """Читает прогон документа для указанной версии пайплайна."""
        row = await self._row_of(document_id, pipeline_version)
        return None if row is None else job_to_domain(row)

    async def start(self, job: ProcessingJob) -> ProcessingJob:
        """Заводит прогон; при конфликте возвращает существующий.

        Два воркера с дубликатом сообщения оба доходят до вставки, и второй
        обязан увидеть первый, а не упасть.
        """
        statement = (
            pg_insert(ProcessingJobRow)
            .values(**job_to_values(job))
            .on_conflict_do_nothing(constraint=JOB_CONSTRAINT)
            .returning(ProcessingJobRow.id)
        )
        if (await self._execute(statement)).scalar_one_or_none() is not None:
            return job

        existing = await self.get(job.document_id, job.pipeline_version)
        if existing is None:  # pragma: no cover — строка исчезла между запросами
            raise RuntimeError("прогон не вставлен и не найден")
        return existing

    async def record_progress(self, job_id: JobId, progress: JobProgressDTO) -> None:
        """Записывает прогресс и heartbeat."""
        statement = (
            update(ProcessingJobRow)
            .where(ProcessingJobRow.id == job_id.value)
            .values(
                pages_text_layer=progress.pages_text_layer,
                pages_ocr=progress.pages_ocr,
                pages_hybrid=progress.pages_hybrid,
                pages_failed=progress.pages_failed,
                chunks_created=progress.chunks_created,
                heartbeat_at=progress.heartbeat_at,
                updated_at=progress.heartbeat_at,
            )
        )
        await self._execute(statement)

    async def finish(self, job: ProcessingJob, *, expected: JobStatus) -> bool:
        """Фиксирует терминальный статус прогона под guard'ом."""
        statement = (
            update(ProcessingJobRow)
            .where(
                ProcessingJobRow.id == job.id.value,
                ProcessingJobRow.status == expected.value,
            )
            .values(
                status=job.status.value,
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
                updated_at=job.finished_at,
            )
            .returning(ProcessingJobRow.id)
        )
        result = await self._execute(statement)
        return result.scalar_one_or_none() is not None

    async def _row_of(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> ProcessingJobRow | None:
        statement = select(ProcessingJobRow).where(
            ProcessingJobRow.document_id == document_id.value,
            ProcessingJobRow.pipeline_version == str(pipeline_version),
        )
        return (await self._execute(statement)).scalar_one_or_none()
