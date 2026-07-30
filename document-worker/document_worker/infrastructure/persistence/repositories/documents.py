"""Доступ к строке документа."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select, update

from document_worker.domain.value_objects.enums import DocumentStatus
from document_worker.infrastructure.persistence.mappers.document import (
    document_to_domain,
)
from document_worker.infrastructure.persistence.models.document import DocumentRow

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.domain.entities.document import Document
    from document_worker.domain.value_objects.identifiers import DocumentId
    from document_worker.domain.value_objects.versioning import PipelineVersion


class SqlAlchemyDocumentRepository:
    """Документ: чтение, захват строки и переходы под guard'ом по статусу."""

    def __init__(self, session: AsyncSession) -> None:
        """Работает в транзакции переданной сессии."""
        self._session = session

    async def get(self, document_id: DocumentId) -> Document | None:
        """Читает документ без блокировки."""
        row = await self._session.get(DocumentRow, document_id.value)
        return None if row is None else document_to_domain(row)

    async def acquire(self, document_id: DocumentId) -> Document | None:
        """Читает документ, блокируя строку до конца транзакции.

        Без блокировки второй воркер успел бы вклиниться между чтением статуса
        и его записью.
        """
        statement = (
            select(DocumentRow)
            .where(DocumentRow.id == document_id.value)
            .with_for_update()
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return None if row is None else document_to_domain(row)

    async def start_processing(
        self,
        document_id: DocumentId,
        *,
        pipeline_version: PipelineVersion,
        at: datetime,
    ) -> bool:
        """Переводит документ в обработку. False — он уже обрабатывается."""
        statement = (
            update(DocumentRow)
            .where(
                DocumentRow.id == document_id.value,
                DocumentRow.status != DocumentStatus.PROCESSING.value,
            )
            .values(
                status=DocumentStatus.PROCESSING.value,
                pipeline_version=str(pipeline_version),
                processing_started_at=at,
                processing_finished_at=None,
                failure_code=None,
                failure_stage=None,
                failure_message=None,
                updated_at=at,
                version=DocumentRow.version + 1,
            )
            .returning(DocumentRow.id)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None

    async def finish(self, document: Document, *, expected: DocumentStatus) -> bool:
        """Фиксирует терминальный результат под guard'ом по статусу.

        Ноль строк означает «кто-то уже завершил документ» и ошибкой не
        является: транзакция коммитится без изменений.
        """
        source = document.source
        statement = (
            update(DocumentRow)
            .where(
                DocumentRow.id == document.id.value,
                DocumentRow.status == expected.value,
            )
            .values(
                status=document.status.value,
                page_count=document.page_count,
                size_bytes=int(source.size),
                checksum=source.checksum.value if source.checksum else None,
                detected_mime_type=source.mime_type.value,
                pipeline_version=str(document.pipeline_version)
                if document.pipeline_version
                else None,
                failure_code=document.failure_code,
                failure_stage=document.failure_stage.value
                if document.failure_stage
                else None,
                failure_message=document.failure_message,
                processing_finished_at=document.processed_at,
                updated_at=document.updated_at,
                version=DocumentRow.version + 1,
            )
            .returning(DocumentRow.id)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None
