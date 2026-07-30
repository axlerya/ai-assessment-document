"""Порты репозиториев.

Ни один метод не коммитит: все работают в транзакции текущего Unit of Work.
Репозитории принимают и возвращают доменные сущности, лёгкие проекции — DTO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from document_worker.application.dto.results import (
        ClaimOutcomeDTO,
        JobProgressDTO,
        MessageClaimDTO,
        MessageOutcome,
        OutboxEventDTO,
        OutboxRecordDTO,
        PageSummaryDTO,
    )
    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.document_chunk import DocumentChunk
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.entities.processing_job import ProcessingJob
    from document_worker.domain.value_objects.enums import (
        DocumentStatus,
        JobStatus,
        PageStatus,
    )
    from document_worker.domain.value_objects.identifiers import (
        DocumentId,
        EventId,
        JobId,
    )
    from document_worker.domain.value_objects.versioning import (
        ChunkingVersion,
        PipelineVersion,
    )


@runtime_checkable
class DocumentRepository(Protocol):
    """Доступ к строке документа."""

    async def get(self, document_id: DocumentId) -> Document | None:
        """Читает документ без блокировки."""
        ...

    async def acquire(self, document_id: DocumentId) -> Document | None:
        """Читает документ, блокируя строку до конца транзакции."""
        ...

    async def start_processing(
        self,
        document_id: DocumentId,
        *,
        pipeline_version: PipelineVersion,
        at: datetime,
    ) -> bool:
        """Переводит документ в обработку. False — статус уже не тот."""
        ...

    async def finish(self, document: Document, *, expected: DocumentStatus) -> bool:
        """Фиксирует терминальный результат под guard'ом по статусу.

        False означает «кто-то уже завершил» и ошибкой не является.
        """
        ...


@runtime_checkable
class DocumentPageRepository(Protocol):
    """Доступ к страницам документа."""

    async def add(self, page: DocumentPage) -> bool:
        """Пишет страницу вместе с её неразборчивыми диапазонами.

        False — строка уже была: повторная доставка гасится ограничением.
        """
        ...

    async def list_persisted_page_numbers(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> frozenset[int]:
        """Номера уже сохранённых страниц — вход для возобновления."""
        ...

    async def list_summaries(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> tuple[PageSummaryDTO, ...]:
        """Метрики страниц без их текста."""
        ...

    async def load_pages(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
        *,
        statuses: frozenset[PageStatus],
    ) -> tuple[DocumentPage, ...]:
        """Читает страницы указанных статусов целиком."""
        ...

    async def count(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> int:
        """Сколько страниц сохранено."""
        ...


@runtime_checkable
class DocumentChunkRepository(Protocol):
    """Доступ к чанкам документа."""

    async def add_all(self, chunks: Sequence[DocumentChunk]) -> int:
        """Пишет все чанки документа одной вставкой и возвращает число строк."""
        ...

    async def count(
        self,
        document_id: DocumentId,
        chunking_version: ChunkingVersion,
    ) -> int:
        """Сколько чанков сохранено."""
        ...


@runtime_checkable
class ProcessingJobRepository(Protocol):
    """Доступ к прогонам обработки."""

    async def get(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> ProcessingJob | None:
        """Читает прогон документа для указанной версии пайплайна."""
        ...

    async def start(self, job: ProcessingJob) -> ProcessingJob:
        """Заводит прогон; при конфликте возвращает существующий."""
        ...

    async def record_progress(self, job_id: JobId, progress: JobProgressDTO) -> None:
        """Записывает прогресс и heartbeat."""
        ...

    async def finish(self, job: ProcessingJob, *, expected: JobStatus) -> bool:
        """Фиксирует терминальный статус прогона под guard'ом."""
        ...


@runtime_checkable
class ProcessedMessageRepository(Protocol):
    """Барьер идемпотентности доставки."""

    async def try_claim(self, claim: MessageClaimDTO) -> ClaimOutcomeDTO:
        """Пытается занять сообщение и сообщает, что делать дальше."""
        ...

    async def mark_completed(
        self,
        event_id: EventId,
        *,
        outcome: MessageOutcome,
        completed_at: datetime,
    ) -> None:
        """Отмечает сообщение обработанным — только вместе с результатом."""
        ...

    async def release(self, event_id: EventId, *, reason: str, at: datetime) -> None:
        """Снимает лиз, оставляя запись незавершённой: следующая доставка продолжит."""
        ...


@runtime_checkable
class OutboxRepository(Protocol):
    """Накопитель исходящих событий."""

    async def enqueue(self, events: Sequence[OutboxEventDTO]) -> int:
        """Кладёт события; повтор гасится уникальностью идентификатора."""
        ...

    async def fetch_pending(
        self,
        *,
        limit: int,
        now: datetime,
        lease_owner: str,
        lease_seconds: int,
    ) -> tuple[OutboxRecordDTO, ...]:
        """Берёт пачку в публикацию.

        Предикат выборки обязан учитывать лиз и сдвигать срок доступности:
        иначе второй relay заберёт те же строки и опубликует их повторно.
        """
        ...

    async def mark_published(
        self,
        event_ids: Sequence[UUID],
        *,
        published_at: datetime,
    ) -> None:
        """Отмечает события опубликованными."""
        ...

    async def reschedule(
        self,
        event_id: UUID,
        *,
        error: str,
        available_at: datetime,
    ) -> None:
        """Переносит публикацию события на более поздний срок."""
        ...

    async def oldest_pending_age_s(self, *, now: datetime) -> float | None:
        """Возраст самого старого неопубликованного события."""
        ...
