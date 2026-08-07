"""Хранилище: что сценарию нужно уметь с ним делать.

Порты объявлены прикладным слоем, а реализованы инфраструктурой. Направление
именно такое: договор диктует тот, кому он нужен, иначе сценарий подстраивается
под удобство SQL, а не наоборот.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from ai_worker.application.dto.messaging import (
        ClaimResult,
        OutboxEventDTO,
        OutboxRecordDTO,
    )
    from ai_worker.application.dto.retrieval import RetrievalHitDTO, RetrievalRunDTO
    from ai_worker.domain.entities.chunk_embedding import ChunkEmbedding
    from ai_worker.domain.entities.document_index import DocumentIndex
    from ai_worker.domain.entities.draft import Draft
    from ai_worker.domain.value_objects.enums import IndexStatus
    from ai_worker.domain.value_objects.hashing import ContentHash
    from ai_worker.domain.value_objects.identifiers import (
        ChunkId,
        DocumentId,
        DraftId,
        EventId,
    )
    from ai_worker.domain.value_objects.versioning import EmbeddingVersion


@runtime_checkable
class DocumentIndexRepository(Protocol):
    """Прогон индексации документа."""

    async def add(self, index: DocumentIndex) -> None:
        """Заводит прогон."""
        ...

    async def get(
        self,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> DocumentIndex | None:
        """Читает прогон по документу и версии."""
        ...

    async def start(self, index: DocumentIndex, *, at: datetime) -> DocumentIndex:
        """Переводит прогон в работу."""
        ...

    async def finish(self, index: DocumentIndex, *, expected: IndexStatus) -> bool:
        """Записывает терминальное состояние под guard'ом по статусу.

        Ложь означает «кто-то уже завершил», а не ошибку: именно эта ветка не
        даёт пометить готовый документ отказом.
        """
        ...


@runtime_checkable
class EmbeddingRepository(Protocol):
    """Эмбеддинги чанков."""

    async def add_many(self, embeddings: Sequence[ChunkEmbedding]) -> int:
        """Пишет пачку и возвращает число реально вставленных строк."""
        ...

    async def stored_hashes(
        self,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> Mapping[ChunkId, ContentHash]:
        """Контрольные суммы уже построенных эмбеддингов документа."""
        ...

    async def count(
        self,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> int:
        """Сколько эмбеддингов у документа в этой версии."""
        ...


@runtime_checkable
class DraftRepository(Protocol):
    """Черновик со всем содержимым."""

    async def save(self, draft: Draft) -> None:
        """Пишет черновик, его утверждения и цитаты одной операцией."""
        ...

    async def get(self, draft_id: DraftId) -> Draft | None:
        """Читает черновик целиком."""
        ...


@runtime_checkable
class RetrievalHistoryRepository(Protocol):
    """История поиска."""

    async def record(
        self,
        run: RetrievalRunDTO,
        hits: Sequence[RetrievalHitDTO],
    ) -> None:
        """Сохраняет прогон вместе с его выдачей."""
        ...


@runtime_checkable
class ProcessedMessageRepository(Protocol):
    """Барьер идемпотентности доставки."""

    async def claim(  # noqa: PLR0913 — заявка описывается всеми этими значениями
        self,
        *,
        event_id: EventId,
        subject_id: uuid.UUID,
        message_type: str,
        lease_owner: str,
        lease_seconds: int,
        at: datetime,
    ) -> ClaimResult:
        """Занимает сообщение и сообщает, что с ним делать."""
        ...

    async def mark_completed(self, event_id: EventId, *, at: datetime) -> None:
        """Отмечает сообщение обработанным."""
        ...

    async def release(self, event_id: EventId, *, at: datetime) -> None:
        """Отпускает захват провалившейся попытки."""
        ...


@runtime_checkable
class OutboxRepository(Protocol):
    """Накопитель исходящих событий."""

    async def enqueue(self, events: Sequence[OutboxEventDTO]) -> int:
        """Кладёт события; повтор гасится уникальностью ключа."""
        ...

    async def fetch_pending(
        self,
        *,
        limit: int,
        now: datetime,
        lease_owner: str,
        lease_seconds: int,
    ) -> tuple[OutboxRecordDTO, ...]:
        """Берёт пачку в публикацию, захватывая строки лизом."""
        ...

    async def mark_published(
        self,
        event_ids: Sequence[uuid.UUID],
        *,
        published_at: datetime,
    ) -> None:
        """Отмечает события опубликованными."""
        ...

    async def reschedule(
        self,
        event_id: uuid.UUID,
        *,
        error: str,
        available_at: datetime,
    ) -> None:
        """Переносит публикацию на более поздний срок."""
        ...
