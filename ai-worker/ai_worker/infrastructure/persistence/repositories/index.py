"""Состояние индексации документа и эмбеддинги его чанков."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ai_worker.domain.value_objects.hashing import ContentHash
from ai_worker.domain.value_objects.identifiers import ChunkId
from ai_worker.infrastructure.persistence.mappers.embedding import embedding_to_row
from ai_worker.infrastructure.persistence.mappers.index import (
    index_to_row,
    row_to_index,
)
from ai_worker.infrastructure.persistence.models.index import (
    ChunkEmbeddingRow,
    DocumentIndexRow,
)
from ai_worker.infrastructure.persistence.repositories.base import (
    SqlAlchemyRepository,
    values_of,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from ai_worker.domain.entities.chunk_embedding import ChunkEmbedding
    from ai_worker.domain.entities.document_index import DocumentIndex
    from ai_worker.domain.value_objects.enums import IndexStatus
    from ai_worker.domain.value_objects.identifiers import DocumentId
    from ai_worker.domain.value_objects.versioning import EmbeddingVersion

EMBEDDING_CONSTRAINT = "uq__ai_chunk_embeddings__chunk__ver"


class SqlAlchemyDocumentIndexRepository(SqlAlchemyRepository):
    """Прогон индексации: заведение, чтение и терминальный переход."""

    async def add(self, index: DocumentIndex) -> None:
        """Заводит прогон индексации."""
        await self._execute(
            pg_insert(DocumentIndexRow).values(values_of(index_to_row(index)))
        )

    async def get(
        self,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> DocumentIndex | None:
        """Читает прогон по документу и версии эмбеддингов."""
        statement = select(DocumentIndexRow).where(
            DocumentIndexRow.document_id == document_id.value,
            DocumentIndexRow.embedding_version == str(embedding_version),
        )
        row = (await self._execute(statement)).mappings().one_or_none()
        return None if row is None else row_to_index(DocumentIndexRow(**row))

    async def start(self, index: DocumentIndex, *, at: datetime) -> DocumentIndex:
        """Переводит прогон в работу и сохраняет переход."""
        running = index.start(at=at)
        await self._execute(
            update(DocumentIndexRow)
            .where(DocumentIndexRow.id == index.id.value)
            .values(status=running.status.value, started_at=at, updated_at=at)
        )
        return running

    async def finish(self, index: DocumentIndex, *, expected: IndexStatus) -> bool:
        """Записывает терминальное состояние под guard'ом по статусу.

        Ноль изменённых строк означает «кто-то уже завершил», а не ошибку:
        именно эта ветка не даёт пометить готовый документ отказом.
        """
        statement = (
            update(DocumentIndexRow)
            .where(
                DocumentIndexRow.id == index.id.value,
                DocumentIndexRow.status == expected.value,
            )
            .values(
                status=index.status.value,
                chunks_total=index.chunks_total,
                chunks_embedded=index.chunks_embedded,
                chunks_failed=index.chunks_failed,
                failure_code=index.failure_code,
                failure_message=index.failure_message,
                finished_at=index.finished_at,
                updated_at=index.finished_at,
            )
            .returning(DocumentIndexRow.id)
        )
        return (await self._execute(statement)).one_or_none() is not None


class SqlAlchemyEmbeddingRepository(SqlAlchemyRepository):
    """Эмбеддинги чанков: пакетная запись и сверка уже построенного."""

    async def add_many(self, embeddings: Sequence[ChunkEmbedding]) -> int:
        """Пишет пачку; повтор гасится уникальным ключом.

        Возвращается число реально вставленных строк, а не размер пачки: на
        повторной доставке они не совпадают, и разница — это и есть ответ на
        вопрос «сколько работы сделано сейчас».
        """
        if not embeddings:
            return 0
        statement = (
            pg_insert(ChunkEmbeddingRow)
            .values([values_of(embedding_to_row(item)) for item in embeddings])
            .on_conflict_do_nothing(constraint=EMBEDDING_CONSTRAINT)
            .returning(ChunkEmbeddingRow.id)
        )
        return len((await self._execute(statement)).scalars().all())

    async def stored_hashes(
        self,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> Mapping[ChunkId, ContentHash]:
        """Контрольные суммы уже построенных эмбеддингов документа.

        По ним пропускается повторный прогон модели: устав прямо запрещает
        пересчитывать эмбеддинг чанка, текст которого не изменился.
        """
        statement = select(
            ChunkEmbeddingRow.chunk_id, ChunkEmbeddingRow.content_hash
        ).where(
            ChunkEmbeddingRow.document_id == document_id.value,
            ChunkEmbeddingRow.embedding_version == str(embedding_version),
        )
        return {
            ChunkId(chunk_id): ContentHash(content_hash)
            for chunk_id, content_hash in await self._execute(statement)
        }

    async def count(
        self,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> int:
        """Сколько эмбеддингов у документа в этой версии."""
        statement = select(func.count()).where(
            ChunkEmbeddingRow.document_id == document_id.value,
            ChunkEmbeddingRow.embedding_version == str(embedding_version),
        )
        return int((await self._execute(statement)).scalar_one())
