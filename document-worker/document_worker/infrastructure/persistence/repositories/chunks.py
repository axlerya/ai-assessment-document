"""Доступ к чанкам документа."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from document_worker.application.errors import ChunkPersistenceMismatchError
from document_worker.infrastructure.persistence.mappers.chunk import chunk_to_values
from document_worker.infrastructure.persistence.models.chunk import DocumentChunkRow
from document_worker.infrastructure.persistence.repositories.base import (
    SqlAlchemyRepository,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.entities.document_chunk import DocumentChunk
    from document_worker.domain.value_objects.identifiers import DocumentId
    from document_worker.domain.value_objects.versioning import ChunkingVersion

CHUNK_CONSTRAINT = "uq__document_chunks__page__start"


class SqlAlchemyDocumentChunkRepository(SqlAlchemyRepository):
    """Чанки: весь документ одной вставкой, со сверкой числа записанных строк."""

    async def add_all(self, chunks: Sequence[DocumentChunk]) -> int:
        """Пишет все чанки документа одной вставкой и возвращает число строк.

        Сквозной номер присваивается перечислением последовательности: доменный
        ordinal плотный внутри страницы и на весь документ не годится.
        """
        if not chunks:
            return 0
        prepared = [
            chunk_to_values(chunk, chunk_index=index)
            for index, chunk in enumerate(chunks)
        ]
        statement = (
            pg_insert(DocumentChunkRow)
            .values(prepared)
            .on_conflict_do_nothing(constraint=CHUNK_CONSTRAINT)
            .returning(DocumentChunkRow.id)
        )
        inserted = (await self._execute(statement)).scalars().all()
        if len(inserted) != len(prepared):
            await self._require_skipped_rows_are_identical(prepared, set(inserted))
        return len(inserted)

    async def count(
        self,
        document_id: DocumentId,
        chunking_version: ChunkingVersion,
    ) -> int:
        """Сколько чанков сохранено."""
        statement = select(func.count()).where(
            DocumentChunkRow.document_id == document_id.value,
            DocumentChunkRow.chunking_version == str(chunking_version),
        )
        return int((await self._execute(statement)).scalar_one())

    async def _require_skipped_rows_are_identical(
        self,
        prepared: list[dict[str, object]],
        inserted: set[object],
    ) -> None:
        """Расхождение — не норма: молчаливая потеря чанков начинается здесь."""
        skipped = [values for values in prepared if values["id"] not in inserted]
        statement = select(
            DocumentChunkRow.page_id,
            DocumentChunkRow.start_offset,
            DocumentChunkRow.content_hash,
        ).where(
            DocumentChunkRow.document_id == skipped[0]["document_id"],
            DocumentChunkRow.chunking_version == skipped[0]["chunking_version"],
        )
        stored = {
            (row.page_id, row.start_offset): row.content_hash
            for row in await self._execute(statement)
        }
        for values in skipped:
            key = (values["page_id"], values["start_offset"])
            if stored.get(key) != values["content_hash"]:
                raise ChunkPersistenceMismatchError(
                    "вставлено не столько чанков, сколько подготовлено",
                    context={
                        "prepared": len(prepared),
                        "inserted": len(inserted),
                        "page_id": str(values["page_id"]),
                        "start_offset": values["start_offset"],
                    },
                )
