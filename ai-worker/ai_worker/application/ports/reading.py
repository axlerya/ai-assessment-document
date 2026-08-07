"""Чтение результатов document-worker.

Единственная точка, знающая про чужие таблицы. Всё остальное ходит сюда, и
граница между сервисами остаётся проверяемой: обход виден как второй адаптер,
а не как ещё один запрос где-то в глубине.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from ai_worker.domain.entities.source_chunk import SourceChunk
    from ai_worker.domain.value_objects.enums import SourceStatus
    from ai_worker.domain.value_objects.identifiers import ChunkId, DocumentId
    from ai_worker.domain.value_objects.versioning import (
        ChunkingVersion,
        PipelineVersion,
    )


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """Что известно о документе из его строки."""

    document_id: DocumentId
    status: SourceStatus
    pipeline_version: PipelineVersion
    page_count: int | None


@runtime_checkable
class ProcessedChunkReader(Protocol):
    """Отдаёт обработанные чанки документа, не меняя их."""

    async def document_summary(
        self,
        document_id: DocumentId,
    ) -> DocumentSummary | None:
        """Сводка документа или `None`, если строки ещё нет."""
        ...

    async def chunking_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[ChunkingVersion, ...]:
        """Версии, под которыми у документа есть чанки."""
        ...

    def chunks(
        self,
        document_id: DocumentId,
        *,
        chunking_version: ChunkingVersion,
    ) -> AsyncIterator[SourceChunk]:
        """Поток чанков документа в порядке страниц и позиций.

        Именно поток: документ на триста страниц даёт тысячи чанков, и
        загружать их разом в память незачем.
        """
        ...

    async def chunk_texts(
        self,
        chunk_ids: Sequence[ChunkId],
    ) -> Mapping[ChunkId, str]:
        """Тексты чанков для сверки цитат с живым источником."""
        ...
