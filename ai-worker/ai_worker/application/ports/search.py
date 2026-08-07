"""Поиск по индексу."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_worker.domain.entities.source_chunk import ChunkQuality, ChunkRef
    from ai_worker.domain.value_objects.identifiers import DocumentId
    from ai_worker.domain.value_objects.scores import Score
    from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector
    from ai_worker.domain.value_objects.versioning import EmbeddingVersion


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Найденный фрагмент вместе с оценкой своей ветви."""

    ref: ChunkRef
    quality: ChunkQuality
    score: Score


@runtime_checkable
class VectorSearch(Protocol):
    """Две ветви поиска по индексу одного документа."""

    async def dense(
        self,
        vector: DenseVector,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
        limit: int,
    ) -> Sequence[SearchHit]:
        """Ищет по смыслу: ловит переформулировку."""
        ...

    async def sparse(
        self,
        vector: SparseVector,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
        limit: int,
    ) -> Sequence[SearchHit]:
        """Ищет по лексемам: ловит номер договора, дату и сумму дословно."""
        ...
