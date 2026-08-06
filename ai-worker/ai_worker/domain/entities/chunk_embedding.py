"""Эмбеддинг чанка в конкретной версии.

Устав требует, чтобы каждый эмбеддинг нёс документ, чанк, номер страницы,
версию, имя модели и момент создания. Все шесть здесь есть, и это не
формальность: без имени модели и версии нельзя ни объяснить выдачу, ни решить,
нужна ли переиндексация.

Равенство — по ключу, а не по векторам. Два прогона одной модели дают
численно разные значения в последних разрядах, и сравнение по содержимому
означало бы, что повторная индексация каждый раз «находит» новые эмбеддинги.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

from ai_worker.domain.value_objects.identifiers import EmbeddingId

if TYPE_CHECKING:
    from ai_worker.domain.entities.source_chunk import (
        ChunkQuality,
        ChunkRef,
        SourceChunk,
    )
    from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
    from ai_worker.domain.value_objects.hashing import ContentHash
    from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector
    from ai_worker.domain.value_objects.versioning import ChunkingVersion


@dataclass(frozen=True, slots=True, eq=False)
class ChunkEmbedding:
    """Плотное и разреженное представления чанка вместе с их происхождением."""

    id: EmbeddingId
    ref: ChunkRef
    quality: ChunkQuality
    chunking_version: ChunkingVersion
    embedding: EmbeddingIdentity
    content_hash: ContentHash
    dense: DenseVector
    sparse: SparseVector
    token_count: int
    heading_path: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        *,
        chunk: SourceChunk,
        embedding: EmbeddingIdentity,
        dense: DenseVector,
        sparse: SparseVector,
    ) -> Self:
        """Строит эмбеддинг чанка с детерминированным ключом."""
        return cls(
            id=EmbeddingId.deterministic(
                chunk_id=chunk.ref.chunk_id, embedding_version=embedding.version
            ),
            ref=chunk.ref,
            quality=chunk.quality,
            chunking_version=chunk.chunking_version,
            embedding=embedding,
            content_hash=chunk.content_hash,
            dense=dense,
            sparse=sparse,
            token_count=chunk.token_count,
            heading_path=chunk.heading_path,
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChunkEmbedding):
            return NotImplemented
        return self.id == other.id

    @override
    def __hash__(self) -> int:
        return hash(self.id)
