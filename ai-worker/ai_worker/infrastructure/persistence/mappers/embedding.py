"""Перевод эмбеддинга чанка в строку и обратно.

Мапперы написаны руками, а не через imperative mapping SQLAlchemy: тот
инструментирует доменный класс в рантайме — подменяет `__init__`, вешает
`InstrumentedAttribute`, привязывает объект к identity map, — что несовместимо
с `frozen=True, slots=True` у значений и нарушает независимость домена.
"""

from __future__ import annotations

from decimal import Decimal

from pgvector import SparseVector as PgSparseVector

from ai_worker.domain.entities.chunk_embedding import ChunkEmbedding
from ai_worker.domain.entities.source_chunk import ChunkQuality, ChunkRef
from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
from ai_worker.domain.value_objects.enums import ExtractionMethod
from ai_worker.domain.value_objects.hashing import ContentHash
from ai_worker.domain.value_objects.identifiers import (
    ChunkId,
    DocumentId,
    EmbeddingId,
    PageId,
)
from ai_worker.domain.value_objects.scores import Ratio
from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector
from ai_worker.domain.value_objects.versioning import ChunkingVersion, EmbeddingVersion
from ai_worker.infrastructure.persistence.base import SPARSE_DIMENSIONS
from ai_worker.infrastructure.persistence.models.index import ChunkEmbeddingRow

# Уверенность хранится в numeric(4,3): округление на входе делает round-trip
# точным, а не «почти точным» в третьем знаке.
_CONFIDENCE_QUANT = Decimal("0.001")


def embedding_to_row(embedding: ChunkEmbedding) -> ChunkEmbeddingRow:
    """Готовит эмбеддинг к записи."""
    return ChunkEmbeddingRow(
        id=embedding.id.value,
        chunk_id=embedding.ref.chunk_id.value,
        document_id=embedding.ref.document_id.value,
        page_id=embedding.ref.page_id.value,
        page_number=embedding.ref.page_number,
        chunking_version=str(embedding.chunking_version),
        embedding_version=str(embedding.embedding.version),
        model_name=embedding.embedding.model_name,
        content_hash=embedding.content_hash.value,
        dense=list(embedding.dense.values),
        sparse=_to_pgvector(embedding.sparse),
        token_count=embedding.token_count,
        extraction_method=embedding.quality.extraction_method.value,
        avg_ocr_confidence=_confidence_to_numeric(embedding.quality.avg_confidence),
        illegible_span_count=embedding.quality.illegible_span_count,
        heading_path=list(embedding.heading_path),
    )


def row_to_embedding(row: ChunkEmbeddingRow) -> ChunkEmbedding:
    """Восстанавливает эмбеддинг из строки."""
    return ChunkEmbedding(
        id=EmbeddingId(row.id),
        ref=ChunkRef(
            chunk_id=ChunkId(row.chunk_id),
            document_id=DocumentId(row.document_id),
            page_id=PageId(row.page_id),
            page_number=row.page_number,
        ),
        quality=ChunkQuality(
            extraction_method=ExtractionMethod(row.extraction_method),
            avg_confidence=_numeric_to_confidence(row.avg_ocr_confidence),
            illegible_span_count=row.illegible_span_count,
        ),
        chunking_version=ChunkingVersion.parse(row.chunking_version),
        embedding=EmbeddingIdentity(
            version=EmbeddingVersion.parse(row.embedding_version),
            model_name=row.model_name,
        ),
        content_hash=ContentHash(row.content_hash),
        dense=DenseVector(tuple(float(value) for value in row.dense)),
        sparse=_from_pgvector(row.sparse),
        token_count=row.token_count,
        heading_path=tuple(row.heading_path),
    )


def _confidence_to_numeric(confidence: Ratio | None) -> Decimal | None:
    if confidence is None:
        return None
    return Decimal(str(confidence.value)).quantize(_CONFIDENCE_QUANT)


def _numeric_to_confidence(value: Decimal | None) -> Ratio | None:
    return None if value is None else Ratio(float(value))


def _to_pgvector(sparse: SparseVector) -> PgSparseVector:
    """Переводит разреженный вектор в представление драйвера."""
    return PgSparseVector(dict(sparse.weights), SPARSE_DIMENSIONS)


def _from_pgvector(value: object) -> SparseVector:
    """Восстанавливает разреженный вектор из представления драйвера.

    Через ORM приходит объект драйвера, через сырой запрос — литерал строкой:
    оба пути реальны, и разбирать их приходится здесь.
    """
    stored = (
        value
        if isinstance(value, PgSparseVector)
        else PgSparseVector.from_text(str(value))
    )
    return SparseVector(
        tuple(
            (index, float(weight))
            for index, weight in zip(stored.indices(), stored.values(), strict=True)
        )
    )
