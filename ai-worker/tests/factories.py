"""Фабрики доменных объектов для тестов.

Значения по умолчанию правдоподобны, но не значимы: тест, которому важно
конкретное поле, задаёт его явно, и по вызову видно, что именно он проверяет.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_worker.domain.constants import DENSE_DIMENSIONS
from ai_worker.domain.entities.source_chunk import ChunkQuality, ChunkRef, SourceChunk
from ai_worker.domain.value_objects.enums import ExtractionMethod
from ai_worker.domain.value_objects.identifiers import ChunkId, DocumentId, PageId
from ai_worker.domain.value_objects.scores import Ratio
from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector
from ai_worker.domain.value_objects.versioning import ChunkingVersion

CHUNK_TEXT = (
    "Договор поставки № 12/АБ от 3 марта 2024 года заключён между ООО «Вектор» "
    "и АО «Полюс» на сумму 1 250 000 рублей."
)
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def make_ref(*, page_number: int = 3) -> ChunkRef:
    """Ссылка на чанк вместе с координатами цитирования."""
    return ChunkRef(
        chunk_id=ChunkId.generate(),
        document_id=DocumentId.generate(),
        page_id=PageId.generate(),
        page_number=page_number,
    )


def make_quality(
    *,
    method: ExtractionMethod = ExtractionMethod.TEXT_LAYER,
    confidence: float | None = None,
    illegible_span_count: int = 0,
) -> ChunkQuality:
    """Признаки надёжности чанка, скопированные из document-worker."""
    return ChunkQuality(
        extraction_method=method,
        avg_confidence=None if confidence is None else Ratio(confidence),
        illegible_span_count=illegible_span_count,
    )


def make_chunk(
    *,
    text: str = CHUNK_TEXT,
    ref: ChunkRef | None = None,
    quality: ChunkQuality | None = None,
    token_count: int = 42,
    chunking_version: ChunkingVersion | None = None,
    heading_path: tuple[str, ...] = ("Предмет договора",),
) -> SourceChunk:
    """Чанк документа в том виде, в каком его отдаёт document-worker."""
    return SourceChunk(
        ref=ref or make_ref(),
        quality=quality or make_quality(),
        text=text,
        token_count=token_count,
        chunking_version=chunking_version or ChunkingVersion(1, 0, 0),
        heading_path=heading_path,
    )


def make_dense(value: float = 0.01) -> DenseVector:
    """Плотный вектор нужной ширины."""
    return DenseVector(tuple(value for _ in range(DENSE_DIMENSIONS)))


def make_sparse(weights: dict[int, float] | None = None) -> SparseVector:
    """Разреженный вектор из нескольких весов."""
    return SparseVector.pruned(weights or {7: 0.9, 19: 0.4, 101: 0.15})
