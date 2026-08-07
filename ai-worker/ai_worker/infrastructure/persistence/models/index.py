"""Строки состояния индексации и эмбеддингов чанков."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from pgvector import SparseVector
from pgvector.sqlalchemy import SPARSEVEC, Vector
from sqlalchemy import (
    CheckConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ai_worker.infrastructure.persistence.base import (
    DENSE_DIMENSIONS,
    SPARSE_DIMENSIONS,
    Base,
)

# Ведущие нули запрещены, мажорная часть начинается с единицы — те же
# правила, что в домене. Иначе "01.0.0" и "1.0.0" стали бы двумя записями
# одной версии, а значит двумя namespace вместо одного.
SEMVER = r"^[1-9][0-9]*\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"


class DocumentIndexRow(Base):
    """Прогон индексации документа в конкретной версии эмбеддингов."""

    __tablename__ = "ai_document_index"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "embedding_version", name="uq__ai_document_index__doc__ver"
        ),
        CheckConstraint(
            "status IN ('pending','indexing','indexed','failed')", name="status"
        ),
        CheckConstraint(
            "source_status IN ('processed','partially_processed')",
            name="source_status",
        ),
        CheckConstraint(f"embedding_version ~ '{SEMVER}'", name="embedding_semver"),
        CheckConstraint(f"chunking_version ~ '{SEMVER}'", name="chunking_semver"),
        CheckConstraint(f"pipeline_version ~ '{SEMVER}'", name="pipeline_semver"),
        CheckConstraint(
            "chunks_embedded >= 0 AND chunks_failed >= 0"
            " AND (chunks_total IS NULL OR chunks_embedded + chunks_failed"
            " <= chunks_total)",
            name="counters",
        ),
        # Документ, у которого не построен ни один эмбеддинг, готовым не
        # считается: поиск по нему вернул бы пустой контекст без единой ошибки.
        CheckConstraint(
            "status <> 'indexed' OR (chunks_total IS NOT NULL"
            " AND chunks_embedded + chunks_failed = chunks_total"
            " AND chunks_embedded > 0)",
            name="indexed_is_complete",
        ),
        CheckConstraint(
            "status <> 'failed' OR failure_code IS NOT NULL", name="failed_has_code"
        ),
        CheckConstraint(
            "status NOT IN ('indexed','failed') OR finished_at IS NOT NULL",
            name="terminal_has_finished",
        ),
        CheckConstraint(
            "status <> 'indexing' OR started_at IS NOT NULL", name="indexing_has_start"
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name="finished_after_started",
        ),
        Index(
            "ix__ai_document_index__stale",
            "started_at",
            postgresql_where=text("status = 'indexing'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column()
    embedding_version: Mapped[str] = mapped_column(String(32))
    chunking_version: Mapped[str] = mapped_column(String(32))
    pipeline_version: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), server_default=text("'pending'"))
    source_status: Mapped[str] = mapped_column(String(24))
    chunks_total: Mapped[int | None] = mapped_column(Integer)
    chunks_embedded: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    chunks_failed: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    source_event_id: Mapped[uuid.UUID] = mapped_column()
    correlation_id: Mapped[uuid.UUID | None] = mapped_column()
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[dt.datetime | None] = mapped_column()
    finished_at: Mapped[dt.datetime | None] = mapped_column()
    created_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))


class ChunkEmbeddingRow(Base):
    """Плотное и разреженное представления чанка.

    Внешнего ключа на `document_chunks` нет: он сделал бы схему неприменимой к
    пустой базе и непроверяемой без миграций соседнего сервиса. Целостность
    цитаты держит сверка с живым текстом чанка при подготовке черновика.
    """

    __tablename__ = "ai_chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "embedding_version",
            name="uq__ai_chunk_embeddings__chunk__ver",
        ),
        CheckConstraint("page_number >= 1", name="page_number"),
        CheckConstraint("token_count >= 1", name="token_count"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="content_hash"),
        CheckConstraint(
            "extraction_method IN ('text_layer','ocr','hybrid')", name="method"
        ),
        # Единица от текстового слоя и единица от распознавания — величины
        # разной природы: в одной колонке они портят любой агрегат качества.
        CheckConstraint(
            "extraction_method <> 'text_layer' OR avg_ocr_confidence IS NULL",
            name="no_confidence_for_text_layer",
        ),
        CheckConstraint(
            "extraction_method NOT IN ('ocr','hybrid')"
            " OR avg_ocr_confidence IS NOT NULL",
            name="ocr_has_confidence",
        ),
        CheckConstraint(
            "avg_ocr_confidence IS NULL"
            " OR (avg_ocr_confidence >= 0 AND avg_ocr_confidence <= 1)",
            name="confidence_range",
        ),
        CheckConstraint("illegible_span_count >= 0", name="illegible_count"),
        CheckConstraint(f"embedding_version ~ '{SEMVER}'", name="embedding_semver"),
        CheckConstraint(f"chunking_version ~ '{SEMVER}'", name="chunking_semver"),
        CheckConstraint("btrim(model_name) <> ''", name="model_name_not_blank"),
        CheckConstraint(
            "jsonb_typeof(heading_path) = 'array'", name="heading_path_is_array"
        ),
        Index(
            "ix__ai_chunk_embeddings__dense",
            "dense",
            postgresql_using="hnsw",
            postgresql_ops={"dense": "vector_cosine_ops"},
        ),
        Index(
            "ix__ai_chunk_embeddings__sparse",
            "sparse",
            postgresql_using="hnsw",
            postgresql_ops={"sparse": "sparsevec_ip_ops"},
        ),
        Index("ix__ai_chunk_embeddings__doc__ver", "document_id", "embedding_version"),
        # Обслуживает сверку осиротевших эмбеддингов, когда путь удаления
        # чанков появится: сейчас его нет ни у одного сервиса.
        Index("ix__ai_chunk_embeddings__chunk", "chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    chunk_id: Mapped[uuid.UUID] = mapped_column()
    document_id: Mapped[uuid.UUID] = mapped_column()
    page_id: Mapped[uuid.UUID] = mapped_column()
    page_number: Mapped[int] = mapped_column(Integer)
    chunking_version: Mapped[str] = mapped_column(String(32))
    embedding_version: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(64))
    dense: Mapped[list[float]] = mapped_column(Vector(DENSE_DIMENSIONS))
    sparse: Mapped[SparseVector] = mapped_column(SPARSEVEC(SPARSE_DIMENSIONS))
    token_count: Mapped[int] = mapped_column(Integer)
    extraction_method: Mapped[str] = mapped_column(String(16))
    avg_ocr_confidence: Mapped[Decimal | None] = mapped_column()
    illegible_span_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    heading_path: Mapped[list[str]] = mapped_column(
        server_default=text("'[]'::jsonb"), default=list
    )
    created_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
