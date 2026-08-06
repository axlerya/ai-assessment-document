"""История поиска: прогон и его попадания.

Хранится не для отчётности. По рангам обеих ветвей видно, что нашло фрагмент —
переформулировка или точная лексема — и что переставил реранкер. На этих же
строках считаются Recall@K, MRR и nDCG на этапе оценки качества.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    REAL,
    Boolean,
    CheckConstraint,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ai_worker.infrastructure.persistence.base import Base
from ai_worker.infrastructure.persistence.models.index import SEMVER


class RetrievalRunRow(Base):
    """Один прогон поиска со счётчиками каждой ступени."""

    __tablename__ = "ai_retrieval_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["draft_id"],
            ["ai_drafts.id"],
            name="fk__ai_retrieval_runs__draft__ai_drafts",
            ondelete="SET NULL",
        ),
        CheckConstraint("btrim(query) <> ''", name="query_not_blank"),
        CheckConstraint("query_hash ~ '^[0-9a-f]{64}$'", name="query_hash"),
        CheckConstraint(f"embedding_version ~ '{SEMVER}'", name="embedding_semver"),
        # Отобрано не больше, чем переранжировано, а переранжировано не больше,
        # чем слито: обратное означает ошибку в подсчёте ступеней.
        CheckConstraint(
            "top_k >= 1 AND dense_candidates >= 0 AND sparse_candidates >= 0"
            " AND fused_candidates >= 0 AND reranked >= 0 AND selected >= 0"
            " AND selected <= reranked AND reranked <= fused_candidates",
            name="counters",
        ),
        CheckConstraint("context_tokens >= 0", name="context_tokens"),
        CheckConstraint("duration_ms >= 0", name="duration"),
        Index(
            "ix__ai_retrieval_runs__document",
            "document_id",
            text("created_at DESC"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    draft_id: Mapped[uuid.UUID | None] = mapped_column()
    document_id: Mapped[uuid.UUID] = mapped_column()
    query: Mapped[str] = mapped_column(Text)
    query_hash: Mapped[str] = mapped_column(String(64))
    embedding_version: Mapped[str] = mapped_column(String(32))
    retrieval_profile: Mapped[str] = mapped_column(String(32))
    top_k: Mapped[int] = mapped_column(Integer)
    dense_candidates: Mapped[int] = mapped_column(Integer)
    sparse_candidates: Mapped[int] = mapped_column(Integer)
    fused_candidates: Mapped[int] = mapped_column(Integer)
    reranked: Mapped[int] = mapped_column(Integer)
    selected: Mapped[int] = mapped_column(Integer)
    context_tokens: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))


class RetrievalHitRow(Base):
    """Одно попадание прогона со своими рангами и оценками."""

    __tablename__ = "ai_retrieval_hits"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id"],
            ["ai_retrieval_runs.id"],
            name="fk__ai_retrieval_hits__run__ai_retrieval_runs",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "run_id", "chunk_id", name="uq__ai_retrieval_hits__run__chunk"
        ),
        UniqueConstraint(
            "run_id", "final_rank", name="uq__ai_retrieval_hits__run__rank"
        ),
        CheckConstraint("page_number >= 1", name="page_number"),
        CheckConstraint("final_rank >= 1", name="final_rank"),
        CheckConstraint(
            "(dense_rank IS NULL OR dense_rank >= 1)"
            " AND (sparse_rank IS NULL OR sparse_rank >= 1)",
            name="ranks",
        ),
        # Попадание, не найденное ни одной ветвью, означает склейку по неверному
        # ключу — то есть чужой чанк в выдаче.
        CheckConstraint(
            "dense_rank IS NOT NULL OR sparse_rank IS NOT NULL", name="has_source"
        ),
        CheckConstraint(
            "num_nonnulls(dense_rank, dense_score) IN (0, 2)"
            " AND num_nonnulls(sparse_rank, sparse_score) IN (0, 2)",
            name="rank_score_pairs",
        ),
        CheckConstraint(
            "NOT selected OR rerank_score IS NOT NULL", name="selected_was_reranked"
        ),
        Index("ix__ai_retrieval_hits__chunk", "chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column()
    chunk_id: Mapped[uuid.UUID] = mapped_column()
    page_number: Mapped[int] = mapped_column(Integer)
    dense_rank: Mapped[int | None] = mapped_column(Integer)
    dense_score: Mapped[float | None] = mapped_column(REAL)
    sparse_rank: Mapped[int | None] = mapped_column(Integer)
    sparse_score: Mapped[float | None] = mapped_column(REAL)
    rrf_score: Mapped[float] = mapped_column(REAL)
    rerank_score: Mapped[float | None] = mapped_column(REAL)
    final_rank: Mapped[int] = mapped_column(Integer)
    selected: Mapped[bool] = mapped_column(Boolean)
