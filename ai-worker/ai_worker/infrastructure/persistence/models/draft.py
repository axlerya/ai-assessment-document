"""Строки черновика, его утверждений и их цитат."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any

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
)

# Под именем `text` в `ai_draft_claims` живёт колонка, и внутри тела класса
# она перекрывает функцию SQLAlchemy. Псевдоним снимает столкновение, не
# трогая имя колонки: оно часть схемы.
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Mapped, mapped_column

from ai_worker.infrastructure.persistence.base import Base
from ai_worker.infrastructure.persistence.models.index import SEMVER


class DraftRow(Base):
    """Черновик вместе со своим происхождением и счётчиками."""

    __tablename__ = "ai_drafts"
    __table_args__ = (
        # По паре, а не по одному запросу: повтор доставки той же версии
        # промпта гасится, а новая версия даёт второй черновик для сравнения.
        UniqueConstraint(
            "request_id", "prompt_version", name="uq__ai_drafts__request__prompt"
        ),
        CheckConstraint("draft_type IN ('case_fact_summary')", name="type"),
        CheckConstraint(
            "status IN ('generated','insufficient_evidence','failed')", name="status"
        ),
        CheckConstraint("btrim(query) <> ''", name="query_not_blank"),
        CheckConstraint(
            "claims_total >= 0 AND claims_grounded >= 0 AND claims_unsupported >= 0"
            " AND evidence_total >= 0"
            " AND claims_grounded + claims_unsupported = claims_total",
            name="counters",
        ),
        # Опубликованный черновик состоит только из подтверждённых утверждений.
        CheckConstraint(
            "status <> 'generated' OR (body IS NOT NULL AND btrim(body) <> ''"
            " AND claims_grounded > 0)",
            name="generated_is_grounded",
        ),
        # Недостаток данных — результат, а не отказ: тело объясняет, чего нет.
        CheckConstraint(
            "status <> 'insufficient_evidence'"
            " OR (body IS NOT NULL AND claims_grounded = 0)",
            name="insufficient_has_body",
        ),
        CheckConstraint(
            "status <> 'failed' OR failure_code IS NOT NULL", name="failed_has_code"
        ),
        CheckConstraint(
            "groundedness IS NULL OR (groundedness >= 0 AND groundedness <= 1)",
            name="groundedness_range",
        ),
        CheckConstraint(
            "(input_tokens IS NULL OR input_tokens >= 0)"
            " AND (output_tokens IS NULL OR output_tokens >= 0)",
            name="tokens",
        ),
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="duration"),
        CheckConstraint(f"prompt_version ~ '{SEMVER}'", name="prompt_semver"),
        CheckConstraint(f"embedding_version ~ '{SEMVER}'", name="embedding_semver"),
        CheckConstraint(f"chunking_version ~ '{SEMVER}'", name="chunking_semver"),
        CheckConstraint(
            "jsonb_typeof(request_payload) = 'object'", name="payload_is_object"
        ),
        Index("ix__ai_drafts__document", "document_id", sql_text("created_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column()
    document_id: Mapped[uuid.UUID] = mapped_column()
    draft_type: Mapped[str] = mapped_column(String(32))
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24))
    body: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
    retrieval_profile: Mapped[str] = mapped_column(String(32))
    embedding_version: Mapped[str] = mapped_column(String(32))
    chunking_version: Mapped[str] = mapped_column(String(32))
    claims_total: Mapped[int] = mapped_column(Integer, server_default=sql_text("0"))
    claims_grounded: Mapped[int] = mapped_column(Integer, server_default=sql_text("0"))
    claims_unsupported: Mapped[int] = mapped_column(
        Integer, server_default=sql_text("0")
    )
    evidence_total: Mapped[int] = mapped_column(Integer, server_default=sql_text("0"))
    groundedness: Mapped[Decimal | None] = mapped_column()
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_message: Mapped[str | None] = mapped_column(Text)
    request_payload: Mapped[dict[str, Any]] = mapped_column(
        server_default=sql_text("'{}'::jsonb"), default=dict
    )
    correlation_id: Mapped[uuid.UUID | None] = mapped_column()
    created_at: Mapped[dt.datetime] = mapped_column(server_default=sql_text("now()"))
    finished_at: Mapped[dt.datetime | None] = mapped_column()


class DraftClaimRow(Base):
    """Утверждение черновика.

    Отклонённые хранятся вместе с подтверждёнными: без них не разобрать, что
    именно модель попыталась додумать. В тело черновика они не попадают.
    """

    __tablename__ = "ai_draft_claims"
    __table_args__ = (
        ForeignKeyConstraint(
            ["draft_id"],
            ["ai_drafts.id"],
            name="fk__ai_draft_claims__draft__ai_drafts",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "draft_id", "claim_index", name="uq__ai_draft_claims__draft__index"
        ),
        CheckConstraint("claim_index >= 0", name="index"),
        CheckConstraint("btrim(text) <> ''", name="text_not_blank"),
        CheckConstraint(
            "section IN ('parties','documents','dates','amounts','open_questions')",
            name="section",
        ),
        CheckConstraint(
            "(supported AND reject_code IS NULL)"
            " OR (NOT supported AND reject_code IS NOT NULL)",
            name="reject_only_when_unsupported",
        ),
        CheckConstraint(
            "reject_code IS NULL OR reject_code IN ('no_citation',"
            "'chunk_not_in_context','quote_not_found','unreliable_evidence_only')",
            name="reject_code",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    draft_id: Mapped[uuid.UUID] = mapped_column()
    claim_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(48))
    text: Mapped[str] = mapped_column(Text)
    supported: Mapped[bool] = mapped_column(Boolean)
    reject_code: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[dt.datetime] = mapped_column(server_default=sql_text("now()"))


class DraftCitationRow(Base):
    """Ссылка утверждения на точный фрагмент текста чанка."""

    __tablename__ = "ai_draft_citations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["claim_id"],
            ["ai_draft_claims.id"],
            name="fk__ai_draft_citations__claim__ai_draft_claims",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "claim_id",
            "chunk_id",
            "quote_start",
            name="uq__ai_draft_citations__claim__chunk__start",
        ),
        CheckConstraint("page_number >= 1", name="page_number"),
        CheckConstraint("quote_start >= 0 AND quote_end > quote_start", name="span"),
        # Цитата — точный срез текста чанка: расхождение длины означает ссылку
        # не на тот фрагмент, и проверить источник по ней уже нельзя.
        CheckConstraint(
            "char_length(quote) = quote_end - quote_start", name="quote_length_matches"
        ),
        CheckConstraint("btrim(quote) <> ''", name="quote_not_blank"),
        Index("ix__ai_draft_citations__chunk", "chunk_id"),
        Index("ix__ai_draft_citations__draft", "draft_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    claim_id: Mapped[uuid.UUID] = mapped_column()
    draft_id: Mapped[uuid.UUID] = mapped_column()
    chunk_id: Mapped[uuid.UUID] = mapped_column()
    page_id: Mapped[uuid.UUID] = mapped_column()
    page_number: Mapped[int] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)
    quote_start: Mapped[int] = mapped_column(Integer)
    quote_end: Mapped[int] = mapped_column(Integer)
    retrieval_score: Mapped[float] = mapped_column(REAL)
    rerank_score: Mapped[float] = mapped_column(REAL)
    reliable: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=sql_text("now()"))
