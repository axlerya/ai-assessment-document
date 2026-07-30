"""Строка таблицы document_chunks."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Mapped, mapped_column

from document_worker.infrastructure.persistence.base import Base


class DocumentChunkRow(Base):
    """Фрагмент текста страницы, сохраняющий связь с источником."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["page_id", "document_id", "page_number"],
            [
                "document_pages.id",
                "document_pages.document_id",
                "document_pages.page_number",
            ],
            ondelete="CASCADE",
            name="fk__document_chunks__page__document_pages",
        ),
        UniqueConstraint(
            "document_id",
            "chunking_version",
            "page_id",
            "start_offset",
            name="uq__document_chunks__page__start",
        ),
        UniqueConstraint(
            "document_id",
            "chunking_version",
            "chunk_index",
            name="uq__document_chunks__document__version__index",
        ),
        CheckConstraint("chunk_index >= 0", name="index"),
        CheckConstraint("page_number >= 1", name="page_number"),
        CheckConstraint("start_offset >= 0 AND end_offset > start_offset", name="span"),
        CheckConstraint(
            "char_length(text) = end_offset - start_offset",
            name="text_len_matches_span",
        ),
        CheckConstraint("btrim(text) <> ''", name="text_not_blank"),
        CheckConstraint(
            "extraction_method IN ('text_layer','ocr','hybrid')", name="method"
        ),
        CheckConstraint(
            "chunking_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'",
            name="chunking_version_semver",
        ),
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
        CheckConstraint(
            "overlap_prefix_chars >= 0"
            " AND overlap_prefix_chars < end_offset - start_offset",
            name="overlap",
        ),
        CheckConstraint("token_count >= 1", name="token_count"),
        CheckConstraint("illegible_span_count >= 0", name="illegible_count"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="content_hash"),
        CheckConstraint(
            "jsonb_typeof(heading_path) = 'array'", name="heading_path_is_array"
        ),
        Index("ix__document_chunks__page", "page_id", "document_id", "page_number"),
        Index(
            "ix__document_chunks__doc_version_page",
            "document_id",
            "chunking_version",
            "page_number",
            "chunk_index",
        ),
        Index(
            "ix__document_chunks__illegible",
            "document_id",
            "chunking_version",
            postgresql_where=sql_text("illegible_span_count > 0"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
            name="fk__document_chunks__document__documents",
        )
    )
    page_id: Mapped[uuid.UUID]
    page_number: Mapped[int] = mapped_column(Integer)
    chunking_version: Mapped[str] = mapped_column(String(32))
    chunk_index: Mapped[int] = mapped_column(Integer)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    overlap_prefix_chars: Mapped[int] = mapped_column(
        Integer, server_default=sql_text("0")
    )
    extraction_method: Mapped[str] = mapped_column(String(16))
    avg_ocr_confidence: Mapped[Decimal | None]
    illegible_span_count: Mapped[int] = mapped_column(
        Integer, server_default=sql_text("0")
    )
    heading_path: Mapped[list[str]] = mapped_column(
        server_default=sql_text("'[]'::jsonb")
    )
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(server_default=sql_text("now()"))
