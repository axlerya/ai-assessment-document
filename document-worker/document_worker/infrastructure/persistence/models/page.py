"""Строки таблиц document_pages и document_illegible_spans."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from sqlalchemy.orm import Mapped, mapped_column

from document_worker.infrastructure.persistence.base import Base


class DocumentPageRow(Base):
    """Страница документа: её текст, способ извлечения и отказ."""

    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "pipeline_version",
            "page_number",
            name="uq__document_pages__document__version__number",
        ),
        UniqueConstraint(
            "id",
            "document_id",
            "page_number",
            name="uq__document_pages__id__document__number",
        ),
        CheckConstraint("page_number >= 1", name="page_number"),
        CheckConstraint(
            "status IN ('extracted','partially_illegible','illegible','failed')",
            name="status",
        ),
        CheckConstraint(
            "extraction_method IN ('text_layer','ocr','hybrid','none')", name="method"
        ),
        CheckConstraint(
            "pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'",
            name="pipeline_version_semver",
        ),
        CheckConstraint(
            "extraction_method <> 'text_layer' OR ocr_confidence IS NULL",
            name="no_confidence_for_text_layer",
        ),
        CheckConstraint(
            "extraction_method NOT IN ('ocr','hybrid') OR ocr_confidence IS NOT NULL",
            name="ocr_has_confidence",
        ),
        CheckConstraint(
            "extraction_method <> 'none' OR ocr_confidence IS NULL",
            name="none_method_has_no_confidence",
        ),
        CheckConstraint(
            "ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 1)",
            name="confidence_range",
        ),
        CheckConstraint("char_length(text) = text_length", name="text_length_matches"),
        CheckConstraint("illegible_span_count >= 0", name="span_count_non_negative"),
        CheckConstraint(
            "(status = 'extracted' AND illegible_span_count = 0)"
            " OR (status IN ('partially_illegible','illegible')"
            " AND illegible_span_count >= 1)"
            " OR (status = 'failed' AND illegible_span_count = 0)",
            name="status_matches_spans",
        ),
        CheckConstraint(
            "num_nonnulls(failure_reason, failure_message, failure_recoverable)"
            " IN (0, 3)",
            name="failure_columns_agree",
        ),
        CheckConstraint(
            "failure_reason IS NULL OR status = 'failed'",
            name="failure_only_when_failed",
        ),
        CheckConstraint(
            "status <> 'failed'"
            " OR (extraction_method = 'none' AND text = ''"
            " AND failure_reason IS NOT NULL)",
            name="failed_page_is_empty",
        ),
        CheckConstraint(
            "failure_reason IS NULL OR failure_reason IN ('render_failed','ocr_failed',"
            "'text_extraction_failed','page_corrupted','timeout')",
            name="failure_reason",
        ),
        CheckConstraint(
            "num_nonnulls(image_bucket, image_key) IN (0, 2)",
            name="image_ref_complete",
        ),
        # Ссылки на рендер не требуем: он воспроизводится из исходного PDF
        # детерминированно, а разрешение — единственное, чего для этого
        # не хватает.
        CheckConstraint(
            "extraction_method NOT IN ('ocr','hybrid') OR render_dpi IS NOT NULL",
            name="ocr_has_render_dpi",
        ),
        CheckConstraint(
            "render_dpi IS NULL OR render_dpi BETWEEN 72 AND 600",
            name="render_dpi_range",
        ),
        CheckConstraint("jsonb_typeof(warnings) = 'array'", name="warnings_is_array"),
        Index(
            "ix__document_pages__resume",
            "document_id",
            "pipeline_version",
            postgresql_include=["page_number", "status"],
        ),
        Index(
            "ix__document_pages__illegible",
            "document_id",
            "pipeline_version",
            postgresql_where=sql_text("illegible_span_count > 0"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
            name="fk__document_pages__document__documents",
        )
    )
    pipeline_version: Mapped[str] = mapped_column(String(32))
    page_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24))
    extraction_method: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text, server_default=sql_text("''"))
    text_length: Mapped[int] = mapped_column(Integer, server_default=sql_text("0"))
    ocr_confidence: Mapped[Decimal | None]
    illegible_span_count: Mapped[int] = mapped_column(
        Integer, server_default=sql_text("0")
    )
    image_bucket: Mapped[str | None] = mapped_column(String(63))
    image_key: Mapped[str | None] = mapped_column(String(1024))
    render_dpi: Mapped[int | None] = mapped_column(SmallInteger)
    warnings: Mapped[list[str]] = mapped_column(server_default=sql_text("'[]'::jsonb"))
    failure_reason: Mapped[str | None] = mapped_column(String(32))
    failure_message: Mapped[str | None] = mapped_column(Text)
    failure_recoverable: Mapped[bool | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=sql_text("now()"))


class IllegibleSpanRow(Base):
    """Неразборчивый фрагмент страницы."""

    __tablename__ = "document_illegible_spans"
    __table_args__ = (
        UniqueConstraint(
            "page_id", "span_index", name="uq__illegible_spans__page__index"
        ),
        UniqueConstraint(
            "page_id", "start_offset", name="uq__illegible_spans__page__start"
        ),
        CheckConstraint("span_index >= 0", name="ck__illegible_spans__span_index"),
        CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck__illegible_spans__span_bounds",
        ),
        CheckConstraint(
            "reason IN ('low_ocr_confidence','no_text_recognized','image_too_noisy',"
            "'handwriting','glyph_mapping_failed')",
            name="ck__illegible_spans__reason",
        ),
        CheckConstraint(
            "end_offset > start_offset OR reason = 'no_text_recognized'",
            name="ck__illegible_spans__zero_length_only_for_no_text",
        ),
        CheckConstraint(
            "reason <> 'no_text_recognized' OR raw_text = ''",
            name="ck__illegible_spans__raw_text_empty_for_no_text",
        ),
        CheckConstraint(
            "end_offset = start_offset"
            " OR char_length(raw_text) = end_offset - start_offset",
            name="ck__illegible_spans__raw_text_length_matches",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck__illegible_spans__confidence_range",
        ),
        CheckConstraint(
            "line_number IS NULL OR line_number >= 1",
            name="ck__illegible_spans__line_number",
        ),
        CheckConstraint(
            "num_nonnulls(bbox_x0, bbox_y0, bbox_x1, bbox_y1) IN (0, 4)",
            name="ck__illegible_spans__bbox_all",
        ),
        CheckConstraint(
            "bbox_x0 IS NULL"
            " OR (bbox_x0 >= 0 AND bbox_y0 >= 0 AND bbox_x1 <= 1 AND bbox_y1 <= 1"
            " AND bbox_x1 >= bbox_x0 AND bbox_y1 >= bbox_y0)",
            name="ck__illegible_spans__bbox_normalized",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "document_pages.id",
            ondelete="CASCADE",
            name="fk__illegible_spans__page__document_pages",
        )
    )
    span_index: Mapped[int] = mapped_column(Integer)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[Decimal]
    raw_text: Mapped[str] = mapped_column(Text)
    line_number: Mapped[int | None] = mapped_column(Integer)
    bbox_x0: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    bbox_y0: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    bbox_x1: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    bbox_y1: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=sql_text("now()"))
