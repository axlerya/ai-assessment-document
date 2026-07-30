"""Строка таблицы documents."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from document_worker.infrastructure.persistence.base import Base


class DocumentRow(Base):
    """Документ: исходный файл, статус обработки и её итог."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "bucket", "object_key", name="uq__documents__bucket__object_key"
        ),
        CheckConstraint(
            "status IN ('pending','processing','processed','partially_processed',"
            "'failed')",
            name="status",
        ),
        CheckConstraint("btrim(object_key) <> ''", name="object_key_not_blank"),
        CheckConstraint("declared_size_bytes > 0", name="declared_size_positive"),
        CheckConstraint("size_bytes IS NULL OR size_bytes > 0", name="size_positive"),
        CheckConstraint(
            "page_count IS NULL OR page_count > 0", name="page_count_positive"
        ),
        CheckConstraint("checksum_algorithm IN ('sha256')", name="checksum_algorithm"),
        CheckConstraint(
            "checksum IS NULL OR checksum ~ '^[0-9a-f]{64}$'", name="checksum_hex"
        ),
        CheckConstraint(
            "source_checksum IS NULL OR source_checksum ~ '^[0-9a-f]{64}$'",
            name="source_checksum_hex",
        ),
        CheckConstraint(
            "pipeline_version IS NULL"
            " OR pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'",
            name="pipeline_version_semver",
        ),
        CheckConstraint("version >= 0", name="version_non_negative"),
        CheckConstraint(
            "jsonb_typeof(source_metadata) = 'object'",
            name="source_metadata_is_object",
        ),
        CheckConstraint(
            "status NOT IN ('processed','partially_processed','failed')"
            " OR processing_finished_at IS NOT NULL",
            name="terminal_has_finished_at",
        ),
        CheckConstraint(
            "status <> 'failed'"
            " OR (failure_code IS NOT NULL AND failure_stage IS NOT NULL)",
            name="failed_has_failure",
        ),
        CheckConstraint(
            "status NOT IN ('processed','partially_processed')"
            " OR (page_count IS NOT NULL AND checksum IS NOT NULL"
            " AND size_bytes IS NOT NULL AND pipeline_version IS NOT NULL)",
            name="success_is_complete",
        ),
        CheckConstraint(
            "processing_finished_at IS NULL OR processing_started_at IS NULL"
            " OR processing_finished_at >= processing_started_at",
            name="finished_after_started",
        ),
        Index(
            "ix__documents__stale_processing",
            "processing_started_at",
            postgresql_where=text("status = 'processing'"),
        ),
        Index("ix__documents__correlation_id", "correlation_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    bucket: Mapped[str] = mapped_column(String(63))
    object_key: Mapped[str] = mapped_column(String(1024))
    original_filename: Mapped[str | None] = mapped_column(String(512))
    declared_mime_type: Mapped[str] = mapped_column(String(255))
    detected_mime_type: Mapped[str | None] = mapped_column(String(255))
    declared_size_bytes: Mapped[int] = mapped_column(BigInteger)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum_algorithm: Mapped[str] = mapped_column(
        String(16), server_default=text("'sha256'")
    )
    source_checksum: Mapped[str | None] = mapped_column(String(64))
    checksum: Mapped[str | None] = mapped_column(String(64))
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), server_default=text("'pending'"))
    pipeline_version: Mapped[str | None] = mapped_column(String(32))
    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        server_default=text("'{}'::jsonb")
    )
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_stage: Mapped[str | None] = mapped_column(String(32))
    failure_message: Mapped[str | None] = mapped_column(Text)
    correlation_id: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    created_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    processing_started_at: Mapped[dt.datetime | None]
    processing_finished_at: Mapped[dt.datetime | None]
