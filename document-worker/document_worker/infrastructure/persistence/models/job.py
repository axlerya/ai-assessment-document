"""Строка таблицы processing_jobs."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from document_worker.infrastructure.persistence.base import Base


class ProcessingJobRow(Base):
    """Одна попытка обработать документ конкретной версией пайплайна."""

    __tablename__ = "processing_jobs"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "pipeline_version",
            name="uq__processing_jobs__document__version",
        ),
        CheckConstraint(
            "status IN ('queued','running','succeeded','failed')", name="status"
        ),
        CheckConstraint("attempt >= 1", name="attempt"),
        CheckConstraint(
            "pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'",
            name="pipeline_version_semver",
        ),
        CheckConstraint(
            "pages_text_layer >= 0 AND pages_ocr >= 0 AND pages_hybrid >= 0"
            " AND pages_failed >= 0 AND chunks_created >= 0",
            name="counters_non_negative",
        ),
        CheckConstraint("pages_total IS NULL OR pages_total >= 0", name="pages_total"),
        CheckConstraint(
            "pages_total IS NULL"
            " OR pages_text_layer + pages_ocr + pages_hybrid + pages_failed"
            " <= pages_total",
            name="counters_fit",
        ),
        CheckConstraint(
            "status <> 'succeeded'"
            " OR (pages_total IS NOT NULL"
            " AND pages_text_layer + pages_ocr + pages_hybrid + pages_failed"
            " = pages_total)",
            name="succeeded_counters_sum",
        ),
        CheckConstraint(
            "status <> 'running' OR started_at IS NOT NULL", name="running_has_start"
        ),
        CheckConstraint(
            "status NOT IN ('succeeded','failed') OR finished_at IS NOT NULL",
            name="terminal_has_finished",
        ),
        CheckConstraint(
            "status <> 'failed'"
            " OR (failure_code IS NOT NULL AND failure_stage IS NOT NULL)",
            name="failed_has_failure",
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name="finished_after_started",
        ),
        Index(
            "uq__processing_jobs__active",
            "document_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
        Index(
            "ix__processing_jobs__stale",
            "heartbeat_at",
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
            name="fk__processing_jobs__document__documents",
        )
    )
    pipeline_version: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), server_default=text("'queued'"))
    attempt: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    trigger_event_id: Mapped[uuid.UUID]
    correlation_id: Mapped[str] = mapped_column(String(128))
    pages_total: Mapped[int | None] = mapped_column(Integer)
    pages_text_layer: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    pages_ocr: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    pages_hybrid: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    pages_failed: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    chunks_created: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_stage: Mapped[str | None] = mapped_column(String(32))
    failure_message: Mapped[str | None] = mapped_column(Text)
    heartbeat_at: Mapped[dt.datetime | None]
    started_at: Mapped[dt.datetime | None]
    finished_at: Mapped[dt.datetime | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
