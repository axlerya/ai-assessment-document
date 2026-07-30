"""Строка таблицы outbox_events."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from document_worker.infrastructure.persistence.base import Base


class OutboxEventRow(Base):
    """Исходящее событие, ожидающее публикации."""

    __tablename__ = "outbox_events"
    __table_args__ = (
        UniqueConstraint("event_id"),
        CheckConstraint(
            "event_type IN ('document.processed','document.partially_processed',"
            "'document.processing.failed')",
            name="event_type",
        ),
        CheckConstraint("btrim(routing_key) <> ''", name="routing_key_not_blank"),
        CheckConstraint(
            "jsonb_typeof(payload) = 'object' AND jsonb_typeof(headers) = 'object'",
            name="jsonb_shapes",
        ),
        CheckConstraint(
            "payload ->> 'event_id' = event_id::text", name="payload_has_event_id"
        ),
        CheckConstraint(
            "payload ->> 'document_id' = aggregate_id::text",
            name="payload_has_document_id",
        ),
        CheckConstraint("attempts >= 0", name="attempts"),
        CheckConstraint(
            "num_nonnulls(lease_owner, lease_expires_at) IN (0, 2)", name="lease_pair"
        ),
        CheckConstraint(
            "published_at IS NULL OR lease_owner IS NULL",
            name="published_has_no_lease",
        ),
        CheckConstraint(
            "published_at IS NULL OR published_at >= occurred_at",
            name="published_after_occurred",
        ),
        Index(
            "ix__outbox_events__unpublished",
            "available_at",
            "id",
            postgresql_where=text("published_at IS NULL"),
        ),
        Index(
            "ix__outbox_events__published_at",
            "published_at",
            postgresql_where=text("published_at IS NOT NULL"),
        ),
        Index("ix__outbox_events__aggregate", "aggregate_id", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    event_id: Mapped[uuid.UUID]
    aggregate_type: Mapped[str] = mapped_column(
        String(32), server_default=text("'document'")
    )
    aggregate_id: Mapped[uuid.UUID]
    event_type: Mapped[str] = mapped_column(String(64))
    routing_key: Mapped[str] = mapped_column(String(255))
    payload: Mapped[dict[str, Any]]
    headers: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    correlation_id: Mapped[str] = mapped_column(String(128))
    occurred_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    available_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[dt.datetime | None]
    published_at: Mapped[dt.datetime | None]
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
