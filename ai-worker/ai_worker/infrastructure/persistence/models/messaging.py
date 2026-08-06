"""Барьер идемпотентности доставки и накопитель исходящих событий.

Одна таблица inbox обслуживает оба потока: `subject_id` — это документ для
индексации и запрос для черновика. Разводить их по двум таблицам значило бы
дважды написать один и тот же захват с одними и теми же четырьмя исходами.
"""

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

from ai_worker.infrastructure.persistence.base import Base


class ProcessedMessageRow(Base):
    """Заявка на обработку сообщения и её исход."""

    __tablename__ = "ai_processed_messages"
    __table_args__ = (
        CheckConstraint("status IN ('in_progress','completed')", name="status"),
        CheckConstraint(
            "num_nonnulls(lease_owner, lease_expires_at) IN (0, 2)", name="lease_pair"
        ),
        # Занятая запись без владельца делает возобновление невозможным, а
        # отказ по конкуренции — вечным.
        CheckConstraint(
            "status <> 'in_progress' OR lease_owner IS NOT NULL",
            name="in_progress_has_lease",
        ),
        CheckConstraint(
            "status <> 'completed' OR lease_owner IS NULL",
            name="completed_has_no_lease",
        ),
        CheckConstraint(
            "status <> 'completed' OR completed_at IS NOT NULL",
            name="completed_has_timestamp",
        ),
        CheckConstraint("attempts >= 1", name="attempts"),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= first_seen_at",
            name="completed_after_seen",
        ),
        Index(
            "ix__ai_processed_messages__stale",
            "lease_expires_at",
            postgresql_where=text("status = 'in_progress'"),
        ),
        Index("ix__ai_processed_messages__subject", "subject_id", "message_type"),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    subject_id: Mapped[uuid.UUID] = mapped_column()
    message_type: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column()
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    correlation_id: Mapped[uuid.UUID | None] = mapped_column()
    first_seen_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    completed_at: Mapped[dt.datetime | None] = mapped_column()


class OutboxEventRow(Base):
    """Исходящее событие, ожидающее публикации."""

    __tablename__ = "ai_outbox_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq__ai_outbox_events__event_id"),
        CheckConstraint(
            "event_type IN ('document.indexed','draft.generated','draft.failed')",
            name="event_type",
        ),
        CheckConstraint("btrim(routing_key) <> ''", name="routing_key_not_blank"),
        CheckConstraint(
            "jsonb_typeof(payload) = 'object' AND jsonb_typeof(headers) = 'object'",
            name="jsonb_shapes",
        ),
        # Оператор `?` обязателен: сравнение с отсутствующим ключом даёт NULL,
        # а CHECK на NULL проходит — именно так пустой payload проезжает мимо
        # ограничения у соседнего сервиса.
        CheckConstraint(
            "payload ? 'event_id' AND payload ->> 'event_id' = event_id::text",
            name="payload_has_event_id",
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
            "ix__ai_outbox_events__unpublished",
            "available_at",
            "id",
            postgresql_where=text("published_at IS NULL"),
        ),
        Index(
            "ix__ai_outbox_events__published_at",
            "published_at",
            postgresql_where=text("published_at IS NOT NULL"),
        ),
        Index("ix__ai_outbox_events__aggregate", "aggregate_id", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    event_id: Mapped[uuid.UUID] = mapped_column()
    aggregate_id: Mapped[uuid.UUID] = mapped_column()
    event_type: Mapped[str] = mapped_column(String(64))
    routing_key: Mapped[str] = mapped_column(String(255))
    payload: Mapped[dict[str, Any]] = mapped_column()
    headers: Mapped[dict[str, Any]] = mapped_column(
        server_default=text("'{}'::jsonb"), default=dict
    )
    correlation_id: Mapped[str] = mapped_column(String(128))
    occurred_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    available_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column()
    published_at: Mapped[dt.datetime | None] = mapped_column()
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
