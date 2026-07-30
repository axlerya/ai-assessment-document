"""Строка таблицы processed_messages — барьер идемпотентности доставки."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import CheckConstraint, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from document_worker.infrastructure.persistence.base import Base


class ProcessedMessageRow(Base):
    """Отметка о сообщении: занято или обработано.

    Внешнего ключа на documents нет намеренно: сообщение может обогнать коммит
    продюсера, и тогда claim было бы некуда записать.
    """

    __tablename__ = "processed_messages"
    __table_args__ = (
        CheckConstraint("status IN ('in_progress','completed')", name="status"),
        CheckConstraint(
            "pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'",
            name="pipeline_version_semver",
        ),
        CheckConstraint(
            "num_nonnulls(lease_owner, lease_expires_at) IN (0, 2)", name="lease_pair"
        ),
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
        CheckConstraint(
            "status <> 'completed' OR outcome IS NOT NULL",
            name="completed_has_outcome",
        ),
        CheckConstraint(
            "outcome IS NULL"
            " OR outcome IN ('processed','partially_processed','failed')",
            name="outcome",
        ),
        CheckConstraint("attempts >= 1", name="attempts"),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= first_seen_at",
            name="completed_after_seen",
        ),
        Index(
            "ix__processed_messages__stale",
            "lease_expires_at",
            postgresql_where=text("status = 'in_progress'"),
        ),
        Index("ix__processed_messages__document", "document_id", "pipeline_version"),
        Index(
            "ix__processed_messages__completed_at",
            "completed_at",
            postgresql_where=text("status = 'completed'"),
        ),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID]
    pipeline_version: Mapped[str] = mapped_column(String(32))
    message_type: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    outcome: Mapped[str | None] = mapped_column(String(24))
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[dt.datetime | None]
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    correlation_id: Mapped[str] = mapped_column(String(128))
    first_seen_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=text("now()"))
    completed_at: Mapped[dt.datetime | None]
