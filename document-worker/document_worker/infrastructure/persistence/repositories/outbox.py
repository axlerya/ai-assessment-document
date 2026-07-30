"""Накопитель исходящих событий.

Выборка захватывает строки лизом и сдвигает срок доступности одним запросом:
без сдвига `available_at` колонки лиза декоративны, и второй relay через
полсекунды заберёт те же строки и опубликует их повторно.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from document_worker.application.dto.results import OutboxRecordDTO
from document_worker.infrastructure.persistence.models.outbox import OutboxEventRow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.dto.results import OutboxEventDTO

EVENT_CONSTRAINT = "uq__outbox_events__event_id"


class SqlAlchemyOutboxRepository:
    """Outbox: запись событий и их выдача relay под лизом."""

    def __init__(self, session: AsyncSession) -> None:
        """Работает в транзакции переданной сессии."""
        self._session = session

    async def enqueue(self, events: Sequence[OutboxEventDTO]) -> int:
        """Кладёт события; повтор гасится уникальностью идентификатора."""
        if not events:
            return 0
        statement = (
            pg_insert(OutboxEventRow)
            .values([_values_of(event) for event in events])
            .on_conflict_do_nothing(constraint=EVENT_CONSTRAINT)
            .returning(OutboxEventRow.id)
        )
        return len((await self._session.execute(statement)).scalars().all())

    async def fetch_pending(
        self,
        *,
        limit: int,
        now: datetime,
        lease_owner: str,
        lease_seconds: int,
    ) -> tuple[OutboxRecordDTO, ...]:
        """Берёт пачку в публикацию, захватывая строки лизом."""
        claimed = (
            select(OutboxEventRow.id)
            .where(
                OutboxEventRow.published_at.is_(None),
                OutboxEventRow.available_at <= now,
                (OutboxEventRow.lease_expires_at.is_(None))
                | (OutboxEventRow.lease_expires_at <= now),
            )
            .order_by(OutboxEventRow.available_at, OutboxEventRow.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
            .cte("claimed")
        )
        until = now + timedelta(seconds=lease_seconds)
        statement = (
            update(OutboxEventRow)
            .where(OutboxEventRow.id.in_(select(claimed.c.id)))
            .values(lease_owner=lease_owner, lease_expires_at=until, available_at=until)
            .returning(
                OutboxEventRow.id,
                OutboxEventRow.event_id,
                OutboxEventRow.routing_key,
                OutboxEventRow.payload,
                OutboxEventRow.headers,
                OutboxEventRow.correlation_id,
                OutboxEventRow.occurred_at,
                OutboxEventRow.attempts,
            )
        )
        return tuple(
            OutboxRecordDTO(
                id=row.id,
                event_id=row.event_id,
                routing_key=row.routing_key,
                payload=row.payload,
                headers=row.headers,
                correlation_id=row.correlation_id,
                occurred_at=row.occurred_at,
                attempts=row.attempts,
            )
            for row in await self._session.execute(statement)
        )

    async def mark_published(
        self,
        event_ids: Sequence[UUID],
        *,
        published_at: datetime,
    ) -> None:
        """Отмечает события опубликованными и снимает лиз."""
        if not event_ids:
            return
        statement = (
            update(OutboxEventRow)
            .where(OutboxEventRow.event_id.in_(event_ids))
            .values(
                published_at=published_at,
                lease_owner=None,
                lease_expires_at=None,
                last_error=None,
            )
        )
        await self._session.execute(statement)

    async def reschedule(
        self,
        event_id: UUID,
        *,
        error: str,
        available_at: datetime,
    ) -> None:
        """Переносит публикацию события на более поздний срок."""
        statement = (
            update(OutboxEventRow)
            .where(OutboxEventRow.event_id == event_id)
            .values(
                available_at=available_at,
                attempts=OutboxEventRow.attempts + 1,
                last_error=error,
                lease_owner=None,
                lease_expires_at=None,
            )
        )
        await self._session.execute(statement)

    async def oldest_pending_age_s(self, *, now: datetime) -> float | None:
        """Возраст самого старого неопубликованного события."""
        statement = select(func.min(OutboxEventRow.occurred_at)).where(
            OutboxEventRow.published_at.is_(None)
        )
        oldest = (await self._session.execute(statement)).scalar_one_or_none()
        return None if oldest is None else (now - oldest).total_seconds()


def _values_of(event: OutboxEventDTO) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "aggregate_id": event.aggregate_id,
        "event_type": event.event_type,
        "routing_key": event.routing_key,
        "payload": event.payload,
        "headers": event.headers,
        "correlation_id": str(event.correlation_id),
        "occurred_at": event.occurred_at,
        "available_at": event.occurred_at,
    }
