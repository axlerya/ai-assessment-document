"""Барьер идемпотентности доставки и накопитель исходящих событий."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ai_worker.application.dto.messaging import (
    ClaimOutcome,
    ClaimResult,
    OutboxRecordDTO,
)
from ai_worker.infrastructure.persistence.models.messaging import (
    OutboxEventRow,
    ProcessedMessageRow,
)
from ai_worker.infrastructure.persistence.repositories.base import (
    SqlAlchemyRepository,
    values_of,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence
    from datetime import datetime

    from ai_worker.application.dto.messaging import OutboxEventDTO
    from ai_worker.domain.value_objects.identifiers import EventId

IN_PROGRESS = "in_progress"
COMPLETED = "completed"
EVENT_CONSTRAINT = "uq__ai_outbox_events__event_id"


class SqlAlchemyProcessedMessageRepository(SqlAlchemyRepository):
    """Заявка на обработку сообщения и её четыре исхода."""

    async def claim(  # noqa: PLR0913 — заявка описывается всеми этими значениями
        self,
        *,
        event_id: EventId,
        subject_id: uuid.UUID,
        message_type: str,
        lease_owner: str,
        lease_seconds: int,
        at: datetime,
    ) -> ClaimResult:
        """Занимает сообщение и сообщает, что с ним делать.

        Гонку двух воркеров разрешает база, а не проверка в Python: строка
        читается под блокировкой, и второй увидит уже занятую.
        """
        until = at + timedelta(seconds=lease_seconds)
        existing = (
            (
                await self._execute(
                    select(ProcessedMessageRow)
                    .where(ProcessedMessageRow.event_id == event_id.value)
                    .with_for_update()
                )
            )
            .mappings()
            .one_or_none()
        )
        if existing is None:
            await self._execute(
                pg_insert(ProcessedMessageRow).values(
                    values_of(
                        ProcessedMessageRow(
                            event_id=event_id.value,
                            subject_id=subject_id,
                            message_type=message_type,
                            status=IN_PROGRESS,
                            lease_owner=lease_owner,
                            lease_expires_at=until,
                            first_seen_at=at,
                            updated_at=at,
                        )
                    )
                )
            )
            return ClaimResult(outcome=ClaimOutcome.PROCEED)
        if existing["status"] == COMPLETED:
            return ClaimResult(outcome=ClaimOutcome.SKIP, attempts=existing["attempts"])
        expires_at = existing["lease_expires_at"]
        if expires_at is not None and expires_at > at:
            # Работу уже кто-то делает: сообщение уходит на повтор, а попытка
            # расходуется — иначе два воркера крутились бы вокруг одного
            # документа до исчерпания лестницы.
            return ClaimResult(
                outcome=ClaimOutcome.REJECT_CONCURRENT, attempts=existing["attempts"]
            )
        attempts = existing["attempts"] + 1
        await self._execute(
            update(ProcessedMessageRow)
            .where(ProcessedMessageRow.event_id == event_id.value)
            .values(
                lease_owner=lease_owner,
                lease_expires_at=until,
                attempts=attempts,
                updated_at=at,
            )
        )
        return ClaimResult(outcome=ClaimOutcome.RESUME, attempts=attempts)

    async def mark_completed(self, event_id: EventId, *, at: datetime) -> None:
        """Отмечает сообщение обработанным и снимает захват."""
        await self._execute(
            update(ProcessedMessageRow)
            .where(ProcessedMessageRow.event_id == event_id.value)
            .values(
                status=COMPLETED,
                lease_owner=None,
                lease_expires_at=None,
                completed_at=at,
                updated_at=at,
            )
        )

    async def release(self, event_id: EventId, *, at: datetime) -> None:
        """Отпускает захват провалившейся попытки.

        Срок лиза приравнивается к отметке изменения: следующая доставка
        видит его протухшим и продолжает работу, не дожидаясь таймаута.
        """
        await self._execute(
            update(ProcessedMessageRow)
            .where(ProcessedMessageRow.event_id == event_id.value)
            .values(lease_expires_at=at, updated_at=at)
        )


class SqlAlchemyOutboxRepository(SqlAlchemyRepository):
    """Исходящие события: запись и выдача реле под лизом."""

    async def enqueue(self, events: Sequence[OutboxEventDTO]) -> int:
        """Кладёт события; повтор гасится уникальностью ключа."""
        if not events:
            return 0
        statement = (
            pg_insert(OutboxEventRow)
            .values([_values_of_event(event) for event in events])
            .on_conflict_do_nothing(constraint=EVENT_CONSTRAINT)
            .returning(OutboxEventRow.id)
        )
        return len((await self._execute(statement)).scalars().all())

    async def fetch_pending(
        self,
        *,
        limit: int,
        now: datetime,
        lease_owner: str,
        lease_seconds: int,
    ) -> tuple[OutboxRecordDTO, ...]:
        """Берёт пачку в публикацию, захватывая строки лизом.

        Срок доступности сдвигается вместе с лизом: без этого колонки лиза
        декоративны, и второй реле через полсекунды заберёт те же строки.
        """
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
            for row in await self._execute(statement)
        )

    async def mark_published(
        self,
        event_ids: Sequence[uuid.UUID],
        *,
        published_at: datetime,
    ) -> None:
        """Отмечает события опубликованными и снимает лиз."""
        if not event_ids:
            return
        await self._execute(
            update(OutboxEventRow)
            .where(OutboxEventRow.event_id.in_(event_ids))
            .values(
                published_at=published_at,
                lease_owner=None,
                lease_expires_at=None,
                last_error=None,
            )
        )

    async def reschedule(
        self,
        event_id: uuid.UUID,
        *,
        error: str,
        available_at: datetime,
    ) -> None:
        """Переносит публикацию события на более поздний срок."""
        await self._execute(
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


def _values_of_event(event: OutboxEventDTO) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "aggregate_id": event.aggregate_id,
        "event_type": event.event_type,
        "routing_key": event.routing_key,
        "payload": event.payload,
        "headers": event.headers,
        "correlation_id": event.correlation_id,
        "occurred_at": event.occurred_at,
        "available_at": event.occurred_at,
    }
