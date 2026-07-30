"""Outbox: идемпотентная запись, лизинг выборки и перенос публикации."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from document_worker.application.dto.results import OutboxEventDTO
from document_worker.infrastructure.persistence.repositories.outbox import (
    SqlAlchemyOutboxRepository,
)
from tests.factories import NOW, new_correlation_id

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.dto.results import OutboxRecordDTO

pytestmark = pytest.mark.integration

LEASE_SECONDS = 30
EVENT_TYPE = "document.processed"


def _event(*, occurred_at: datetime = NOW) -> OutboxEventDTO:
    event_id = uuid.uuid4()
    aggregate_id = uuid.uuid4()
    return OutboxEventDTO(
        event_id=event_id,
        aggregate_id=aggregate_id,
        event_type=EVENT_TYPE,
        routing_key=EVENT_TYPE,
        payload={"event_id": str(event_id), "document_id": str(aggregate_id)},
        correlation_id=new_correlation_id(),
        occurred_at=occurred_at,
        headers={"x-attempt": "1"},
    )


async def test_enqueue_writes_the_events(session: AsyncSession) -> None:
    repository = SqlAlchemyOutboxRepository(session)

    written = await repository.enqueue([_event(), _event()])

    assert written == 2


async def test_enqueue_is_idempotent_by_event_id(session: AsyncSession) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    event = _event()
    await repository.enqueue([event])

    written = await repository.enqueue([event])

    assert written == 0


async def test_enqueue_of_empty_sequence_writes_nothing(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)

    assert await repository.enqueue([]) == 0


async def test_fetch_pending_returns_the_event_with_its_payload(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    event = _event()
    await repository.enqueue([event])

    records = await _fetch(repository)

    assert len(records) == 1
    assert records[0].event_id == event.event_id
    assert records[0].payload == event.payload
    assert records[0].headers == event.headers
    assert records[0].routing_key == EVENT_TYPE


async def test_fetch_pending_skips_events_with_live_lease(
    session: AsyncSession,
) -> None:
    # Без лиза в предикате второй relay забрал бы те же строки и опубликовал
    # каждое событие дважды.
    repository = SqlAlchemyOutboxRepository(session)
    await repository.enqueue([_event()])
    await _fetch(repository, owner="relay-1")

    again = await _fetch(repository, owner="relay-2")

    assert again == ()


async def test_lease_shifts_available_at(session: AsyncSession) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    await repository.enqueue([_event()])

    await _fetch(repository)

    assert await _available_at(session) == NOW + timedelta(seconds=LEASE_SECONDS)


async def test_fetch_pending_respects_available_at(session: AsyncSession) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    await repository.enqueue([_event()])
    await _fetch(repository)

    records = await _fetch(repository, now=NOW + timedelta(seconds=LEASE_SECONDS + 1))

    assert len(records) == 1


async def test_fetch_pending_honours_the_limit(session: AsyncSession) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    await repository.enqueue([_event(), _event(), _event()])

    records = await _fetch(repository, limit=2)

    assert len(records) == 2


async def test_fetch_pending_skips_published_events(session: AsyncSession) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    event = _event()
    await repository.enqueue([event])
    await _fetch(repository)
    await repository.mark_published([event.event_id], published_at=NOW)

    again = await _fetch(repository, now=NOW + timedelta(minutes=5))

    assert again == ()


async def test_mark_published_sets_timestamp_and_clears_lease(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    event = _event()
    await repository.enqueue([event])
    await _fetch(repository)

    await repository.mark_published([event.event_id], published_at=NOW)

    row = await _row(session)
    assert row["published_at"] is not None
    assert row["lease_owner"] is None


async def test_reschedule_moves_publication_and_counts_the_attempt(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    event = _event()
    await repository.enqueue([event])
    await _fetch(repository)
    later = NOW + timedelta(minutes=5)

    await repository.reschedule(
        event.event_id, error="брокер молчит", available_at=later
    )

    row = await _row(session)
    assert row["available_at"] == later
    assert row["attempts"] == 1
    assert row["last_error"] == "брокер молчит"
    assert row["lease_owner"] is None


async def test_rescheduled_event_is_fetched_again_after_its_time(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    event = _event()
    await repository.enqueue([event])
    await _fetch(repository)
    later = NOW + timedelta(minutes=5)
    await repository.reschedule(
        event.event_id, error="брокер молчит", available_at=later
    )

    records = await _fetch(repository, now=later)

    assert len(records) == 1


async def test_mark_published_of_nothing_touches_nothing(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    await repository.enqueue([_event()])

    await repository.mark_published([], published_at=NOW)

    assert len(await _fetch(repository)) == 1


async def test_oldest_pending_age_is_none_without_backlog(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)

    assert await repository.oldest_pending_age_s(now=NOW) is None


async def test_oldest_pending_age_counts_from_the_earliest_event(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyOutboxRepository(session)
    await repository.enqueue(
        [_event(occurred_at=NOW - timedelta(seconds=90)), _event(occurred_at=NOW)]
    )

    age = await repository.oldest_pending_age_s(now=NOW)

    assert age == pytest.approx(90.0)


async def _fetch(
    repository: SqlAlchemyOutboxRepository,
    *,
    limit: int = 10,
    now: datetime = NOW,
    owner: str = "relay-1",
) -> tuple[OutboxRecordDTO, ...]:
    return await repository.fetch_pending(
        limit=limit, now=now, lease_owner=owner, lease_seconds=LEASE_SECONDS
    )


async def _row(session: AsyncSession) -> dict[str, object]:
    result = await session.execute(
        text(
            "SELECT published_at, lease_owner, available_at, attempts, last_error"
            " FROM outbox_events"
        )
    )
    return dict(result.mappings().one())


async def _available_at(session: AsyncSession) -> datetime:
    result = await session.execute(text("SELECT available_at FROM outbox_events"))
    return result.scalar_one()  # type: ignore[no-any-return]
