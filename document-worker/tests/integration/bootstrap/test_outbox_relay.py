"""Фоновая публикация накопленных событий на живых базе и брокере."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from faststream.rabbit import RabbitBroker

from document_worker.application.config import OutboxConfig
from document_worker.application.dto.results import OutboxEventDTO
from document_worker.application.errors import BrokerUnavailableError
from document_worker.application.use_cases.publish_outbox_events import (
    PublishOutboxEvents,
)
from document_worker.domain.value_objects.identifiers import CorrelationId
from document_worker.infrastructure.messaging.declare import declare_topology
from document_worker.infrastructure.messaging.outbox_publisher import (
    RabbitEventPublisher,
)
from document_worker.infrastructure.messaging.topology import (
    AUDIT_QUEUE,
    build_topology,
)
from tests.fakes.system import FixedClock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from document_worker.application.dto.results import OutboxRecordDTO
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from tests.conftest import Management

pytestmark = pytest.mark.integration

NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
CONSUMER_TIMEOUT_MS = 7_200_000
LEASE_OWNER = "relay-1"
EVENT_TYPE = "document.processed"


class BrokenPublisher:
    """Брокер, до которого не достучаться."""

    async def publish(self, record: OutboxRecordDTO) -> None:
        """Всегда отказывает."""
        del record
        raise BrokerUnavailableError("брокер недоступен")


@pytest.fixture
async def relay_broker(
    rabbitmq_url: str,
    isolated_vhost: str,
) -> AsyncIterator[RabbitBroker]:
    """Брокер с объявленной топологией и очередью аудита событий."""
    connected = RabbitBroker(f"{rabbitmq_url}{isolated_vhost}")
    await connected.connect()
    topology = build_topology(
        consumer_timeout_ms=CONSUMER_TIMEOUT_MS, declare_audit_queue=True
    )
    await declare_topology(connected, topology)
    try:
        yield connected
    finally:
        await connected.stop()


def _relay(
    uow_factory: UnitOfWorkFactory,
    publisher: object,
    *,
    batch_size: int = 10,
) -> PublishOutboxEvents:
    return PublishOutboxEvents(
        uow_factory=uow_factory,
        publisher=publisher,  # type: ignore[arg-type]
        clock=FixedClock(NOW),
        config=OutboxConfig(batch_size=batch_size),
        lease_owner=LEASE_OWNER,
    )


def _event(document_id: uuid.UUID) -> OutboxEventDTO:
    return OutboxEventDTO(
        event_id=uuid.uuid4(),
        aggregate_id=document_id,
        event_type=EVENT_TYPE,
        routing_key=EVENT_TYPE,
        payload={"document_id": str(document_id), "status": "processed"},
        correlation_id=CorrelationId(str(uuid.uuid4())),
        occurred_at=NOW,
    )


async def _enqueue(
    uow_factory: UnitOfWorkFactory,
    document_id: uuid.UUID,
    count: int = 1,
) -> None:
    async with uow_factory(statement_timeout_ms=5_000) as uow:
        await uow.outbox.enqueue([_event(document_id) for _ in range(count)])
        await uow.commit()


async def _pending_age(uow_factory: UnitOfWorkFactory) -> float | None:
    async with uow_factory(statement_timeout_ms=5_000, read_only=True) as uow:
        return await uow.outbox.oldest_pending_age_s(now=NOW)


async def test_relay_publishes_pending_events(
    uow_factory: UnitOfWorkFactory,
    relay_broker: RabbitBroker,
    management: Management,
    isolated_vhost: str,
) -> None:
    await _enqueue(uow_factory, uuid.uuid4(), count=2)
    topology = build_topology(consumer_timeout_ms=CONSUMER_TIMEOUT_MS)
    relay = _relay(
        uow_factory,
        RabbitEventPublisher(broker=relay_broker, exchange=topology.events),
    )

    result = await relay.execute()

    assert result.published == 2
    assert len(management.fetch(AUDIT_QUEUE, vhost=isolated_vhost)) == 2


async def test_published_events_are_not_sent_twice(
    uow_factory: UnitOfWorkFactory,
    relay_broker: RabbitBroker,
    management: Management,
    isolated_vhost: str,
) -> None:
    # Второй проход обязан увидеть пустую очередь: иначе каждое событие
    # уезжает потребителям столько раз, сколько тикнул таймер.
    await _enqueue(uow_factory, uuid.uuid4())
    topology = build_topology(consumer_timeout_ms=CONSUMER_TIMEOUT_MS)
    relay = _relay(
        uow_factory,
        RabbitEventPublisher(broker=relay_broker, exchange=topology.events),
    )
    await relay.execute()

    again = await relay.execute()

    assert again.fetched == 0
    assert len(management.fetch(AUDIT_QUEUE, vhost=isolated_vhost)) == 1


async def test_empty_outbox_costs_one_query(
    uow_factory: UnitOfWorkFactory,
    relay_broker: RabbitBroker,
) -> None:
    topology = build_topology(consumer_timeout_ms=CONSUMER_TIMEOUT_MS)
    relay = _relay(
        uow_factory,
        RabbitEventPublisher(broker=relay_broker, exchange=topology.events),
    )

    result = await relay.execute()

    assert result == type(result)(fetched=0, published=0, failed=0)


async def test_broker_outage_leaves_the_event_pending(
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Событие остаётся в очереди: отметить его опубликованным значит потерять
    # его навсегда.
    await _enqueue(uow_factory, uuid.uuid4())
    relay = _relay(uow_factory, BrokenPublisher())

    result = await relay.execute()

    assert result.failed == 1
    assert await _pending_age(uow_factory) is not None


async def test_event_is_published_after_the_broker_returns(
    uow_factory: UnitOfWorkFactory,
    relay_broker: RabbitBroker,
    management: Management,
    isolated_vhost: str,
) -> None:
    await _enqueue(uow_factory, uuid.uuid4())
    broken = _relay(uow_factory, BrokenPublisher())
    await broken.execute()
    topology = build_topology(consumer_timeout_ms=CONSUMER_TIMEOUT_MS)
    healthy = PublishOutboxEvents(
        uow_factory=uow_factory,
        publisher=RabbitEventPublisher(broker=relay_broker, exchange=topology.events),
        clock=FixedClock(NOW + timedelta(hours=1)),
        config=OutboxConfig(),
        lease_owner=LEASE_OWNER,
    )

    result = await healthy.execute()

    assert result.published == 1
    assert len(management.fetch(AUDIT_QUEUE, vhost=isolated_vhost)) == 1


async def test_outage_stops_the_batch_instead_of_timing_out_on_every_event(
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Каждая следующая публикация к лежащему брокеру — это ещё один таймаут,
    # и пачка из сотни событий превратилась бы в минуты ожидания.
    await _enqueue(uow_factory, uuid.uuid4(), count=5)
    relay = _relay(uow_factory, BrokenPublisher())

    result = await relay.execute()

    assert result.fetched == 5
    assert result.failed == 1
