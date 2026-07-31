"""Объявление топологии на живом RabbitMQ.

Аргументы очередей — это поведение брокера, а не наше: `x-expires` тихо
удаляет отложенные сообщения, `x-max-length` теряет команды, `reject-publish`
на DLQ останавливает конвейер. Проверить их можно только объявлением.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aio_pika.exceptions import ChannelPreconditionFailed

from document_worker.infrastructure.messaging.declare import declare_topology
from document_worker.infrastructure.messaging.topology import (
    COMMANDS_EXCHANGE,
    DLQ_QUEUE,
    PROCESS_REQUESTED_QUEUE,
    RETRY_LADDER,
    RK_PROCESS_REQUESTED,
    RK_RETRY_BASE,
    UNROUTED_QUEUE,
    build_topology,
    retry_level_for_attempt,
)

if TYPE_CHECKING:
    from faststream.rabbit import RabbitBroker

pytestmark = pytest.mark.integration

CONSUMER_TIMEOUT_MS = 7_200_000
DELIVERY_LIMIT = 20


async def _arguments_of(broker: RabbitBroker, queue: str) -> dict[str, object]:
    """Аргументы очереди так, как их видит брокер."""
    channel = await broker._connection.channel()
    declared = await channel.declare_queue(queue, passive=True)
    return dict(declared.arguments or {})


def _topology(**overrides: object) -> object:
    return build_topology(
        consumer_timeout_ms=CONSUMER_TIMEOUT_MS,
        delivery_limit=DELIVERY_LIMIT,
        **overrides,  # type: ignore[arg-type]
    )


async def test_declare_creates_every_exchange_and_queue(broker: RabbitBroker) -> None:
    topology = _topology()

    await declare_topology(broker, topology)

    for queue in (PROCESS_REQUESTED_QUEUE, DLQ_QUEUE, UNROUTED_QUEUE):
        assert await _arguments_of(broker, queue) is not None


async def test_main_queue_arguments_match_the_specification(
    broker: RabbitBroker,
) -> None:
    await declare_topology(broker, _topology())

    arguments = await _arguments_of(broker, PROCESS_REQUESTED_QUEUE)

    assert arguments["x-queue-type"] == "quorum"
    assert arguments["x-delivery-limit"] == DELIVERY_LIMIT
    assert arguments["x-dead-letter-strategy"] == "at-least-once"
    assert arguments["x-overflow"] == "reject-publish"
    assert arguments["x-consumer-timeout"] == CONSUMER_TIMEOUT_MS


async def test_main_queue_has_no_max_length(broker: RabbitBroker) -> None:
    # При reject-publish предел длины превратил бы лавину повторов в отказ
    # публикации, при drop-head — в тихую потерю команд.
    await declare_topology(broker, _topology())

    assert "x-max-length" not in await _arguments_of(broker, PROCESS_REQUESTED_QUEUE)


async def test_retry_queues_have_the_expected_ttl_ladder(
    broker: RabbitBroker,
) -> None:
    await declare_topology(broker, _topology())

    for level, ttl_ms in RETRY_LADDER:
        arguments = await _arguments_of(broker, f"{RK_RETRY_BASE}.{level}.q")
        assert arguments["x-message-ttl"] == ttl_ms
        assert arguments["x-dead-letter-exchange"] == COMMANDS_EXCHANGE
        assert arguments["x-dead-letter-routing-key"] == RK_PROCESS_REQUESTED


async def test_retry_queues_have_no_expires_argument(broker: RabbitBroker) -> None:
    # Очередь с истёкшим x-expires удаляется вместе с содержимым, и отложенные
    # сообщения при этом не дедлеттерятся — они просто исчезают.
    await declare_topology(broker, _topology())

    for level, _ in RETRY_LADDER:
        arguments = await _arguments_of(broker, f"{RK_RETRY_BASE}.{level}.q")
        assert "x-expires" not in arguments


async def test_dlq_drops_the_oldest_instead_of_rejecting(
    broker: RabbitBroker,
) -> None:
    # Переполненная DLQ с reject-publish перестала бы принимать dead-letter,
    # и переполнение разбора заблокировало бы обработку живых документов.
    await declare_topology(broker, _topology())

    arguments = await _arguments_of(broker, DLQ_QUEUE)

    assert arguments["x-overflow"] == "drop-head"
    assert arguments["x-queue-type"] == "quorum"
    assert arguments["x-max-length"] == 100_000


async def test_declare_is_idempotent(broker: RabbitBroker) -> None:
    topology = _topology()
    await declare_topology(broker, topology)

    await declare_topology(broker, topology)

    assert await _arguments_of(broker, PROCESS_REQUESTED_QUEUE)


async def test_declare_fails_on_argument_drift(broker: RabbitBroker) -> None:
    # Рассинхронизация топологии между окружениями обязана падать на старте,
    # а не проявляться потерянными сообщениями.
    await declare_topology(broker, _topology())

    with pytest.raises(ChannelPreconditionFailed):
        await declare_topology(
            broker, build_topology(consumer_timeout_ms=CONSUMER_TIMEOUT_MS + 1)
        )


async def test_command_with_unknown_routing_key_lands_in_unrouted(
    broker: RabbitBroker,
) -> None:
    # Опечатка в ключе иначе теряет команду молча: без ошибки у публикующей
    # стороны и без записи где-либо.
    await declare_topology(broker, _topology())

    await broker.publish(
        b"{}", exchange=COMMANDS_EXCHANGE, routing_key="document.process.typo"
    )

    assert await _message_count(broker, UNROUTED_QUEUE) == 1


async def test_retry_base_routing_key_is_bound_to_the_first_level(
    broker: RabbitBroker,
) -> None:
    # Ключ устава без суффикса обязан работать и давать первую ступень.
    await declare_topology(broker, _topology())

    await broker.publish(b"{}", exchange="documents.retry", routing_key=RK_RETRY_BASE)

    assert await _message_count(broker, f"{RK_RETRY_BASE}.{RETRY_LADDER[0][0]}.q") == 1


async def test_audit_queue_is_declared_only_when_asked(broker: RabbitBroker) -> None:
    await declare_topology(broker, _topology(declare_audit_queue=True))

    assert await _message_count(broker, "document.events.audit.q") == 0


@pytest.mark.parametrize(
    ("attempt", "level"),
    [(number, level) for number, (level, _) in enumerate(RETRY_LADDER, start=1)],
)
def test_retry_ladder_selects_queue_by_attempt_number(attempt: int, level: str) -> None:
    assert retry_level_for_attempt(attempt) == level


@pytest.mark.parametrize("attempt", [0, len(RETRY_LADDER) + 1])
def test_attempt_outside_the_ladder_is_rejected(attempt: int) -> None:
    with pytest.raises(ValueError, match="вне лестницы"):
        retry_level_for_attempt(attempt)


async def _message_count(broker: RabbitBroker, queue: str) -> int:
    channel = await broker._connection.channel()
    declared = await channel.declare_queue(queue, passive=True)
    return int(declared.declaration_result.message_count or 0)
