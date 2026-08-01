"""Объявление топологии на живом RabbitMQ.

Аргументы очередей — это поведение брокера, а не наше: `x-expires` тихо
удаляет отложенные сообщения, `x-max-length` теряет команды, `reject-publish`
на DLQ останавливает конвейер. Проверить их можно только объявлением.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from aio_pika.exceptions import ChannelPreconditionFailed
from faststream.rabbit import RabbitBroker

from document_worker.infrastructure.messaging.declare import declare_topology
from document_worker.infrastructure.messaging.topology import (
    AUDIT_QUEUE,
    COMMANDS_EXCHANGE,
    DLQ_MAX_LENGTH,
    DLQ_QUEUE,
    PROCESS_REQUESTED_QUEUE,
    RETRY_EXCHANGE,
    RETRY_LADDER,
    RK_PROCESS_REQUESTED,
    RK_RETRY_BASE,
    UNROUTED_QUEUE,
    build_topology,
    retry_level_for_attempt,
    retry_queue_name,
)

if TYPE_CHECKING:
    from document_worker.infrastructure.messaging.topology import Topology
    from tests.conftest import Management

pytestmark = pytest.mark.integration

CONSUMER_TIMEOUT_MS = 7_200_000
DELIVERY_LIMIT = 20


def _drifted_ladder() -> tuple[tuple[str, int], ...]:
    level, ttl_ms = RETRY_LADDER[0]
    return ((level, ttl_ms + 1_000), *RETRY_LADDER[1:])


def _topology(*, declare_audit_queue: bool = False) -> Topology:
    return build_topology(
        consumer_timeout_ms=CONSUMER_TIMEOUT_MS,
        delivery_limit=DELIVERY_LIMIT,
        declare_audit_queue=declare_audit_queue,
    )


async def test_declare_creates_every_queue(
    broker: RabbitBroker,
    management: Management,
) -> None:
    await declare_topology(broker, _topology())

    for queue in (PROCESS_REQUESTED_QUEUE, DLQ_QUEUE, UNROUTED_QUEUE):
        assert management.queue(queue)["name"] == queue


async def test_main_queue_arguments_match_the_specification(
    broker: RabbitBroker,
    management: Management,
) -> None:
    await declare_topology(broker, _topology())

    arguments = management.arguments_of(PROCESS_REQUESTED_QUEUE)

    assert arguments["x-queue-type"] == "quorum"
    assert arguments["x-delivery-limit"] == DELIVERY_LIMIT
    assert arguments["x-dead-letter-strategy"] == "at-least-once"
    assert arguments["x-overflow"] == "reject-publish"
    assert arguments["x-consumer-timeout"] == CONSUMER_TIMEOUT_MS


async def test_main_queue_has_no_max_length(
    broker: RabbitBroker,
    management: Management,
) -> None:
    # При reject-publish предел длины превратил бы лавину повторов в отказ
    # публикации, при drop-head — в тихую потерю команд.
    await declare_topology(broker, _topology())

    assert "x-max-length" not in management.arguments_of(PROCESS_REQUESTED_QUEUE)


async def test_retry_queues_have_the_expected_ttl_ladder(
    broker: RabbitBroker,
    management: Management,
) -> None:
    await declare_topology(broker, _topology())

    for level, ttl_ms in RETRY_LADDER:
        arguments = management.arguments_of(retry_queue_name(level))
        assert arguments["x-message-ttl"] == ttl_ms
        assert arguments["x-dead-letter-exchange"] == COMMANDS_EXCHANGE
        assert arguments["x-dead-letter-routing-key"] == RK_PROCESS_REQUESTED


async def test_retry_queues_have_no_expires_argument(
    broker: RabbitBroker,
    management: Management,
) -> None:
    # Очередь с истёкшим x-expires удаляется вместе с содержимым, и отложенные
    # сообщения при этом не дедлеттерятся — они просто исчезают.
    await declare_topology(broker, _topology())

    for level, _ in RETRY_LADDER:
        assert "x-expires" not in management.arguments_of(retry_queue_name(level))


async def test_dlq_drops_the_oldest_instead_of_rejecting(
    broker: RabbitBroker,
    management: Management,
) -> None:
    # Переполненная DLQ с reject-publish перестала бы принимать dead-letter,
    # и переполнение разбора заблокировало бы обработку живых документов.
    await declare_topology(broker, _topology())

    arguments = management.arguments_of(DLQ_QUEUE)

    assert arguments["x-overflow"] == "drop-head"
    assert arguments["x-queue-type"] == "quorum"
    assert arguments["x-max-length"] == DLQ_MAX_LENGTH


async def test_declare_is_idempotent(
    broker: RabbitBroker,
    management: Management,
) -> None:
    topology = _topology()
    await declare_topology(broker, topology)

    await declare_topology(broker, topology)

    assert management.arguments_of(PROCESS_REQUESTED_QUEUE)


async def test_declare_fails_on_argument_drift(
    broker: RabbitBroker,
    rabbitmq_url: str,
) -> None:
    # Рассинхронизация топологии между окружениями обязана падать на старте,
    # а не проявляться потерянными сообщениями. Второе соединение здесь — это
    # второй инстанс с разъехавшейся настройкой, а не повтор в том же процессе.
    await declare_topology(broker, _topology())
    drifted = RabbitBroker(rabbitmq_url)
    await drifted.connect()

    try:
        with pytest.raises(ChannelPreconditionFailed):
            await declare_topology(
                drifted,
                build_topology(
                    consumer_timeout_ms=CONSUMER_TIMEOUT_MS,
                    retry_ladder=_drifted_ladder(),
                ),
            )
    finally:
        await drifted.stop()


async def test_command_with_unknown_routing_key_lands_in_unrouted(
    broker: RabbitBroker,
    management: Management,
) -> None:
    # Опечатка в ключе иначе теряет команду молча: без ошибки у публикующей
    # стороны и без записи где-либо.
    await declare_topology(broker, _topology())
    before = len(management.fetch(UNROUTED_QUEUE))

    await broker.publish(
        b"{}", exchange=COMMANDS_EXCHANGE, routing_key="document.process.typo"
    )

    assert await _grew_to(management, UNROUTED_QUEUE, before + 1)


async def test_retry_base_routing_key_is_bound_to_the_first_level(
    broker: RabbitBroker,
    management: Management,
) -> None:
    # Ключ устава без суффикса обязан работать и давать первую ступень.
    await declare_topology(broker, _topology())
    first = retry_queue_name(RETRY_LADDER[0][0])
    before = len(management.fetch(first))

    await broker.publish(b"{}", exchange=RETRY_EXCHANGE, routing_key=RK_RETRY_BASE)

    assert await _grew_to(management, first, before + 1)


async def test_audit_queue_is_declared_only_when_asked(
    broker: RabbitBroker,
    management: Management,
) -> None:
    # В проде у обменника событий может не быть ни одного binding'а, и тогда
    # все события уходят в ловушку — очередь аудита нужна только dev и CI.
    assert AUDIT_QUEUE not in {queue.name for queue in _topology().queues}

    await declare_topology(broker, _topology(declare_audit_queue=True))

    assert management.queue(AUDIT_QUEUE)["name"] == AUDIT_QUEUE


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


async def _grew_to(management: Management, queue: str, expected: int) -> bool:
    """Ждёт появления сообщения в очереди.

    Считается содержимое, а не статистика: `messages` у quorum-очередей
    обновляется циклом в несколько секунд и в этот момент показывает ноль.
    """
    for _ in range(100):
        if len(management.fetch(queue)) >= expected:
            return True
        await asyncio.sleep(0.1)
    return False
