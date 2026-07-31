"""Объявление топологии. Идемпотентно и выполняется до старта потребителей.

Расхождение аргументов с уже существующей очередью не глушится: топология,
разъехавшаяся между окружениями, обязана падать на старте, а не проявляться
потерянными сообщениями.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faststream.rabbit import RabbitBroker

    from document_worker.infrastructure.messaging.topology import Topology


async def declare_topology(broker: RabbitBroker, topology: Topology) -> None:
    """Объявляет обменники, очереди и связи между ними."""
    exchanges = {
        exchange.name: await broker.declare_exchange(exchange)
        for exchange in topology.exchanges
    }
    queues = {
        queue.name: await broker.declare_queue(queue) for queue in topology.queues
    }
    for binding in topology.bindings:
        await queues[binding.queue].bind(
            exchanges[binding.exchange], routing_key=binding.routing_key
        )
