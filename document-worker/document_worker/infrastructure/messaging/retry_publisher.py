"""Публикация отложенных повторов и копий на разбор.

Повтор — это публикация приложения, а не dead-lettering рабочей очереди:
`x-dead-letter-routing-key` статичен, поэтому выбрать ступень задержки по
номеру попытки при дедлеттеринге невозможно.

Публикуется сырое тело, а не пересериализованная модель: копия байт-в-байт
равна оригиналу, и различие версий библиотеки между релизами не может
незаметно изменить payload.

Пара «публикация копии → подтверждение оригинала» неатомарна, и порядок строго
такой: сбой между ними даёт дубль, а не потерю, а дубль гасит идемпотентность.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from document_worker.infrastructure.messaging.topology import RK_DLQ, RK_RETRY_BASE

if TYPE_CHECKING:
    from faststream.rabbit import RabbitBroker, RabbitExchange

    from document_worker.infrastructure.messaging.topology import Topology

DEFAULT_PUBLISH_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class RetryPublisher:
    """Кладёт копии сообщений на ступени задержки и в очередь разбора."""

    broker: RabbitBroker
    topology: Topology
    publish_timeout_s: float = DEFAULT_PUBLISH_TIMEOUT_S

    async def schedule(
        self,
        body: bytes,
        headers: dict[str, Any],
        *,
        attempt: int,
    ) -> None:
        """Кладёт копию на ступень задержки, соответствующую номеру попытки.

        Raises:
            ValueError: Номер попытки вне лестницы.
        """
        levels = self.topology.levels
        if not 1 <= attempt <= len(levels):
            raise ValueError(f"попытка {attempt} вне лестницы повторов")
        await self._publish(
            body,
            headers,
            exchange=self.topology.retry,
            routing_key=f"{RK_RETRY_BASE}.{levels[attempt - 1]}",
        )

    async def send_to_dlq(self, body: bytes, headers: dict[str, Any]) -> None:
        """Кладёт копию в очередь разбора.

        Публикуется копия, а не reject оригинала: отклонённое сообщение уходит
        в DLQ как есть, и оператор получает тело без причины отказа.
        """
        await self._publish(
            body,
            headers,
            exchange=self.topology.dead_letter_exchange,
            routing_key=RK_DLQ,
        )

    async def _publish(
        self,
        body: bytes,
        headers: dict[str, Any],
        *,
        exchange: RabbitExchange,
        routing_key: str,
    ) -> None:
        await self.broker.publish(
            body,
            exchange=exchange,
            routing_key=routing_key,
            headers=headers,
            content_type="application/json",
            persist=True,
            timeout=self.publish_timeout_s,
        )
