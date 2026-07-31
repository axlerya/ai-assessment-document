"""Публикация событий в шину поверх соединения FastStream.

Публикация ждёт подтверждения брокера: неподтверждённая отправка — это его
недоступность, а не успех, и отметить такое событие опубликованным значит
потерять его навсегда.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faststream.rabbit import RabbitBroker, RabbitExchange

    from document_worker.application.dto.results import OutboxRecordDTO

HEADER_EVENT_ID = "x-event-id"
HEADER_CORRELATION_ID = "x-correlation-id"


@dataclass(frozen=True, slots=True)
class RabbitEventPublisher:
    """Отправляет строку outbox в обменник событий."""

    broker: RabbitBroker
    exchange: RabbitExchange
    publish_timeout_s: float = 5.0

    async def publish(self, record: OutboxRecordDTO) -> None:
        """Публикует одно событие и ждёт подтверждения брокера."""
        await self.broker.publish(
            json.dumps(record.payload, ensure_ascii=False).encode(),
            exchange=self.exchange,
            routing_key=record.routing_key,
            headers={
                **record.headers,
                HEADER_EVENT_ID: str(record.event_id),
                HEADER_CORRELATION_ID: record.correlation_id or "",
            },
            correlation_id=record.correlation_id,
            message_id=str(record.event_id),
            content_type="application/json",
            persist=True,
            timeout=self.publish_timeout_s,
        )
