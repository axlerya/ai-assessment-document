"""Перевод доменных событий в строки outbox.

Домен про доставку не знает, поэтому ключ маршрутизации и представление
полезной нагрузки живут здесь. `event_id` не выдумывается: он детерминирован
по документу, версии пайплайна и типу события, и повторное завершение даёт
тот же ключ, который гасит уникальное ограничение outbox.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from document_worker.application.dto.results import OutboxEventDTO
from document_worker.domain.value_objects.confidence import OcrConfidence

if TYPE_CHECKING:
    from document_worker.domain.events import DomainEvent


def to_outbox_event(event: DomainEvent) -> OutboxEventDTO:
    """Готовит доменное событие к записи в outbox."""
    event_type = type(event).event_type
    return OutboxEventDTO(
        event_id=event.event_id.value,
        aggregate_id=event.document_id.value,
        event_type=event_type,
        # Топологию объявляет messaging; здесь ключ совпадает с типом события,
        # и второго источника имени не заводится.
        routing_key=event_type,
        payload={
            field.name: _jsonable(getattr(event, field.name)) for field in fields(event)
        },
        correlation_id=event.correlation_id,
        occurred_at=event.occurred_at,
    )


def _jsonable(value: object) -> object:
    # Проверка на StrEnum идёт до строковой: член перечисления и есть строка.
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, OcrConfidence):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    # Идентификаторы и версии печатаются своим каноническим представлением.
    return str(value)
