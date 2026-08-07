"""Идемпотентность доставки и накопитель исходящих событий."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID


class ClaimOutcome(StrEnum):
    """Чем закончилась попытка занять сообщение.

    Четыре исхода — полный словарь: пятого случая у захвата не бывает, и
    подписчик обязан уметь обработать каждый.
    """

    PROCEED = "proceed"
    RESUME = "resume"
    REJECT_CONCURRENT = "reject_concurrent"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """Что делать с сообщением после попытки его занять."""

    outcome: ClaimOutcome
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class OutboxEventDTO:
    """Готовое к записи в outbox событие."""

    event_id: UUID
    aggregate_id: UUID
    event_type: str
    routing_key: str
    payload: dict[str, Any]
    correlation_id: str
    occurred_at: datetime
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutboxRecordDTO:
    """Строка outbox, взятая в публикацию."""

    id: int
    event_id: UUID
    routing_key: str
    payload: dict[str, Any]
    headers: dict[str, str]
    correlation_id: str
    occurred_at: datetime
    attempts: int
