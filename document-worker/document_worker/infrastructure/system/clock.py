"""Источник времени и идентификаторов.

Прямых вызовов `datetime.now()` и `uuid4()` в слоях нет: иначе поведение,
зависящее от времени, нельзя ни повторить, ни проверить.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


class SystemClock:
    """Системное время в UTC."""

    def now(self) -> datetime:
        """Текущий момент в UTC с указанием зоны."""
        return datetime.now(UTC)


class Uuid4IdGenerator:
    """Случайные идентификаторы."""

    def new_uuid(self) -> uuid.UUID:
        """Новый случайный UUID."""
        return uuid.uuid4()
