"""Управляемые время и идентификаторы.

Тест, зависящий от системных часов, воспроизвести нельзя: лиз то протух, то
нет. Здесь время двигает сам тест.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import TYPE_CHECKING

from document_worker.domain.constants import NS_DOCWORKER

if TYPE_CHECKING:
    from datetime import datetime


class FixedClock:
    """Часы, которые стоят, пока их не подвинут."""

    def __init__(self, moment: datetime) -> None:
        """Ставит часы на указанный момент."""
        self._moment = moment

    def now(self) -> datetime:
        """Текущее показание."""
        return self._moment

    def advance(self, *, seconds: float) -> datetime:
        """Двигает часы вперёд."""
        self._moment += timedelta(seconds=seconds)
        return self._moment


class SequentialIdGenerator:
    """Идентификаторы, предсказуемые по порядку вызова."""

    def __init__(self) -> None:
        """Начинает нумерацию с нуля."""
        self.issued = 0

    def new_uuid(self) -> uuid.UUID:
        """Следующий идентификатор последовательности."""
        self.issued += 1
        return uuid.uuid5(NS_DOCWORKER, f"sequential:{self.issued}")
