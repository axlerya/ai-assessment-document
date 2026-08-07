"""Системное окружение сценариев."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime


@runtime_checkable
class Clock(Protocol):
    """Источник текущего момента.

    Порт, а не прямой вызов: сценарии пишут отметки времени в базу и
    сравнивают их между собой, и подменяемые часы — единственный способ
    проверить это тестом, не завися от скорости машины.
    """

    def now(self) -> datetime:
        """Текущий момент в UTC с указанием зоны."""
        ...
