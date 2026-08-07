"""Готовность сервиса."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Исход одной проверки готовности."""

    name: str
    healthy: bool
    detail: str | None = None


@runtime_checkable
class HealthProbe(Protocol):
    """Проверяет, что внешняя система отвечает."""

    async def check(self) -> HealthStatus:
        """Опрашивает систему и возвращает вердикт."""
        ...
