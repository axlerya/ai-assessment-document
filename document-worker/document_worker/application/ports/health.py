"""Пробы готовности.

Порт нужен, чтобы presentation не импортировал ORM и драйверы ради health.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Итог одной пробы."""

    name: str
    healthy: bool
    detail: str | None = None


@runtime_checkable
class HealthProbe(Protocol):
    """Проверяет доступность одной внешней системы."""

    @property
    def name(self) -> str:
        """Имя проверяемой системы."""
        ...

    async def check(self) -> HealthStatus:
        """Выполняет проверку."""
        ...
