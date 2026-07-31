"""Пробы доступности внешних систем.

Проба базы делает самый дешёвый запрос, какой есть: она отвечает на вопрос
«соединение живо», а не «схема та». Аддитивная миграция, накатанная раньше
кода, не должна выводить под из ротации — иначе выкатка ломает сама себя.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text

from document_worker.application.ports.health import HealthStatus

if TYPE_CHECKING:
    from faststream.rabbit import RabbitBroker
    from sqlalchemy.ext.asyncio import AsyncEngine

PROBE_TIMEOUT_S = 3.0


@dataclass(frozen=True, slots=True)
class DatabaseProbe:
    """Проверяет, что соединение с PostgreSQL живо."""

    engine: AsyncEngine

    @property
    def name(self) -> str:
        """Имя проверяемой системы."""
        return "postgres"

    async def check(self) -> HealthStatus:
        """Берёт соединение из пула и выполняет тривиальный запрос."""
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as error:  # noqa: BLE001 — причина уходит в ответ пробы
            return HealthStatus(name=self.name, healthy=False, detail=str(error)[:200])
        return HealthStatus(name=self.name, healthy=True)


@dataclass(frozen=True, slots=True)
class BrokerProbe:
    """Проверяет, что соединение с брокером установлено."""

    broker: RabbitBroker

    @property
    def name(self) -> str:
        """Имя проверяемой системы."""
        return "rabbitmq"

    async def check(self) -> HealthStatus:
        """Спрашивает у брокера его собственное состояние."""
        try:
            alive = await self.broker.ping(timeout=PROBE_TIMEOUT_S)
        # Библиотека отвечает на недоступность ложью, а не исключением.
        except Exception as error:  # noqa: BLE001 — причина уходит в ответ пробы  # pragma: no cover
            return HealthStatus(name=self.name, healthy=False, detail=str(error)[:200])
        return HealthStatus(
            name=self.name,
            healthy=alive,
            detail=None if alive else "соединение с брокером не установлено",
        )
