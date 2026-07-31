"""Пробы живости и готовности.

Живость и готовность отвечают на разные вопросы, и путать их дорого: живость,
падающая от недоступной базы, вызывает перезапуск воркера — база от этого не
поднимется, а очередь тем временем растёт.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from faststream.rabbit import RabbitBroker

from document_worker.application.ports.health import HealthStatus
from document_worker.bootstrap.app import (
    HTTP_OK,
    HTTP_UNAVAILABLE,
    LIVENESS_PATH,
    READINESS_PATH,
    build_readiness_route,
    create_app,
    liveness_route,
    readiness,
)
from document_worker.infrastructure.health import BrokerProbe, DatabaseProbe
from document_worker.infrastructure.persistence.engine import build_engine

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from document_worker.application.ports.health import HealthProbe

pytestmark = pytest.mark.integration

SCOPE = {"type": "http", "method": "GET", "path": "/health/live", "headers": []}


class BrokenProbe:
    """Система, до которой не достучаться."""

    @property
    def name(self) -> str:
        """Имя проверяемой системы."""
        return "broken"

    async def check(self) -> HealthStatus:
        """Всегда отвечает отказом."""
        return HealthStatus(name=self.name, healthy=False, detail="соединения нет")


async def test_readiness_is_green_when_the_database_answers(
    migrated_engine: AsyncEngine,
) -> None:
    probes: list[HealthProbe] = [DatabaseProbe(engine=migrated_engine)]

    status, payload = await readiness(probes)

    assert status == HTTP_OK
    assert payload["ready"] is True


async def test_readiness_is_red_when_a_system_is_down() -> None:
    status, payload = await readiness([BrokenProbe()])

    assert status == HTTP_UNAVAILABLE
    assert payload["ready"] is False


async def test_readiness_names_the_system_that_failed() -> None:
    # Ответ без имени системы заставляет оператора искать причину вслепую.
    _, payload = await readiness([BrokenProbe()])

    checks = payload["checks"]
    assert checks[0]["name"] == "broken"
    assert checks[0]["detail"]


async def test_unreachable_database_is_a_failed_probe_not_a_crash() -> None:
    # Проба, поднимающая исключение, роняет весь ответ готовности.
    dead = build_engine(
        "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none",
        pool_size=1,
        max_overflow=0,
        pool_timeout_s=1.0,
    )
    try:
        status, _ = await readiness([DatabaseProbe(engine=dead)])
    finally:
        await dead.dispose()

    assert status == HTTP_UNAVAILABLE


def test_liveness_and_readiness_live_on_different_paths() -> None:
    # Живость, падающая от недоступной базы, вызывает перезапуск воркера — база
    # от этого не поднимется, а очередь тем временем растёт.
    assert LIVENESS_PATH != READINESS_PATH


async def test_broker_probe_sees_a_live_connection(rabbitmq_url: str) -> None:
    connected = RabbitBroker(rabbitmq_url)
    await connected.connect()
    try:
        assert (await BrokerProbe(broker=connected).check()).healthy
    finally:
        await connected.stop()


async def test_broker_probe_reports_a_dead_connection() -> None:
    # Неподключённый брокер обязан быть красной пробой, а не исключением.
    probe = BrokerProbe(broker=RabbitBroker("amqp://127.0.0.1:1"))

    assert not (await probe.check()).healthy


async def test_routes_answer_over_asgi(migrated_engine: AsyncEngine) -> None:
    # Маршруты — это обёртки над логикой проб, и вызвать их можно только так.
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request"}

    async def send(message: Any) -> None:
        sent.append(message)

    await liveness_route(SCOPE, receive, send)
    await build_readiness_route([DatabaseProbe(engine=migrated_engine)])(
        SCOPE, receive, send
    )

    starts = [message for message in sent if message["type"] == "http.response.start"]
    assert [message["status"] for message in starts] == [HTTP_OK, HTTP_OK]


def test_app_wires_both_probes(migrated_engine: AsyncEngine) -> None:
    app = create_app(
        RabbitBroker("amqp://127.0.0.1:1"), [DatabaseProbe(engine=migrated_engine)]
    )

    assert app is not None
