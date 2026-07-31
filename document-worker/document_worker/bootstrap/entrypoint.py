"""Точка входа сервиса.

Остановка обязана быть аккуратной: брокер перестаёт брать новые сообщения,
текущее доводится до конца, реле снимается, и только потом освобождаются
соединения. Оборванная посередине обработка стоит документу лишней попытки,
а при выкатке таких обрывов столько же, сколько подов.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import uvicorn

from document_worker.bootstrap.app import create_app
from document_worker.bootstrap.composition import build_services
from document_worker.bootstrap.outbox import OutboxRelay, running
from document_worker.infrastructure.config.settings import AppSettings
from document_worker.infrastructure.health import BrokerProbe, DatabaseProbe
from document_worker.observability.logging import configure_logging
from document_worker.observability.metrics import Metrics

if TYPE_CHECKING:
    from document_worker.application.ports.health import HealthProbe

HTTP_HOST = "0.0.0.0"  # noqa: S104 — в контейнере слушаем все интерфейсы
HTTP_PORT = 8080


async def serve(
    settings: AppSettings,
) -> None:  # pragma: no cover — проверяется запуском контейнера
    """Поднимает сервис и держит его до остановки."""
    async with build_services(settings) as services:
        probes: list[HealthProbe] = [
            DatabaseProbe(engine=services.engine),
            BrokerProbe(broker=services.broker),
        ]
        relay = OutboxRelay(
            publish=services.publish_outbox, config=settings.processing_config().outbox
        )
        async with running(relay):
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(services.broker, probes, Metrics()),
                    host=HTTP_HOST,
                    port=HTTP_PORT,
                    log_config=None,
                )
            )
            await server.serve()


def run() -> None:  # pragma: no cover — проверяется запуском контейнера
    """Запускает сервис."""
    configure_logging()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(AppSettings()))
