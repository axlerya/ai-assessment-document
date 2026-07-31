"""Сборка ASGI-приложения: health-пробы живут на одном порту с сервисом.

Живость и готовность отвечают на разные вопросы. Живость — «процесс не завис»,
и падать от недоступной базы она не имеет права: перезапуск воркера базу не
поднимет, а очередь тем временем растёт. Готовность — «сейчас есть смысл
давать работу», и вот она внешние системы проверяет.

Проба готовности спрашивает у базы только живость соединения, а не соответствие
схемы: аддитивная миграция, накатанная раньше кода, не должна выводить под из
ротации — иначе выкатка ломает сама себя.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from faststream.asgi import AsgiFastStream, AsgiResponse, get

# Scope импортируется в рантайме: fast_depends резолвит аннотации обработчика
# через get_type_hints() и имя из блока TYPE_CHECKING не найдёт.
from faststream.asgi.types import Scope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faststream.rabbit import RabbitBroker

    from document_worker.application.ports.health import HealthProbe

LIVENESS_PATH = "/health/live"
READINESS_PATH = "/health/ready"

HTTP_OK = 200
HTTP_UNAVAILABLE = 503

_ALIVE_RESPONSE = AsgiResponse(body=b"alive", status_code=HTTP_OK)


@get
async def liveness_route(scope: Scope) -> AsgiResponse:
    """Отвечает 200, пока процесс жив. Внешние системы — забота готовности."""
    del scope
    return _ALIVE_RESPONSE


async def readiness(probes: Sequence[HealthProbe]) -> tuple[int, dict[str, Any]]:
    """Опрашивает пробы и собирает ответ готовности."""
    statuses = [await probe.check() for probe in probes]
    healthy = all(status.healthy for status in statuses)
    payload = {
        "ready": healthy,
        "checks": [
            {"name": status.name, "healthy": status.healthy, "detail": status.detail}
            for status in statuses
        ],
    }
    return HTTP_OK if healthy else HTTP_UNAVAILABLE, payload


def build_readiness_route(probes: Sequence[HealthProbe]) -> Any:
    """Оборачивает опрос проб в маршрут ASGI."""

    @get
    async def readiness_route(scope: Scope) -> AsgiResponse:
        del scope
        status_code, payload = await readiness(probes)
        return AsgiResponse(
            body=json.dumps(payload, ensure_ascii=False).encode(),
            status_code=status_code,
            headers={"content-type": "application/json"},
        )

    return readiness_route


def create_app(
    broker: RabbitBroker,
    probes: Sequence[HealthProbe],
) -> AsgiFastStream:
    """Собирает ASGI-приложение сервиса поверх подключённого брокера."""
    return AsgiFastStream(
        broker,
        asgi_routes=[
            (LIVENESS_PATH, liveness_route),
            (READINESS_PATH, build_readiness_route(probes)),
        ],
    )
