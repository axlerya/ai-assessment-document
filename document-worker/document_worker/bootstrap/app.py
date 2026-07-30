"""Сборка ASGI-приложения: health-пробы и метрики живут на одном порту."""

from __future__ import annotations

from faststream.asgi import AsgiFastStream, AsgiResponse, get

# Scope импортируется в рантайме: fast_depends резолвит аннотации обработчика
# через get_type_hints() и имя из блока TYPE_CHECKING не найдёт.
from faststream.asgi.types import Scope

LIVENESS_PATH = "/health/live"

_HTTP_OK = 200
_ALIVE_RESPONSE = AsgiResponse(body=b"alive", status_code=_HTTP_OK)


@get
async def liveness_route(scope: Scope) -> AsgiResponse:
    """Отвечает 200, пока процесс жив. Внешние системы — забота /health/ready."""
    del scope
    return _ALIVE_RESPONSE


def create_app() -> AsgiFastStream:
    """Собирает ASGI-приложение сервиса."""
    return AsgiFastStream(asgi_routes=[(LIVENESS_PATH, liveness_route)])
