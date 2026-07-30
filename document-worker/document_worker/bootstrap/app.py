"""Сборка ASGI-приложения сервиса (ADR-0008).

Приложение — `AsgiFastStream` на единственном порту: на нём же живут health-пробы
и метрики. Второго HTTP-сервера в проекте нет.
"""

from __future__ import annotations

from faststream.asgi import AsgiFastStream, AsgiResponse, get

# Scope импортируется в рантайме намеренно: fast_depends резолвит аннотации
# обработчика через get_type_hints(), и имя из блока TYPE_CHECKING он не найдёт.
from faststream.asgi.types import Scope

LIVENESS_PATH = "/health/live"

_HTTP_OK = 200
_ALIVE_RESPONSE = AsgiResponse(body=b"alive", status_code=_HTTP_OK)


@get
async def liveness_route(scope: Scope) -> AsgiResponse:
    """Отвечает 200, пока процесс жив.

    Проба намеренно не обращается ни к PostgreSQL, ни к брокеру, ни к S3:
    недоступность зависимостей — тема `/health/ready`, а не `/health/live`.

    Args:
        scope: ASGI-scope запроса.

    Returns:
        Ответ с кодом 200.
    """
    del scope
    return _ALIVE_RESPONSE


def create_app() -> AsgiFastStream:
    """Собирает ASGI-приложение сервиса.

    Returns:
        Приложение FastStream с HTTP-маршрутами сервиса.
    """
    return AsgiFastStream(asgi_routes=[(LIVENESS_PATH, liveness_route)])
