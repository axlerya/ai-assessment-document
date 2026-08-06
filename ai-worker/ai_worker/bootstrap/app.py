"""Сборка ASGI-приложения: пробы живут на одном порту с сервисом.

Живость и готовность отвечают на разные вопросы. Живость — «процесс не завис»,
и падать от недоступного брокера или базы она не имеет права: перезапуск
воркера их не поднимет, а очередь тем временем растёт. Готовность — «сейчас
есть смысл давать работу», и вот она внешние системы проверяет; она появится
вместе с ними, а не раньше.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from faststream.asgi import AsgiFastStream, AsgiResponse, get

# Scope импортируется в рантайме: fast_depends резолвит аннотации обработчика
# через get_type_hints() и имя из блока TYPE_CHECKING не найдёт.
from faststream.asgi.types import Scope

if TYPE_CHECKING:
    from faststream.rabbit import RabbitBroker

LIVENESS_PATH = "/health/live"

HTTP_OK = 200

_ALIVE_RESPONSE = AsgiResponse(body=b"alive", status_code=HTTP_OK)


@get
async def liveness_route(scope: Scope) -> AsgiResponse:
    """Отвечает 200, пока процесс жив. Внешние системы — забота готовности."""
    del scope
    return _ALIVE_RESPONSE


def create_app(broker: RabbitBroker) -> AsgiFastStream:
    """Собирает ASGI-приложение сервиса поверх переданного брокера."""
    return AsgiFastStream(broker, asgi_routes=[(LIVENESS_PATH, liveness_route)])
