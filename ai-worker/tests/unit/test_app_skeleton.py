"""Сборка ASGI-приложения: проба живости отвечает без внешних систем."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pytest
from faststream.asgi import AsgiFastStream
from faststream.rabbit import RabbitBroker

from ai_worker.bootstrap.app import HTTP_OK, LIVENESS_PATH, create_app

pytestmark = pytest.mark.unit

# Форма сообщения ASGI: словарь, но принимающая сторона объявляет его
# изменяемым отображением, и точное совпадение типа требует mypy --strict.
type Message = MutableMapping[str, Any]

# Адрес заведомо недоступен: проба живости не имеет права о нём знать.
UNREACHABLE_BROKER_URL = "amqp://nobody:nobody@127.0.0.1:1/"
HTTP_NOT_FOUND = 404


def _broker() -> RabbitBroker:
    return RabbitBroker(UNREACHABLE_BROKER_URL)


async def _call(app: AsgiFastStream, path: str) -> list[Message]:
    """Прогоняет запрос через ASGI-интерфейс приложения.

    Args:
        app: Приложение, которое отвечает на запрос.
        path: Запрашиваемый путь.

    Returns:
        Сообщения, отданные приложением в `send`.
    """
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "scheme": "http",
    }
    await app(scope, receive, send)
    return sent


def _status_of(sent: list[Message]) -> int:
    start = next(
        message for message in sent if message["type"] == "http.response.start"
    )
    status: int = start["status"]
    return status


def test_liveness_lives_on_the_expected_path() -> None:
    # Путь зашит в HEALTHCHECK образа и в пробу оркестратора: его смена без
    # правки развёртывания выводит сервис из ротации молча.
    assert LIVENESS_PATH == "/health/live"


def test_create_app_returns_an_asgi_application() -> None:
    app = create_app(_broker())

    assert isinstance(app, AsgiFastStream)
    assert callable(app)


async def test_liveness_answers_without_reaching_the_broker() -> None:
    # Живость отвечает на вопрос «процесс не завис». Падать от недоступного
    # брокера она не имеет права: перезапуск воркера брокер не поднимет.
    sent = await _call(create_app(_broker()), LIVENESS_PATH)

    assert _status_of(sent) == HTTP_OK


async def test_unknown_path_is_not_answered_as_alive() -> None:
    sent = await _call(create_app(_broker()), "/health/ready")

    assert _status_of(sent) == HTTP_NOT_FOUND


def test_liveness_route_is_registered_once() -> None:
    app = create_app(_broker())

    paths = [path for path, _ in app.routes]
    assert paths.count(LIVENESS_PATH) == 1


def test_app_is_built_around_the_given_broker() -> None:
    broker = _broker()

    app = create_app(broker)

    assert broker in app.brokers
