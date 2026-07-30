"""Тесты композиционного корня: ASGI-приложение и liveness-эндпоинт (ADR-0008)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import uvicorn
from faststream.asgi import AsgiFastStream

from document_worker.bootstrap import entrypoint
from document_worker.bootstrap.app import LIVENESS_PATH, create_app

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from faststream.asgi.types import ASGIApp

pytestmark = pytest.mark.unit

HTTP_OK = 200


async def _call_asgi(app: ASGIApp, path: str) -> tuple[int, bytes]:
    """Вызывает ASGI-приложение напрямую, без поднятия HTTP-сервера.

    Args:
        app: ASGI-приложение.
        path: Путь запроса.

    Returns:
        Код ответа и тело ответа.
    """
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 51234),
        "server": ("127.0.0.1", 8080),
    }
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> MutableMapping[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return status, body


def test_create_app_returns_asgi_application() -> None:
    app = create_app()

    assert isinstance(app, AsgiFastStream), (
        "ADR-0008: приложение — AsgiFastStream, второго HTTP-сервера в проекте нет"
    )


async def test_liveness_endpoint_returns_200_without_dependencies() -> None:
    app = create_app()

    status, body = await _call_asgi(app, LIVENESS_PATH)

    assert status == HTTP_OK, "liveness обязан отвечать, не трогая внешние системы"
    assert body


def test_run_starts_asgi_server_on_port_8080(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)

    entrypoint.run()

    assert isinstance(captured["app"], AsgiFastStream)
    assert captured["kwargs"]["port"] == 8080, "ADR-0008: единственный порт — 8080"
