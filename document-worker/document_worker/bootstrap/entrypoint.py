"""Точка входа сервиса для `[project.scripts]` (ADR-0008)."""

from __future__ import annotations

import uvicorn

from document_worker.bootstrap.app import create_app

# Значения станут полями AppSettings, когда появится слой конфигурации.
HTTP_HOST = "0.0.0.0"  # noqa: S104 — слушать все интерфейсы внутри контейнера норма
HTTP_PORT = 8080  # ADR-0008: единственный порт сервиса


def run() -> None:
    """Запускает ASGI-приложение сервиса."""
    uvicorn.run(create_app(), host=HTTP_HOST, port=HTTP_PORT, log_config=None)
