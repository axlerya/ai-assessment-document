"""Точка входа сервиса для `[project.scripts]`."""

from __future__ import annotations

import uvicorn

from document_worker.bootstrap.app import create_app

# Станут полями AppSettings, когда появится слой конфигурации.
HTTP_HOST = "0.0.0.0"  # noqa: S104 — в контейнере слушаем все интерфейсы
HTTP_PORT = 8080


def run() -> None:
    """Запускает ASGI-приложение сервиса."""
    uvicorn.run(create_app(), host=HTTP_HOST, port=HTTP_PORT, log_config=None)
