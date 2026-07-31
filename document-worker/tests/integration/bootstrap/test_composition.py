"""Композиционный корень на живых зависимостях.

Собрать граф можно только там, где есть база, брокер и хранилище: расхождение
конструкторов иначе вскрывается при первом запуске контейнера, а не в тесте.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.bootstrap.composition import build_services
from document_worker.infrastructure.config.settings import AppSettings
from document_worker.infrastructure.messaging.topology import PROCESS_REQUESTED_QUEUE

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine
    from testcontainers.community.minio import MinioContainer

    from tests.integration.conftest import Management

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(  # noqa: PLR0913, PLR0917 — настройки собираются из всех живых зависимостей
    monkeypatch: pytest.MonkeyPatch,
    migrated_engine: AsyncEngine,
    rabbitmq_url: str,
    isolated_vhost: str,
    minio_container: MinioContainer,
    tmp_path: Path,
) -> AppSettings:
    """Настройки, направленные на поднятые контейнеры."""
    credentials = minio_container.get_config()
    monkeypatch.setenv(
        "DATABASE__DSN",
        migrated_engine.url.render_as_string(hide_password=False),
    )
    monkeypatch.setenv("RABBIT__URL", f"{rabbitmq_url}{isolated_vhost}")
    monkeypatch.setenv("S3__ENDPOINT_URL", f"http://{credentials['endpoint']}")
    monkeypatch.setenv("S3__ACCESS_KEY", credentials["access_key"])
    monkeypatch.setenv("S3__SECRET_KEY", credentials["secret_key"])
    monkeypatch.setenv("PROCESSING__TEMP_DIR", str(tmp_path / "work"))
    return AppSettings()


async def test_services_assemble_the_orchestrator(settings: AppSettings) -> None:
    async with build_services(settings) as services:
        assert services.process_document.config.consumer_name
        assert services.router.subscribers


async def test_topology_is_declared_before_consumers_start(
    settings: AppSettings,
    management: Management,
    isolated_vhost: str,
) -> None:
    # Подписчик на очередь, которой ещё нет, поднял бы её с чужими аргументами.
    async with build_services(settings):
        arguments = management.arguments_of(
            PROCESS_REQUESTED_QUEUE, vhost=isolated_vhost
        )

    assert arguments["x-queue-type"] == "quorum"


async def test_temp_directory_is_created(settings: AppSettings) -> None:
    async with build_services(settings):
        assert settings.processing.temp_dir.is_dir()


async def test_resources_are_released_on_exit(settings: AppSettings) -> None:
    # Незакрытый пул процессов переживает остановку сервиса и держит память.
    async with build_services(settings) as services:
        broker = services.broker

    assert not broker.running
