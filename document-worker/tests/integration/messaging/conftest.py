"""Живой RabbitMQ для тестов топологии.

Аргументы очередей проверяются только настоящим брокером: in-memory брокер
FastStream их не хранит, а AMQP их не отдаёт — passive-declare возвращает лишь
имя и счётчики. Поэтому рядом поднимается management API.
"""

from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pytest
import pytest_asyncio
from faststream.rabbit import RabbitBroker
from testcontainers.community.rabbitmq import RabbitMqContainer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

RABBITMQ_IMAGE = "rabbitmq:4.2-management-alpine"
MANAGEMENT_PORT = 15672
DEFAULT_VHOST = "/"


@dataclass(frozen=True, slots=True)
class Management:
    """Тонкий клиент management API: только то, чего не отдаёт AMQP."""

    base_url: str
    auth: str

    def queue(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Описание очереди так, как его хранит брокер."""
        return self._get(f"queues/{quote(vhost, safe='')}/{quote(name, safe='')}")

    def arguments_of(self, name: str) -> dict[str, Any]:
        """Аргументы очереди."""
        return dict(self.queue(name).get("arguments", {}))

    def message_count(self, name: str) -> int:
        """Сколько сообщений лежит в очереди."""
        return int(self.queue(name).get("messages", 0))

    def _get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310 — адрес контейнера теста
            f"{self.base_url}/api/{path}",
            headers={"Authorization": f"Basic {self.auth}"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            payload: dict[str, Any] = json.load(response)
        return payload


@pytest.fixture(scope="session")
def rabbitmq_container() -> Iterator[RabbitMqContainer]:
    """Поднимает RabbitMQ один раз на весь прогон."""
    container = RabbitMqContainer(RABBITMQ_IMAGE)
    container.with_exposed_ports(MANAGEMENT_PORT)
    with container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_url(rabbitmq_container: RabbitMqContainer) -> str:
    """URL подключения к контейнеру."""
    params = rabbitmq_container.get_connection_params()
    return (
        f"amqp://{params.credentials.username}:{params.credentials.password}"
        f"@{params.host}:{params.port}/"
    )


@pytest.fixture(scope="session")
def management(rabbitmq_container: RabbitMqContainer) -> Management:
    """Доступ к management API того же контейнера."""
    params = rabbitmq_container.get_connection_params()
    host = rabbitmq_container.get_container_host_ip()
    port = rabbitmq_container.get_exposed_port(MANAGEMENT_PORT)
    credentials = f"{params.credentials.username}:{params.credentials.password}"
    return Management(
        base_url=f"http://{host}:{port}",
        auth=b64encode(credentials.encode()).decode(),
    )


@pytest_asyncio.fixture
async def broker(rabbitmq_url: str) -> AsyncIterator[RabbitBroker]:
    """Подключённый брокер; топологию объявляет сам тест."""
    connected = RabbitBroker(rabbitmq_url)
    await connected.connect()
    try:
        yield connected
    finally:
        await connected.stop()
