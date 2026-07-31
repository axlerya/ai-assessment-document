"""Живой RabbitMQ для тестов топологии и доставки.

Аргументы очередей проверяются только настоящим брокером: in-memory брокер
FastStream их не хранит, а AMQP их не отдаёт — passive-declare возвращает лишь
имя и счётчики. Поэтому рядом поднимается management API.

Тестам доставки нужен ещё и свой vhost: очереди в них общие по именам, а
подсчёт сообщений в соседнем тесте иначе становится случайным.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
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
    user: str

    def queue(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Описание очереди так, как его хранит брокер."""
        return self._request("GET", f"queues/{_q(vhost)}/{_q(name)}")

    def arguments_of(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Аргументы очереди."""
        return dict(self.queue(name, vhost=vhost).get("arguments", {}))

    def message_count(self, name: str, *, vhost: str = DEFAULT_VHOST) -> int:
        """Сколько сообщений лежит в очереди."""
        return int(self.queue(name, vhost=vhost).get("messages", 0))

    def peek(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Первое сообщение очереди, не снимая его оттуда."""
        messages = self._request(
            "POST",
            f"queues/{_q(vhost)}/{_q(name)}/get",
            body={
                "count": 1,
                "ackmode": "ack_requeue_true",
                "encoding": "auto",
                "truncate": 50_000,
            },
        )
        return dict(messages[0])  # type: ignore[index]

    def create_vhost(self, name: str) -> None:
        """Создаёт vhost и даёт на него права пользователю теста."""
        self._request("PUT", f"vhosts/{_q(name)}", body={})
        self._request(
            "PUT",
            f"permissions/{_q(name)}/{_q(self.user)}",
            body={"configure": ".*", "write": ".*", "read": ".*"},
        )

    def delete_vhost(self, name: str) -> None:
        """Удаляет vhost вместе со всем содержимым."""
        self._request("DELETE", f"vhosts/{_q(name)}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> Any:
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(  # noqa: S310 — адрес контейнера теста
            f"{self.base_url}/api/{path}",
            method=method,
            data=payload,
            headers={
                "Authorization": f"Basic {self.auth}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            raw = response.read()
        return json.loads(raw) if raw else None


def _q(value: str) -> str:
    return quote(value, safe="")


@pytest.fixture(scope="session")
def rabbitmq_container() -> Iterator[RabbitMqContainer]:
    """Поднимает RabbitMQ один раз на весь прогон."""
    container = RabbitMqContainer(RABBITMQ_IMAGE)
    container.with_exposed_ports(MANAGEMENT_PORT)
    with container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_url(rabbitmq_container: RabbitMqContainer) -> str:
    """Базовый URL подключения; vhost дописывает тот, кому он нужен."""
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
    user = params.credentials.username
    credentials = f"{user}:{params.credentials.password}"
    return Management(
        base_url=f"http://{host}:{port}",
        auth=b64encode(credentials.encode()).decode(),
        user=user,
    )


@pytest.fixture
def isolated_vhost(management: Management) -> Iterator[str]:
    """Свой vhost на тест: очереди у тестов доставки одноимённые."""
    name = f"test-{uuid.uuid4().hex[:12]}"
    management.create_vhost(name)
    try:
        yield name
    finally:
        management.delete_vhost(name)


@pytest_asyncio.fixture
async def broker(rabbitmq_url: str) -> AsyncIterator[RabbitBroker]:
    """Подключённый брокер; топологию объявляет сам тест."""
    connected = RabbitBroker(rabbitmq_url)
    await connected.connect()
    try:
        yield connected
    finally:
        await connected.stop()
