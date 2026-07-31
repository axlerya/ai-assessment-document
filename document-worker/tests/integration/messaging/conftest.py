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
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

RABBITMQ_IMAGE = "rabbitmq:4.2-management-alpine"
AMQP_PORT = 5672
MANAGEMENT_PORT = 15672
DEFAULT_VHOST = "/"
USER = "guest"
PASSWORD = "guest"  # noqa: S105 — учётные данные образа по умолчанию
READY_LOG = r"Server startup complete"


@dataclass(frozen=True, slots=True)
class Management:
    """Тонкий клиент management API: только то, чего не отдаёт AMQP."""

    base_url: str
    auth: str
    user: str

    def queue(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Описание очереди так, как его хранит брокер."""
        described: dict[str, Any] = self._request(
            "GET", f"queues/{_q(vhost)}/{_q(name)}"
        )
        return described

    def arguments_of(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Аргументы очереди."""
        return dict(self.queue(name, vhost=vhost).get("arguments", {}))

    def message_count(self, name: str, *, vhost: str = DEFAULT_VHOST) -> int:
        """Сколько сообщений лежит в очереди по данным статистики."""
        return int(self.queue(name, vhost=vhost).get("messages", 0))

    def fetch(
        self,
        name: str,
        *,
        count: int = 10,
        vhost: str = DEFAULT_VHOST,
    ) -> list[dict[str, Any]]:
        """Читает сообщения очереди, возвращая их обратно.

        Счётчик из статистики обновляется с задержкой, особенно у quorum-очередей,
        а чтение показывает содержимое очереди прямо сейчас.
        """
        messages = self._request(
            "POST",
            f"queues/{_q(vhost)}/{_q(name)}/get",
            body={
                "count": count,
                "ackmode": "ack_requeue_true",
                "encoding": "auto",
                "truncate": 50_000,
            },
        )
        return list(messages or [])

    def peek(self, name: str, *, vhost: str = DEFAULT_VHOST) -> dict[str, Any]:
        """Первое сообщение очереди, не снимая его оттуда."""
        return self.fetch(name, count=1, vhost=vhost)[0]

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
def rabbitmq_container() -> Iterator[DockerContainer]:
    """Поднимает RabbitMQ один раз на весь прогон.

    Готовность определяется по журналу, а не пробным подключением: пробник
    готового контейнера иногда попадает на порт управления и принимает ответ
    HTTP за несовместимую версию протокола.
    """
    container = (
        DockerContainer(RABBITMQ_IMAGE)
        .with_exposed_ports(AMQP_PORT, MANAGEMENT_PORT)
        .with_env("RABBITMQ_DEFAULT_USER", USER)
        .with_env("RABBITMQ_DEFAULT_PASS", PASSWORD)
        .waiting_for(LogMessageWaitStrategy(READY_LOG))
    )
    with container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_url(rabbitmq_container: DockerContainer) -> str:
    """Базовый URL подключения; vhost дописывает тот, кому он нужен."""
    host = rabbitmq_container.get_container_host_ip()
    port = rabbitmq_container.get_exposed_port(AMQP_PORT)
    return f"amqp://{USER}:{PASSWORD}@{host}:{port}/"


@pytest.fixture(scope="session")
def management(rabbitmq_container: DockerContainer) -> Management:
    """Доступ к management API того же контейнера."""
    host = rabbitmq_container.get_container_host_ip()
    port = rabbitmq_container.get_exposed_port(MANAGEMENT_PORT)
    return Management(
        base_url=f"http://{host}:{port}",
        auth=b64encode(f"{USER}:{PASSWORD}".encode()).decode(),
        user=USER,
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
