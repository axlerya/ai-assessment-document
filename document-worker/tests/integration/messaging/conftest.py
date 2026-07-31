"""Живой RabbitMQ для тестов топологии.

Аргументы очередей проверяются только настоящим брокером: in-memory брокер
FastStream их не хранит и о расхождении не скажет.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from faststream.rabbit import RabbitBroker
from testcontainers.community.rabbitmq import RabbitMqContainer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

RABBITMQ_IMAGE = "rabbitmq:4.2-alpine"


@pytest.fixture(scope="session")
def rabbitmq_container() -> Iterator[RabbitMqContainer]:
    """Поднимает RabbitMQ один раз на весь прогон."""
    with RabbitMqContainer(RABBITMQ_IMAGE) as container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_url(rabbitmq_container: RabbitMqContainer) -> str:
    """URL подключения к контейнеру."""
    params = rabbitmq_container.get_connection_params()
    return (
        f"amqp://{params.credentials.username}:{params.credentials.password}"
        f"@{params.host}:{params.port}/"
    )


@pytest_asyncio.fixture
async def broker(rabbitmq_url: str) -> AsyncIterator[RabbitBroker]:
    """Подключённый брокер; топологию объявляет сам тест."""
    connected = RabbitBroker(rabbitmq_url)
    await connected.connect()
    try:
        yield connected
    finally:
        await connected.close()
