"""Фоновый цикл публикации: живучесть и остановка."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from document_worker.application.config import OutboxConfig
from document_worker.application.dto.results import PublishOutboxEventsResult
from document_worker.bootstrap.outbox import OutboxRelay, running

pytestmark = pytest.mark.integration

FAST = OutboxConfig(poll_interval_s=0.01, backoff_base_s=0.01, backoff_cap_s=0.05)


@dataclass
class SpyPublisher:
    """Публикатор, чей исход задаёт тест."""

    results: list[PublishOutboxEventsResult] = field(default_factory=list)
    error: Exception | None = None
    calls: int = 0

    async def execute(self) -> PublishOutboxEventsResult:
        """Возвращает заданный исход или поднимает заданную ошибку."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.results:
            return self.results.pop(0)
        return PublishOutboxEventsResult(fetched=0, published=0, failed=0)


async def _wait_for_calls(publisher: SpyPublisher, expected: int) -> bool:
    for _ in range(100):
        if publisher.calls >= expected:
            return True
        await asyncio.sleep(0.01)
    return False


async def test_relay_keeps_publishing_until_stopped() -> None:
    publisher = SpyPublisher()
    relay = OutboxRelay(publish=publisher, config=FAST)  # type: ignore[arg-type]

    async with running(relay):
        assert await _wait_for_calls(publisher, 3)


async def test_relay_survives_a_broken_publication() -> None:
    # Упавшее реле молча перестаёт публиковать, и события копятся до
    # перезапуска процесса.
    publisher = SpyPublisher(error=RuntimeError("база недоступна"))
    relay = OutboxRelay(publish=publisher, config=FAST)  # type: ignore[arg-type]

    async with running(relay):
        assert await _wait_for_calls(publisher, 3)


async def test_relay_stops_with_its_block() -> None:
    # Незавершённая фоновая задача держит процесс и не даёт остановиться.
    publisher = SpyPublisher()
    relay = OutboxRelay(publish=publisher, config=FAST)  # type: ignore[arg-type]

    async with running(relay) as task:
        await _wait_for_calls(publisher, 1)

    assert task.done()
