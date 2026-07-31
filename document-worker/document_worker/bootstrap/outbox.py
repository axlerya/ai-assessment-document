"""Фоновая публикация накопленных событий.

Реле встроено в воркер, а не вынесено отдельным процессом: событий немного,
а отдельный процесс потребовал бы своей конфигурации, своего развёртывания и
своего мониторинга ради одного цикла в полсекунды.

Цикл не падает от недоступности брокера: пауза растёт вдвое до потолка, и
события просто ждут. Отметить их опубликованными при недоступном брокере
значило бы потерять их навсегда.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from document_worker.application.config import OutboxConfig
    from document_worker.application.use_cases.publish_outbox_events import (
        PublishOutboxEvents,
    )


@dataclass(frozen=True, slots=True)
class OutboxRelay:
    """Крутит публикацию накопленных событий, пока его не остановят."""

    publish: PublishOutboxEvents
    config: OutboxConfig

    async def run(self) -> None:
        """Публикует события до отмены задачи."""
        idle = self.config.poll_interval_s
        while True:
            try:
                result = await self.publish.execute()
            except Exception:  # noqa: BLE001 — цикл переживает любую поломку
                # Упавшее реле молча перестаёт публиковать, и события копятся
                # до перезапуска процесса.
                idle = min(idle * 2, self.config.backoff_cap_s)
            else:
                idle = (
                    self.config.poll_interval_s
                    if result.published
                    else min(idle * 2, self.config.backoff_cap_s)
                    if result.failed
                    else self.config.poll_interval_s
                )
            await asyncio.sleep(idle)


@contextlib.asynccontextmanager
async def running(relay: OutboxRelay) -> AsyncIterator[asyncio.Task[None]]:
    """Держит реле запущенным на время блока и снимает его при выходе."""
    task = asyncio.create_task(relay.run(), name="outbox-relay")
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
