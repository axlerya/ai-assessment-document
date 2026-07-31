"""Общий пул процессов для PDF-тестов.

Контекст `spawn` поднимает интерпретатор заново, поэтому пул один на прогон:
иначе каждый тест платил бы за старт процесса больше, чем за сам разбор.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest_asyncio

from document_worker.infrastructure.cpu.executor import CpuPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def cpu_pool() -> AsyncIterator[CpuPool]:
    """Пул процессов под синхронные вызовы PDF-библиотек."""
    async with CpuPool(max_workers=2) as pool:
        yield pool
