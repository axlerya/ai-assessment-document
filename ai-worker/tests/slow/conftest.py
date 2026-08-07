"""Настоящая модель для медленных тестов.

Веса весят порядка двух гигабайт, поэтому они не качаются на каждом прогоне.
Если каталога модели нет, тесты пропускаются с указанием, чем его наполнить:
молчаливый пропуск выглядел бы как пройденная проверка.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from ai_worker.application.errors import EmbeddingModelMissing
from ai_worker.infrastructure.cpu.executor import CpuPool
from ai_worker.infrastructure.embedding.model_registry import (
    model_dir_from_env,
    verify,
)
from ai_worker.infrastructure.embedding.runtime import PooledEncoderRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Каталог с весами bge-m3, проверенный по контрольным суммам."""
    directory = model_dir_from_env()
    try:
        verify(directory)
    except EmbeddingModelMissing as error:
        pytest.skip(f"нет модели в {directory}: {error.message}")
    return directory


@pytest_asyncio.fixture(loop_scope="session")
async def runtime(model_dir: Path) -> AsyncIterator[PooledEncoderRuntime]:
    """Прогон модели в отдельном процессе, как в бою."""
    async with CpuPool() as pool:
        prepared = PooledEncoderRuntime(
            pool=pool,
            model_dir=model_dir,
            max_input_tokens=1024,
        )
        await prepared.prewarm(timeout_s=600.0)
        yield prepared
