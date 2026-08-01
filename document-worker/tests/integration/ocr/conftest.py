"""Модели распознавания для интеграционных тестов.

Файлы не лежат в репозитории: бинарник в git невозможно прочитать в ревью.
Они скачиваются один раз в каталог кэша и сверяются по контрольной сумме — той
же процедурой, которой их укладывает в образ сборка.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from document_worker.infrastructure.cpu.executor import CpuPool
from document_worker.infrastructure.ocr.model_registry import (
    download_missing,
    model_dir_from_env,
    verify,
)
from document_worker.infrastructure.ocr.preprocessor import OpenCvImagePreprocessor
from document_worker.infrastructure.ocr.rapidocr_engine import RapidOcrEngine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Каталог с моделями; недостающие докачиваются один раз на прогон."""
    directory = model_dir_from_env()
    download_missing(directory)
    verify(directory)
    return directory


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def ocr_pool() -> AsyncIterator[CpuPool]:
    """Отдельный пул процессов: сессия ONNX поднимается один раз на процесс."""
    async with CpuPool(max_workers=1) as pool:
        yield pool


@pytest.fixture
def engine(model_dir: Path, ocr_pool: CpuPool) -> RapidOcrEngine:
    """Движок распознавания поверх локальных моделей."""
    return RapidOcrEngine(pool=ocr_pool, model_dir=model_dir)


@pytest.fixture
def preprocessor(ocr_pool: CpuPool) -> OpenCvImagePreprocessor:
    """Предобработка в том же пуле."""
    return OpenCvImagePreprocessor(pool=ocr_pool)
