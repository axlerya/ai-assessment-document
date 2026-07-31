"""Пул процессов для синхронных библиотек.

pikepdf, pdfplumber и pypdfium2 синхронны и держат GIL десятки секунд на
документ. В корутине это останавливает всё: heartbeat прогона, health-проверки
и подтверждение сообщений брокеру.

Контекст `spawn` выбран намеренно: `fork` вместе с потоками asyncio и
нативными библиотеками даёт зависший дочерний процесс. Плата за это —
переимпорт модулей в каждом рабочем процессе, поэтому импорты верхнего уровня
должны оставаться дешёвыми.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, ParamSpec, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

P = ParamSpec("P")
R = TypeVar("R")

# Нативные библиотеки по умолчанию разворачивают собственные пулы потоков, и
# на N процессов это даёт N×M потоков на те же ядра.
SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def _limit_threads() -> None:
    os.environ.update(SINGLE_THREAD_ENVIRONMENT)


class CpuPool:
    """Отдельный пул процессов под синхронные вычисления."""

    def __init__(self, *, max_workers: int = 1) -> None:
        """Готовит пул; процессы поднимаются при входе в контекст."""
        self._max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None

    async def __aenter__(self) -> Self:
        """Поднимает пул."""
        self._executor = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_limit_threads,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Останавливает пул, дожидаясь незавершённых задач."""
        executor = self._require_executor()
        self._executor = None
        await asyncio.to_thread(executor.shutdown, wait=True)

    async def run(
        self,
        function: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Выполняет функцию в рабочем процессе.

        Функция и аргументы обязаны быть picklable: путь к файлу и числа —
        да, открытый документ — нет.
        """
        if kwargs:
            msg = "аргументы по имени в пул процессов не передаются"
            raise TypeError(msg)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._require_executor(), function, *args)

    def _require_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            msg = "пул процессов используется вне своего контекста"
            raise RuntimeError(msg)
        return self._executor
