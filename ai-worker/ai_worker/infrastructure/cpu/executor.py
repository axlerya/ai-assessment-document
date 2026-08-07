"""Пул процессов для синхронного инференса (ADR-0016).

`transformers` и токенизаторы синхронны и держат GIL секундами. В корутине это
останавливает всё: heartbeat брокера, подтверждение сообщений и health-проверку,
а заявленный таймаут обработки становится неисполнимым.

Контекст `spawn` выбран намеренно: `fork` вместе с потоками asyncio и нативными
библиотеками даёт зависший дочерний процесс. Плата — переимпорт модулей в каждом
рабочем процессе, поэтому импорты верхнего уровня обязаны оставаться дешёвыми, а
модель грузится один раз на процесс и живёт в его памяти.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Final, ParamSpec, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

P = ParamSpec("P")
R = TypeVar("R")

# torch и onnxruntime по умолчанию разворачивают собственные пулы потоков, и на
# N процессов это даёт N×M потоков на те же ядра: латентность инференса
# перестаёт быть предсказуемой, а таймаут — осмысленным.
SINGLE_THREAD_ENVIRONMENT: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
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

        Функция и аргументы обязаны быть picklable: путь к каталогу модели и
        строки — да, загруженная модель — нет.
        """
        self._reject_keywords(kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._require_executor(), function, *args)

    async def run_within(
        self,
        timeout_s: float,
        function: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """То же, но с ограничением по времени.

        Raises:
            TimeoutError: Задача не уложилась в отведённое время.
        """
        self._reject_keywords(kwargs)
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._require_executor(), function, *args)
        try:
            async with asyncio.timeout(timeout_s):
                return await future
        except TimeoutError:
            # Одно ожидание таймаута не реализует: пул не умеет отменять уже
            # начатую задачу, и зависший процесс остался бы занят навсегда.
            # Поэтому он убивается, а пул поднимается заново — вместе с
            # повторной загрузкой модели, что дешевле застрявшей очереди.
            await self._restart()
            raise

    @staticmethod
    def _reject_keywords(kwargs: dict[str, object]) -> None:
        if kwargs:
            msg = "аргументы по имени в пул процессов не передаются"
            raise TypeError(msg)

    async def _restart(self) -> None:
        executor = self._require_executor()
        self._executor = None
        for process in tuple(executor._processes.values()):  # noqa: SLF001 — публичного доступа к рабочим процессам у пула нет
            process.kill()
        await asyncio.to_thread(executor.shutdown, wait=False, cancel_futures=True)
        await self.__aenter__()

    def _require_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            msg = "пул процессов используется вне своего контекста"
            raise RuntimeError(msg)
        return self._executor
