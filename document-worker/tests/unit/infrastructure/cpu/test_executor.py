"""Пул процессов под синхронные вычисления."""

from __future__ import annotations

import os

import pytest

from document_worker.infrastructure.cpu.executor import (
    SINGLE_THREAD_ENVIRONMENT,
    CpuPool,
)

pytestmark = pytest.mark.unit


def double(value: int) -> int:
    """Считается в рабочем процессе, поэтому объявлена на уровне модуля."""
    return value * 2


def thread_limits() -> dict[str, str]:
    """Возвращает ограничения потоков, выставленные в рабочем процессе."""
    return {name: os.environ.get(name, "") for name in SINGLE_THREAD_ENVIRONMENT}


async def test_pool_runs_the_function_in_a_worker() -> None:
    async with CpuPool() as pool:
        assert await pool.run(double, 21) == 42


async def test_worker_limits_native_thread_pools() -> None:
    # Нативные библиотеки разворачивают свои пулы, и на N процессов это даёт
    # N×M потоков на те же ядра.
    async with CpuPool() as pool:
        limits = await pool.run(thread_limits)

    assert limits == SINGLE_THREAD_ENVIRONMENT


async def test_keyword_arguments_are_rejected() -> None:
    # run_in_executor их не передаёт, и молча терять аргумент нельзя.
    async with CpuPool() as pool:
        with pytest.raises(TypeError, match="по имени"):
            await pool.run(double, value=21)


async def test_pool_outside_its_context_is_an_error() -> None:
    pool = CpuPool()

    with pytest.raises(RuntimeError, match="вне своего контекста"):
        await pool.run(double, 21)


async def test_pool_is_closed_after_the_block() -> None:
    pool = CpuPool()
    async with pool:
        await pool.run(double, 1)

    with pytest.raises(RuntimeError, match="вне своего контекста"):
        await pool.run(double, 1)
