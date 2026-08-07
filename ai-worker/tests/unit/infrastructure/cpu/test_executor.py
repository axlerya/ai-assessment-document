"""Пул процессов под синхронный инференс (ADR-0016)."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from ai_worker.infrastructure.cpu.executor import SINGLE_THREAD_ENVIRONMENT, CpuPool

pytestmark = pytest.mark.unit

# Тик фоновой корутины и допустимый пропуск: ADR-0016 разрешает не более двух
# пропущенных тиков подряд, то есть разрыв в три интервала.
TICK_S = 0.05
MAX_GAP_S = TICK_S * 3


def double(value: int) -> int:
    """Считается в рабочем процессе, поэтому объявлена на уровне модуля."""
    return value * 2


def thread_limits() -> dict[str, str]:
    """Ограничения потоков, выставленные в рабочем процессе."""
    return {name: os.environ.get(name, "") for name in SINGLE_THREAD_ENVIRONMENT}


def burn(seconds: float) -> int:
    """Держит процессор столько же, сколько прогон модели."""
    deadline = time.perf_counter() + seconds
    spins = 0
    while time.perf_counter() < deadline:
        spins += 1
    return spins


def hang() -> None:
    """Не возвращается: так выглядит зависший прогон модели."""
    while True:
        time.sleep(0.1)


async def _measure_gaps(gaps: list[float], stop: asyncio.Event) -> None:
    previous = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(TICK_S)
        now = time.perf_counter()
        gaps.append(now - previous)
        previous = now


async def test_pool_runs_the_function_in_a_worker() -> None:
    async with CpuPool() as pool:
        assert await pool.run(double, 21) == 42


async def test_worker_limits_native_thread_pools() -> None:
    # На N процессов нативные пулы дают N×M потоков на те же ядра, и латентность
    # инференса перестаёт быть предсказуемой.
    async with CpuPool() as pool:
        limits = await pool.run(thread_limits)

    assert limits == SINGLE_THREAD_ENVIRONMENT


async def test_inference_does_not_block_event_loop() -> None:
    # Секунда синхронного счёта прямо в корутине глушит heartbeat брокера и
    # health-эндпоинт; в пуле процессов она не задевает цикл событий.
    gaps: list[float] = []
    stop = asyncio.Event()
    ticker = asyncio.create_task(_measure_gaps(gaps, stop))
    try:
        async with CpuPool() as pool:
            await pool.run(burn, 1.0)
    finally:
        stop.set()
        await ticker

    assert gaps, "фоновая корутина не тикнула ни разу"
    assert max(gaps) < MAX_GAP_S, f"цикл событий встал на {max(gaps):.3f} с"


async def test_call_is_bounded_by_its_timeout() -> None:
    async with CpuPool() as pool:
        with pytest.raises(TimeoutError):
            await pool.run_within(0.2, hang)


async def test_pool_survives_a_hung_worker() -> None:
    # Пул не умеет отменять начатую задачу: без перезапуска зависший процесс
    # остался бы занят навсегда, и следующее сообщение ждало бы вечно.
    async with CpuPool() as pool:
        with pytest.raises(TimeoutError):
            await pool.run_within(0.2, hang)

        assert await pool.run_within(10.0, double, 21) == 42


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
