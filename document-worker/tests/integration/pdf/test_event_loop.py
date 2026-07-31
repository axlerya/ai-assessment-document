"""Синхронные вызовы PDF не останавливают цикл событий.

Разбор документа держит GIL десятки секунд. В корутине это останавливает всё
разом: heartbeat прогона, health-проверки и подтверждение сообщений брокеру —
воркер выглядит мёртвым, хотя работает.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from document_worker.infrastructure.pdf.pdfplumber_text_reader import (
    PdfPlumberDocumentReader,
)
from document_worker.infrastructure.pdf.pikepdf_inspector import PikePdfInspector
from document_worker.infrastructure.pdf.pypdfium2_page_renderer import (
    PdfiumPageRenderer,
)
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.infrastructure.cpu.executor import CpuPool

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TICK_S = 0.01
MAX_GAP_S = 0.5
PAGES = 12


async def _ticker(gaps: list[float], stop: asyncio.Event) -> None:
    """Тикает, пока его не остановят, и записывает паузы между тиками."""
    previous = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(TICK_S)
        now = time.perf_counter()
        gaps.append(now - previous)
        previous = now


async def _while_ticking(work: object) -> list[float]:
    gaps: list[float] = []
    stop = asyncio.Event()
    ticker = asyncio.create_task(_ticker(gaps, stop))
    try:
        await work  # type: ignore[misc]
    finally:
        stop.set()
        await ticker
    return gaps


async def test_pdf_operations_do_not_block_event_loop(
    cpu_pool: CpuPool,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=PAGES)
    inspector = PikePdfInspector(pool=cpu_pool, max_pages=300)
    reader = PdfPlumberDocumentReader(pool=cpu_pool)
    renderer = PdfiumPageRenderer(pool=cpu_pool, max_pixels=40_000_000)

    async def work() -> None:
        await inspector.inspect(path)
        async with reader.open(path) as handle:
            await handle.probe()
        async with renderer.session(path) as session:
            for number in range(1, PAGES + 1):
                await session.render(number, dpi=200)

    gaps = await _while_ticking(work())

    assert gaps, "фоновая корутина не получила ни одного тика"
    assert max(gaps) < MAX_GAP_S, f"цикл событий встал на {max(gaps):.2f} с"
