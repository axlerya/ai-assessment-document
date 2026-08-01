"""Движок распознавания на настоящих моделях."""

from __future__ import annotations

import asyncio
import io
import time
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from document_worker.application.dto.ocr import (
    ConfidenceSource,
    PageImage,
    PreprocessProfile,
)
from document_worker.application.errors import PageOcrTimeoutError
from document_worker.infrastructure.ocr.rapidocr_engine import ENGINE_PARAMS
from tests.fakes.page_images import DEFAULT_LINES, blank_page_png, make_page_png

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from document_worker.application.dto.ocr import OcrResult, RecognizedWordDTO
    from document_worker.infrastructure.ocr.preprocessor import OpenCvImagePreprocessor
    from document_worker.infrastructure.ocr.rapidocr_engine import RapidOcrEngine

pytestmark = pytest.mark.integration

LANGUAGES = ("ru", "en")
TIMEOUT_S = 60.0
MIN_CLEAN_SCAN_CONFIDENCE = 0.8
TICK_S = 0.05
MAX_MISSED_TICKS = 2


def page_image(png: bytes) -> PageImage:
    with Image.open(io.BytesIO(png)) as image:
        return PageImage(
            number=1,
            png=png,
            width_px=image.width,
            height_px=image.height,
            dpi=150,
        )


async def recognize(
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
    png: bytes,
    *,
    timeout_s: float = TIMEOUT_S,
) -> OcrResult:
    prepared = await preprocessor.prepare(
        page_image(png), profile=PreprocessProfile.DEFAULT
    )
    return await engine.recognize(prepared, languages=LANGUAGES, timeout_s=timeout_s)


async def test_recognizes_clean_scan_with_expected_confidence(
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
) -> None:
    # Уверенность проверяется взвешенной по длине слова — той величиной, в
    # которой её считает сервис. Отдельное короткое слово получает от движка
    # заведомо шумный score даже когда прочитано верно.
    result = await recognize(engine, preprocessor, make_page_png())

    recognized = " ".join(word.text for word in result.words)
    assert recognized == " ".join(DEFAULT_LINES).replace("  ", " ")
    assert _weighted_confidence(result.words) > MIN_CLEAN_SCAN_CONFIDENCE


def _weighted_confidence(words: Sequence[RecognizedWordDTO]) -> float:
    weight = sum(len(word.text) for word in words)
    return sum(word.confidence * len(word.text) for word in words) / weight


async def test_returns_per_word_confidence_and_boxes(
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
) -> None:
    # На пословных боксах и уверенности построены неразборчивые диапазоны:
    # без них оператору нечего показать поверх скана.
    result = await recognize(engine, preprocessor, make_page_png())

    assert result.words
    for word in result.words:
        assert 0.0 <= word.confidence <= 1.0
        assert 0.0 <= word.bbox.x0 < word.bbox.x1 <= 1.0
        assert 0.0 <= word.bbox.y0 < word.bbox.y1 <= 1.0
        assert word.confidence_source is ConfidenceSource.WORD


async def test_words_follow_reading_order(
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
) -> None:
    result = await recognize(engine, preprocessor, make_page_png())

    assert [word.line_index for word in result.words] == sorted(
        word.line_index for word in result.words
    )
    assert result.line_count >= 1


async def test_empty_image_returns_zero_words_without_exception(
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
) -> None:
    # Пустой результат — валидный результат: дальше его разбирает политика
    # читаемости, а не обработчик исключений.
    result = await recognize(engine, preprocessor, blank_page_png())

    assert result.words == ()
    assert result.line_count == 0


async def test_engine_keeps_low_score_words() -> None:
    # По умолчанию RapidOCR молча выбрасывает результаты ниже порога, а для нас
    # низкоуверенное чтение обязано дойти до политики и стать диапазоном.
    assert ENGINE_PARAMS["Global.text_score"] == 0.0


async def test_uses_only_the_given_model_directory(
    model_dir: Path,
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
) -> None:
    assert engine.model_dir == model_dir

    result = await recognize(engine, preprocessor, make_page_png())

    assert result.engine_version


async def test_timeout_is_enforced_and_raises_page_level_error(
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
) -> None:
    # Ожидание без убийства процесса таймаута не реализует: воркер остался бы
    # занят навсегда, а страница висела бы до таймаута документа.
    prepared = await preprocessor.prepare(
        page_image(make_page_png()), profile=PreprocessProfile.DEFAULT
    )

    with pytest.raises(PageOcrTimeoutError):
        await engine.recognize(prepared, languages=LANGUAGES, timeout_s=0.001)


async def test_engine_call_does_not_block_event_loop(
    engine: RapidOcrEngine,
    preprocessor: OpenCvImagePreprocessor,
) -> None:
    # Синхронный вызов из корутины останавливает heartbeat брокера, и
    # необработанное сообщение уходит на повтор по кругу.
    gaps: list[float] = []

    async def tick() -> None:
        previous = time.monotonic()
        while True:
            await asyncio.sleep(TICK_S)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now

    ticker = asyncio.create_task(tick())
    try:
        await recognize(engine, preprocessor, make_page_png())
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)

    assert gaps
    assert max(gaps) < TICK_S * (MAX_MISSED_TICKS + 1)
