"""Разбор вывода движка распознавания.

Модели здесь не нужны: проверяется перевод результата в координаты страницы,
порядок чтения и деградация до уверенности строки.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from document_worker.application.dto.ocr import (
    ConfidenceSource,
    PageImage,
    PageTransform,
    PreparedPage,
)
from document_worker.application.errors import OcrEngineError, PageOcrTimeoutError
from document_worker.infrastructure.ocr.rapidocr_engine import (
    RapidOcrEngine,
    _to_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

pytestmark = pytest.mark.unit

WIDTH = 1000
HEIGHT = 500


@dataclass(frozen=True, slots=True)
class FakeOutput:
    """То, что отдаёт RapidOCR: регионы, тексты, score и разбор на слова."""

    boxes: Sequence[Sequence[Sequence[float]]]
    txts: Sequence[str]
    scores: Sequence[float]
    word_results: Sequence[Sequence[tuple[str, float, Any]]] = ()


class FakePool:
    """Пул, который отдаёт заранее заданный исход."""

    def __init__(self, outcome: Callable[[], Any]) -> None:
        """Запоминает, чем закончится вызов."""
        self._outcome = outcome

    async def run_within(self, timeout_s: float, function: Any, /, *args: Any) -> Any:  # noqa: ARG002 — подменяет пул целиком
        return self._outcome()


def prepared_page() -> PreparedPage:
    """Страница без предобработки: координаты изображения совпадают со страницей."""
    return PreparedPage(
        image=PageImage(number=3, png=b"", width_px=WIDTH, height_px=HEIGHT, dpi=300),
        transform=PageTransform.identity(width_px=WIDTH, height_px=HEIGHT),
        applied=(),
    )


def quad(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_word_boxes_are_translated_into_page_coordinates() -> None:
    raw = FakeOutput(
        boxes=[quad(100, 50, 300, 100)],
        txts=["ДОГОВОР"],
        scores=[0.9],
        word_results=[[("ДОГОВОР", 0.9, quad(100, 50, 300, 100))]],
    )

    result = _to_result(raw, prepared_page(), elapsed_ms=5)

    word = result.words[0]
    assert word.confidence_source is ConfidenceSource.WORD
    assert word.bbox.x0 == pytest.approx(0.1)
    assert word.bbox.y1 == pytest.approx(0.2)


def test_region_without_word_boxes_falls_back_to_line_confidence() -> None:
    # Деградация обязана быть видна в данных: иначе оценка строки уйдёт
    # в неразборчивые диапазоны как оценка слова.
    raw = FakeOutput(
        boxes=[quad(0, 0, 800, 50), quad(0, 100, 400, 160)],
        txts=["ПОСТАВКА ТОВАРА", "СРОК"],
        scores=[0.7, 0.8],
    )

    result = _to_result(raw, prepared_page(), elapsed_ms=1)

    assert [word.text for word in result.words] == ["ПОСТАВКА", "ТОВАРА", "СРОК"]
    assert all(word.confidence_source is ConfidenceSource.LINE for word in result.words)
    # Бокс региона раздан словам пропорционально длине, а не поровну.
    assert result.words[0].bbox.x1 < result.words[1].bbox.x1
    assert result.line_count == 2
    assert result.median_line_height_px == pytest.approx(55.0)


def test_word_without_its_own_box_uses_the_region_box() -> None:
    raw = FakeOutput(
        boxes=[quad(0, 0, 200, 40)],
        txts=["АКТ"],
        scores=[0.6],
        word_results=[[("АКТ", 0.6, None)]],
    )

    result = _to_result(raw, prepared_page(), elapsed_ms=1)

    assert result.words[0].bbox.x1 == pytest.approx(0.2)


def test_regions_are_sorted_into_reading_order() -> None:
    raw = FakeOutput(
        boxes=[quad(0, 300, 200, 340), quad(0, 20, 200, 60)],
        txts=["НИЖНЯЯ", "ВЕРХНЯЯ"],
        scores=[0.9, 0.9],
    )

    result = _to_result(raw, prepared_page(), elapsed_ms=1)

    assert [word.text for word in result.words] == ["ВЕРХНЯЯ", "НИЖНЯЯ"]


def test_empty_output_yields_no_words() -> None:
    result = _to_result(
        FakeOutput(boxes=[], txts=[], scores=[]), prepared_page(), elapsed_ms=1
    )

    assert result.words == ()
    assert result.median_line_height_px == 0.0


def test_output_without_boxes_yields_no_words() -> None:
    raw = FakeOutput(boxes=None, txts=None, scores=None)  # type: ignore[arg-type]

    assert _to_result(raw, prepared_page(), elapsed_ms=1).words == ()


def test_blank_region_text_yields_no_words() -> None:
    raw = FakeOutput(boxes=[quad(0, 0, 10, 10)], txts=["   "], scores=[0.5])

    assert _to_result(raw, prepared_page(), elapsed_ms=1).words == ()


async def test_engine_failure_becomes_page_level_error(tmp_path: Path) -> None:
    # Сбой вывода не должен ронять документ: страница уходит в отказ со своей
    # причиной, обработка продолжается.
    def fail() -> Any:
        raise RuntimeError("onnxruntime failed")

    engine = RapidOcrEngine(pool=FakePool(fail), model_dir=tmp_path)  # type: ignore[arg-type]

    with pytest.raises(OcrEngineError):
        await engine.recognize(prepared_page(), languages=("ru",), timeout_s=1.0)


async def test_engine_timeout_becomes_page_level_error(tmp_path: Path) -> None:
    def expire() -> Any:
        raise TimeoutError

    engine = RapidOcrEngine(pool=FakePool(expire), model_dir=tmp_path)  # type: ignore[arg-type]

    with pytest.raises(PageOcrTimeoutError):
        await engine.recognize(prepared_page(), languages=("ru",), timeout_s=1.0)
