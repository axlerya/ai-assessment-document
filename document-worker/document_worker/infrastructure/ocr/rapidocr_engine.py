"""Адаптер RapidOCR.

Сессия ONNX поднимается один раз на рабочий процесс и живёт в модульной
переменной: её создание стоит секунду-две, а страниц через процесс проходят
сотни. Модели указываются явными путями, автозагрузка отключена — сеть в
рантайме запрещена.

Порог `text_score` опущен до нуля намеренно. По умолчанию RapidOCR молча
выбрасывает результаты со score ниже порога, а для нас низкоуверенное чтение
обязано дойти до политики и стать неразборчивым диапазоном, а не исчезнуть.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from document_worker.application.dto.ocr import (
    ConfidenceSource,
    OcrResult,
    RecognizedWordDTO,
)
from document_worker.application.errors import OcrEngineError, PageOcrTimeoutError
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.infrastructure.ocr.model_registry import resolve

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from document_worker.application.dto.ocr import PreparedPage
    from document_worker.infrastructure.cpu.executor import CpuPool

# Пороги детектора и режим вывода. Значения по умолчанию у RapidOCR рассчитаны
# на «сценовые» снимки: 960 px по длинной стороне для A4@300 DPI дают даунскейл
# втрое, и детектор теряет мелкий текст договора.
ENGINE_PARAMS: Final[Mapping[str, object]] = {
    "Global.text_score": 0.0,
    "Global.return_word_box": True,
    "Global.use_cls": True,
    "Global.log_level": "error",
    "Det.limit_type": "max",
    "Det.limit_side_len": 1536,
    "Det.box_thresh": 0.5,
    # Параллелим по страницам, а не внутри страницы: пул процессов уже занял
    # все ядра, и вложенный параллелизм только конкурирует с ним за них.
    "EngineConfig.onnxruntime.intra_op_num_threads": 1,
}

_ENGINES: dict[str, Any] = {}


def recognize_page(png: bytes, model_dir: str) -> tuple[Any, float]:
    """Распознаёт изображение. Выполняется в рабочем процессе."""
    engine = _engine_for(model_dir)
    started = time.monotonic()
    result = engine(png)
    return result, time.monotonic() - started


def _engine_for(model_dir: str) -> Any:
    engine = _ENGINES.get(model_dir)
    if engine is None:
        # Импорт стоит секунды и нужен только в рабочем процессе.
        from pathlib import Path  # noqa: PLC0415

        from rapidocr import RapidOCR  # noqa: PLC0415

        paths = resolve(Path(model_dir))
        engine = RapidOCR(
            params={
                **ENGINE_PARAMS,
                "Det.model_path": str(paths["det"]),
                "Cls.model_path": str(paths["cls"]),
                "Rec.model_path": str(paths["rec"]),
            }
        )
        _ENGINES[model_dir] = engine
    return engine


@dataclass(frozen=True, slots=True)
class RapidOcrEngine:
    """Распознавание страницы поверх RapidOCR и локальных моделей."""

    pool: CpuPool
    model_dir: Path

    async def recognize(
        self,
        page: PreparedPage,
        *,
        languages: Sequence[str],  # noqa: ARG002 — восточнославянская модель покрывает и кириллицу, и латиницу
        timeout_s: float,
        options: Mapping[str, str] | None = None,  # noqa: ARG002 — профиль применён предобработкой
    ) -> OcrResult:
        """Распознаёт страницу, не блокируя цикл событий.

        Raises:
            PageOcrTimeoutError: Страница не уложилась в отведённое время.
            OcrEngineError: Движок не смог выполнить вывод.
        """
        try:
            raw, elapsed = await self.pool.run_within(
                timeout_s,
                recognize_page,
                page.image.png,
                str(self.model_dir),
            )
        except TimeoutError as error:
            raise PageOcrTimeoutError(
                "распознавание страницы не уложилось в таймаут",
                page_number=page.image.number,
                context={"timeout_s": timeout_s},
            ) from error
        except (RuntimeError, ValueError, OSError) as error:
            raise OcrEngineError(
                "движок распознавания не выполнил вывод",
                page_number=page.image.number,
                context={"reason": type(error).__name__},
            ) from error
        return _to_result(raw, page, elapsed_ms=int(elapsed * 1000))


def _to_result(raw: Any, page: PreparedPage, *, elapsed_ms: int) -> OcrResult:
    """Переводит вывод движка в нормализованные координаты страницы."""
    regions = _regions(raw)
    words: list[RecognizedWordDTO] = []
    heights: list[float] = []
    for line_index, region in enumerate(regions):
        heights.append(region.height_px)
        words.extend(_words_of(region, page, line_index=line_index))
    return OcrResult(
        words=tuple(words),
        line_count=len(regions),
        median_line_height_px=_median(heights),
        engine_version=_engine_version(),
        elapsed_ms=elapsed_ms,
    )


@dataclass(frozen=True, slots=True)
class _Region:
    """Текстовый регион движка вместе с разбором на слова."""

    text: str
    score: float
    quad: tuple[tuple[float, float], ...]
    words: tuple[tuple[str, float, tuple[tuple[float, float], ...] | None], ...]

    @property
    def height_px(self) -> float:
        """Высота региона в пикселях подготовленного изображения."""
        ys = [y for _, y in self.quad]
        return max(ys) - min(ys)


def _regions(raw: Any) -> tuple[_Region, ...]:
    boxes = getattr(raw, "boxes", None)
    texts = getattr(raw, "txts", None)
    scores = getattr(raw, "scores", None)
    if boxes is None or texts is None or scores is None:
        return ()
    word_results = getattr(raw, "word_results", ()) or ()
    regions = [
        _Region(
            text=text,
            score=float(score),
            quad=tuple((float(x), float(y)) for x, y in box),
            words=tuple(
                (str(item[0]), float(item[1]), _quad_of(item[2]))
                for item in (word_results[index] if index < len(word_results) else ())
            ),
        )
        for index, (box, text, score) in enumerate(
            zip(boxes, texts, scores, strict=True)
        )
    ]
    # Порядок чтения: сверху вниз, внутри строки — слева направо. Движок
    # отдаёт регионы в порядке детекции, который к чтению отношения не имеет.
    regions.sort(key=lambda region: (min(y for _, y in region.quad), region.quad[0][0]))
    return tuple(regions)


def _quad_of(box: Any) -> tuple[tuple[float, float], ...] | None:
    if box is None:
        return None
    return tuple((float(point[0]), float(point[1])) for point in box)


def _words_of(
    region: _Region,
    page: PreparedPage,
    *,
    line_index: int,
) -> list[RecognizedWordDTO]:
    if region.words:
        return [
            RecognizedWordDTO(
                text=text,
                confidence=score,
                bbox=_bbox(quad or region.quad, page),
                line_index=line_index,
                word_index=word_index,
                confidence_source=ConfidenceSource.WORD,
            )
            for word_index, (text, score, quad) in enumerate(region.words)
        ]
    # Пословных боксов не пришло: строка делится по пробелам, бокс региона
    # раздаётся словам пропорционально длине. Деградация явная и видна в данных.
    return list(_split_region(region, page, line_index=line_index))


def _split_region(
    region: _Region,
    page: PreparedPage,
    *,
    line_index: int,
) -> list[RecognizedWordDTO]:
    tokens = region.text.split()
    if not tokens:
        return []
    xs = [x for x, _ in region.quad]
    ys = [y for _, y in region.quad]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    total = sum(len(token) for token in tokens)
    words: list[RecognizedWordDTO] = []
    cursor = 0
    for word_index, token in enumerate(tokens):
        start = left + (right - left) * cursor / total
        cursor += len(token)
        end = left + (right - left) * cursor / total
        words.append(
            RecognizedWordDTO(
                text=token,
                confidence=region.score,
                bbox=_bbox(_box(start, end, top, bottom), page),
                line_index=line_index,
                word_index=word_index,
                confidence_source=ConfidenceSource.LINE,
            )
        )
    return words


def _box(
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> tuple[tuple[float, float], ...]:
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def _bbox(quad: Sequence[tuple[float, float]], page: PreparedPage) -> BoundingBox:
    """Переводит четырёхугольник в осевой прямоугольник страницы."""
    corners = [page.transform.to_page(x, y) for x, y in quad]
    xs = [min(max(x, 0.0), 1.0) for x, _ in corners]
    ys = [min(max(y, 0.0), 1.0) for _, y in corners]
    left, top = min(xs), min(ys)
    return BoundingBox(
        x0=left,
        y0=top,
        x1=max(*xs, left + _MIN_SIDE),
        y1=max(*ys, top + _MIN_SIDE),
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _engine_version() -> str:
    """Версия движка: она попадает в логи и объясняет смену качества."""
    from importlib.metadata import version  # noqa: PLC0415 — нужен только тут

    return f"rapidocr-{version('rapidocr')}"


# Вырожденный прямоугольник домен отвергает, а движок иногда отдаёт бокс
# нулевой высоты на одиночном символе.
_MIN_SIDE: Final[float] = 1e-6
