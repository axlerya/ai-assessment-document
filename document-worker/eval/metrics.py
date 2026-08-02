"""Метрики качества обработки документа.

Считаются по тому же тексту, который сервис отдаёт потребителю, и по тому же
маркеру неразборчивости, что определён в домене: своя копия формата разошлась
бы с ним молча, и метрика перестала бы измерять работающий сервис.

Регистр не понижается. В юридическом тексте «Стороны» и «стороны» — разные
вещи, и слепая к регистру метрика прятала бы настоящую ошибку распознавания.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import TYPE_CHECKING

from rapidfuzz.distance import Levenshtein

from document_worker.domain.markers import MARKER_RE

if TYPE_CHECKING:
    from collections.abc import Sequence

Span = tuple[int, int]

SOFT_HYPHEN = "­"
BOUNDARY_TOLERANCE = 20

_WHITESPACE_RE = re.compile(r"\s+")
_QUOTES = str.maketrans(dict.fromkeys("«»“”„‟", '"') | dict.fromkeys("‘’‚‛", "'"))
_DASHES = str.maketrans(dict.fromkeys("—–‒−", "-"))


def normalize_for_scoring(text: str) -> str:
    """Приводит текст к виду, в котором его сравнивают с эталоном.

    Маркеры снимаются до сравнения: маркер — представление, а не текст, и его
    символы, попав в расстояние, наказывали бы сервис за честную пометку
    сильнее, чем за молчание.
    """
    stripped = MARKER_RE.sub("", text)
    folded = unicodedata.normalize("NFC", stripped).replace(SOFT_HYPHEN, "")
    unified = folded.translate(_QUOTES).translate(_DASHES)
    return _WHITESPACE_RE.sub(" ", unified).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Доля символов эталона, которые пришлось бы исправить."""
    expected = normalize_for_scoring(reference)
    actual = normalize_for_scoring(hypothesis)
    if not expected:
        # Делить не на что. Пустота против пустоты — точное совпадение, а
        # текст на месте пустоты — выдумка, и она обязана быть видна.
        return 0.0 if not actual else 1.0
    return Levenshtein.distance(expected, actual) / len(expected)


def wer(reference: str, hypothesis: str) -> float:
    """То же, что `cer`, но единицей сравнения служит слово."""
    expected = normalize_for_scoring(reference).split()
    actual = normalize_for_scoring(hypothesis).split()
    if not expected:
        return 0.0 if not actual else 1.0
    return Levenshtein.distance(expected, actual) / len(expected)


def span_iou(expected: Sequence[Span], actual: Sequence[Span]) -> float:
    """Символьное пересечение эталонных и фактических диапазонов к их union.

    Считается по символам, а не по диапазонам: один эталонный фрагмент сервис
    вправе пометить двумя, и наоборот — совпадение границ здесь не требуется.
    """
    left = _characters(expected)
    right = _characters(actual)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def boundary_f1(
    expected: Sequence[int],
    actual: Sequence[int],
    tolerance: int = BOUNDARY_TOLERANCE,
) -> float:
    """F1 совпадения границ с допуском в символах.

    Каждая эталонная граница закрывается не более чем одной фактической:
    иначе дробление документа на мелкие куски улучшало бы метрику.
    """
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    matched = _matched_within(expected, actual, tolerance)
    if matched == 0:
        return 0.0
    precision = matched / len(actual)
    recall = matched / len(expected)
    return 2 * precision * recall / (precision + recall)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ранговая корреляция двух рядов.

    Ранговая, а не линейная: от уверенности требуется предсказывать порядок
    качества, а не совпадать с ним по шкале.
    """
    if len(xs) != len(ys) or len(xs) < 2:  # noqa: PLR2004 — по одной точке корреляции нет
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def _characters(spans: Sequence[Span]) -> set[int]:
    return {position for start, end in spans for position in range(start, end)}


def _matched_within(
    expected: Sequence[int],
    actual: Sequence[int],
    tolerance: int,
) -> int:
    candidates = sorted(actual)
    taken = [False] * len(candidates)
    matched = 0
    for boundary in sorted(expected):
        nearest = _nearest_free(candidates, taken, boundary, tolerance)
        if nearest is not None:
            taken[nearest] = True
            matched += 1
    return matched


def _nearest_free(
    candidates: Sequence[int],
    taken: Sequence[bool],
    boundary: int,
    tolerance: int,
) -> int | None:
    found: int | None = None
    closest = tolerance + 1
    for index, candidate in enumerate(candidates):
        distance = abs(candidate - boundary)
        if not taken[index] and distance < closest:
            found, closest = index, distance
    return found


def _ranks(values: Sequence[float]) -> list[float]:
    """Средние ранги: связанные значения получают одинаковый."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        shared = (start + end) / 2 + 1
        for index in order[start : end + 1]:
            ranks[index] = shared
        start = end + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    spread = math.sqrt(sum(value**2 for value in dx) * sum(value**2 for value in dy))
    if spread == 0:
        # Ряд без разброса не упорядочивает ничего, и корреляции нет.
        return 0.0
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / spread
