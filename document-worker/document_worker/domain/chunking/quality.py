"""Качество чанка: уверенность, неразборчивость, пригодность для поиска.

Уверенность взвешивается по символам между двумя доступными величинами:
средней по странице и уверенностью каждого структурного диапазона. Пословной
уверенности в схеме нет, поэтому читаемым символам приписывается средняя по
странице — для читаемой части это оценка сверху, для неразборчивой точное
значение.

Неразборчивость считается исключительно по структурным диапазонам. Маркеров в
каноническом тексте нет по построению, и поиск маркера регуляркой дал бы
двойной учёт одного и того же дефекта и завышенную долю.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from document_worker.domain.value_objects.confidence import OcrConfidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan

CHUNK_ILLEGIBLE_RATIO_THRESHOLD: Final[float] = 0.60
MIN_LEGIBLE_CHARS: Final[int] = 40
MIN_RETRIEVABLE_CONFIDENCE: Final[float] = 0.45
MIN_RETRIEVABLE_TOKENS: Final[int] = 24


@dataclass(frozen=True, slots=True)
class ChunkQuality:
    """Что известно о качестве фрагмента страницы."""

    avg_confidence: OcrConfidence | None
    illegible_span_count: int
    illegible_char_count: int
    illegible_char_ratio: float
    legible_char_count: int
    is_fully_illegible: bool
    is_retrievable: bool


class ChunkQualityEvaluator:
    """Считает качество фрагмента по данным его страницы."""

    def evaluate(
        self,
        *,
        page: DocumentPage,
        span: TextSpan,
        own_tokens: int,
    ) -> ChunkQuality:
        """Оценивает фрагмент страницы, заданный диапазоном."""
        overlaps: list[tuple[IllegibleSpan, int]] = [
            (illegible, _overlap_length(span, illegible.span))
            for illegible in page.illegible_spans
            if illegible.span.overlaps(span)
        ]
        illegible_chars = sum(length for _, length in overlaps)
        total = span.length
        ratio = illegible_chars / total if total else 0.0
        legible = _legible_char_count(page, span)
        # Порог читаемых символов применим только там, где неразборчивость
        # вообще есть: чистый короткий чанк не нечитаем, он просто короткий,
        # и это решает порог токенов.
        fully_illegible = ratio >= CHUNK_ILLEGIBLE_RATIO_THRESHOLD or (
            bool(overlaps) and legible < MIN_LEGIBLE_CHARS
        )
        confidence = _average_confidence(page, total=total, overlaps=overlaps)
        return ChunkQuality(
            avg_confidence=confidence,
            illegible_span_count=len(overlaps),
            illegible_char_count=illegible_chars,
            illegible_char_ratio=ratio,
            legible_char_count=legible,
            is_fully_illegible=fully_illegible,
            is_retrievable=_is_retrievable(
                fully_illegible=fully_illegible,
                confidence=confidence,
                own_tokens=own_tokens,
            ),
        )


def only_chunk_of_document(quality: ChunkQuality) -> ChunkQuality:
    """Единственный читаемый чанк документа индексируется всегда.

    Документ из одного слова иначе оказался бы полностью неиндексируемым:
    порог в токенах для него недостижим по построению.
    """
    if quality.is_fully_illegible or quality.is_retrievable:
        return quality
    return ChunkQuality(
        avg_confidence=quality.avg_confidence,
        illegible_span_count=quality.illegible_span_count,
        illegible_char_count=quality.illegible_char_count,
        illegible_char_ratio=quality.illegible_char_ratio,
        legible_char_count=quality.legible_char_count,
        is_fully_illegible=False,
        is_retrievable=True,
    )


def _overlap_length(span: TextSpan, other: TextSpan) -> int:
    return max(0, min(span.end, other.end) - max(span.start, other.start))


def _legible_char_count(page: DocumentPage, span: TextSpan) -> int:
    """Непробельные символы чанка вне неразборчивых диапазонов."""
    content = page.text.content
    covered = bytearray(span.length)
    for illegible in page.illegible_spans:
        start = max(span.start, illegible.span.start)
        end = min(span.end, illegible.span.end)
        for index in range(start - span.start, end - span.start):
            covered[index] = 1
    return sum(
        1
        for offset, character in enumerate(span.slice_of(content))
        if not covered[offset] and not character.isspace()
    )


def _average_confidence(
    page: DocumentPage,
    *,
    total: int,
    overlaps: Sequence[tuple[IllegibleSpan, int]],
) -> OcrConfidence | None:
    """Средняя уверенность фрагмента; у текстового слоя её не существует."""
    page_confidence = page.confidence
    if page_confidence is None or total == 0:
        return None
    illegible_chars = sum(length for _, length in overlaps)
    weighted = (total - illegible_chars) * page_confidence.value
    weighted += sum(
        length * illegible.confidence.value for illegible, length in overlaps
    )
    return OcrConfidence(weighted / total)


def _is_retrievable(
    *,
    fully_illegible: bool,
    confidence: OcrConfidence | None,
    own_tokens: int,
) -> bool:
    if fully_illegible or own_tokens < MIN_RETRIEVABLE_TOKENS:
        return False
    # Отсутствие уверенности — свойство текстового слоя, а не признак брака.
    return confidence is None or not confidence.is_below(MIN_RETRIEVABLE_CONFIDENCE)
