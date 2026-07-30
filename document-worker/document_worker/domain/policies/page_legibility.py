"""Политика читаемости страницы: статус по уверенности распознанных слов.

Уверенность взвешивается по длине слова, а не усредняется по словам. OCR
систематически занижает уверенность коротким токенам — предлогам, знакам
препинания, номерам пунктов; простое среднее уравняло бы такой токен со словом
«ответственность». Вес по длине совпадает с величиной, в которой измеряется
результат работы сервиса, доля неразборчивого и среднее по документу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    IllegibleReason,
    PageStatus,
)
from document_worker.domain.value_objects.quality import PageLegibilityVerdict
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.value_objects.geometry import BoundingBox
    from document_worker.domain.value_objects.text import RecognizedWord

WARNING_SPARSE_TEXT = "sparse_text"
WARNING_LOW_MEAN_CONFIDENCE = "low_mean_confidence"
WARNING_HEAVILY_DEGRADED = "heavily_degraded"
WARNING_MANY_FRAGMENTS = "many_illegible_fragments"
WARNING_CRITICAL_WORDS = "critical_words_present"


@dataclass(frozen=True, slots=True)
class PageLegibilityPolicy:
    """Определяет статус страницы и собирает неразборчивые диапазоны."""

    word_low_confidence: float = 0.60
    word_critical_confidence: float = 0.35
    page_min_mean_confidence: float = 0.75
    page_critical_mean_confidence: float = 0.50
    illegible_ratio_threshold: float = 0.35
    warn_illegible_ratio: float = 0.05
    min_words_per_page: int = 5
    min_chars_per_page: int = 40
    merge_gap_chars: int = 3
    many_fragments_warning: int = 10

    def evaluate(
        self,
        *,
        method: ExtractionMethod,
        words: Sequence[RecognizedWord],
        content: str,
    ) -> PageLegibilityVerdict:
        """Возвращает вердикт о читаемости страницы."""
        if method is ExtractionMethod.TEXT_LAYER:
            return self._text_layer_verdict()

        total_weight = sum(len(word.text) for word in words)
        if not words or total_weight == 0:
            return self._nothing_recognized_verdict()
        if (
            len(words) < self.min_words_per_page
            or total_weight < self.min_chars_per_page
        ):
            return self._sparse_verdict(content)

        mean_confidence = _weighted_mean(words)
        spans = self._collect_spans(words, content)
        illegible_ratio = sum(span.span.length for span in spans) / max(len(content), 1)
        return PageLegibilityVerdict(
            status=self._decide(spans, mean_confidence, illegible_ratio),
            mean_confidence=mean_confidence,
            illegible_spans=spans,
            illegible_ratio=illegible_ratio,
            warnings=self._warnings(words, spans, mean_confidence, illegible_ratio),
        )

    def _text_layer_verdict(self) -> PageLegibilityVerdict:
        # Качество слоя уже подтвердила политика выбора способа извлечения,
        # а уверенности у текстового слоя не существует.
        return PageLegibilityVerdict(
            status=PageStatus.EXTRACTED,
            mean_confidence=OcrConfidence.ZERO,
            illegible_spans=(),
            illegible_ratio=0.0,
            warnings=(),
        )

    def _nothing_recognized_verdict(self) -> PageLegibilityVerdict:
        return PageLegibilityVerdict(
            status=PageStatus.ILLEGIBLE,
            mean_confidence=OcrConfidence.ZERO,
            illegible_spans=(
                IllegibleSpan(
                    span=TextSpan(0, 0),
                    confidence=OcrConfidence.ZERO,
                    reason=IllegibleReason.NO_TEXT_RECOGNIZED,
                    raw_text="",
                ),
            ),
            illegible_ratio=0.0,
            warnings=(),
        )

    def _sparse_verdict(self, content: str) -> PageLegibilityVerdict:
        return PageLegibilityVerdict(
            status=PageStatus.ILLEGIBLE,
            mean_confidence=OcrConfidence.ZERO,
            illegible_spans=(
                IllegibleSpan(
                    span=TextSpan(0, len(content)),
                    confidence=OcrConfidence.ZERO,
                    reason=IllegibleReason.IMAGE_TOO_NOISY,
                    raw_text=content,
                ),
            ),
            illegible_ratio=1.0 if content else 0.0,
            warnings=(WARNING_SPARSE_TEXT,),
        )

    def _collect_spans(
        self,
        words: Sequence[RecognizedWord],
        content: str,
    ) -> tuple[IllegibleSpan, ...]:
        # Клякса или печать поверх текста разрезаются распознавателем на 2–4
        # токена; без склейки оператор получил бы три маркера на один дефект.
        groups: list[list[RecognizedWord]] = []
        for word in words:
            if not word.confidence.is_below(self.word_low_confidence):
                continue
            if groups and self._is_adjacent(groups[-1][-1], word):
                groups[-1].append(word)
            else:
                groups.append([word])
        return tuple(self._build_span(group, content) for group in groups)

    def _is_adjacent(self, previous: RecognizedWord, current: RecognizedWord) -> bool:
        return current.span.start - previous.span.end <= self.merge_gap_chars

    def _build_span(
        self,
        group: Sequence[RecognizedWord],
        content: str,
    ) -> IllegibleSpan:
        span = TextSpan(group[0].span.start, group[-1].span.end)
        recognized_anything = any(word.text for word in group)
        return IllegibleSpan(
            span=span,
            confidence=_weighted_mean(group),
            reason=IllegibleReason.LOW_OCR_CONFIDENCE
            if recognized_anything
            else IllegibleReason.NO_TEXT_RECOGNIZED,
            raw_text=span.slice_of(content),
            bbox=_union_of(group),
            line_number=group[0].line_number,
        )

    def _decide(
        self,
        spans: tuple[IllegibleSpan, ...],
        mean_confidence: OcrConfidence,
        illegible_ratio: float,
    ) -> PageStatus:
        if illegible_ratio > self.illegible_ratio_threshold or mean_confidence.is_below(
            self.page_critical_mean_confidence
        ):
            return PageStatus.ILLEGIBLE
        # Вернуть EXTRACTED вместе с непустым списком нельзя: страница с таким
        # вердиктом не собирается.
        if spans:
            return PageStatus.PARTIALLY_ILLEGIBLE
        return PageStatus.EXTRACTED

    def _warnings(
        self,
        words: Sequence[RecognizedWord],
        spans: tuple[IllegibleSpan, ...],
        mean_confidence: OcrConfidence,
        illegible_ratio: float,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if mean_confidence.is_below(self.page_min_mean_confidence):
            warnings.append(WARNING_LOW_MEAN_CONFIDENCE)
        if illegible_ratio > self.warn_illegible_ratio:
            warnings.append(WARNING_HEAVILY_DEGRADED)
        if len(spans) > self.many_fragments_warning:
            warnings.append(WARNING_MANY_FRAGMENTS)
        if any(
            word.confidence.is_below(self.word_critical_confidence) for word in words
        ):
            warnings.append(WARNING_CRITICAL_WORDS)
        return tuple(warnings)


def _weighted_mean(words: Sequence[RecognizedWord]) -> OcrConfidence:
    pairs = [(word.confidence, len(word.text)) for word in words]
    return OcrConfidence.weighted_mean(pairs) or OcrConfidence.ZERO


def _union_of(words: Sequence[RecognizedWord]) -> BoundingBox | None:
    boxes = [word.bbox for word in words if word.bbox is not None]
    if not boxes:
        return None
    united = boxes[0]
    for box in boxes[1:]:
        united = united.union(box)
    return united
