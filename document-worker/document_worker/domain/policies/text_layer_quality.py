"""Политика выбора способа извлечения: хватает ли текстового слоя.

Вход — чисто числовой снимок страницы. PDF адаптер разбирает сам, домен только
считает. Все пороги — поля политики, в теле методов чисел нет.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.value_objects.enums import ExtractionMethod

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.value_objects.paging import PageNumber

REASON_CLEAN_SPARSE_PAGE = "clean_sparse_page"
REASON_TOO_FEW_CHARS = "too_few_chars"
REASON_LOW_ALNUM_RATIO = "low_alnum_ratio"
REASON_MOJIBAKE = "mojibake"
REASON_UNDECODABLE_GLYPHS = "undecodable_glyphs"
REASON_CONTROL_CHARS = "control_chars"
REASON_TEXT_OVER_RASTER = "text_over_raster"
REASON_MEDIUM_QUALITY = "medium_quality"
REASON_LOW_QUALITY_SCORE = "low_quality_score"


@dataclass(frozen=True, slots=True)
class TextLayerProbe:
    """Числовой снимок текстового слоя одной страницы."""

    page_number: PageNumber
    char_count: int
    alnum_count: int
    word_count: int
    replacement_char_count: int
    control_char_count: int
    undecodable_char_count: int
    raster_area_ratio: float
    mean_word_length: float
    dictionary_word_ratio: float | None = None

    def ratio_of(self, count: int) -> float:
        """Доля символов от общего числа; пустая страница даёт ноль."""
        return count / max(self.char_count, 1)


@dataclass(frozen=True, slots=True)
class TextLayerVerdict:
    """Решение по одной странице."""

    page_number: PageNumber
    decision: ExtractionMethod
    score: float
    hard_reject: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentExtractionPlan:
    """План извлечения по всему документу."""

    verdicts: tuple[TextLayerVerdict, ...]
    pages_needing_render: tuple[PageNumber, ...]
    render_all_pages: bool
    dominant_method: ExtractionMethod


@dataclass(frozen=True, slots=True)
class TextLayerQualityPolicy:
    """Решает, брать ли текст из слоя, распознавать страницу или совместить."""

    clean_page_min_chars: int = 16
    clean_page_max_raster_ratio: float = 0.05

    min_chars_per_page: int = 120
    min_alnum_ratio: float = 0.55
    max_replacement_ratio: float = 0.02
    max_undecodable_ratio: float = 0.05
    max_control_ratio: float = 0.01

    dense_page_chars: int = 600
    good_dictionary_ratio: float = 0.80

    accept_score: float = 0.70
    hybrid_score: float = 0.45
    max_raster_area_ratio: float = 0.60
    render_all_pages_ratio: float = 0.30

    weight_density: float = 0.30
    weight_dictionary: float = 0.25
    weight_decodable: float = 0.20
    weight_encoding: float = 0.15
    weight_word_length: float = 0.10

    ideal_word_length: float = 7.5

    def score_weights(self) -> dict[str, float]:
        """Веса компонентов оценки качества слоя."""
        return {
            "density": self.weight_density,
            "dictionary": self.weight_dictionary,
            "decodable": self.weight_decodable,
            "encoding": self.weight_encoding,
            "word_length": self.weight_word_length,
        }

    def evaluate(self, probe: TextLayerProbe) -> TextLayerVerdict:
        """Решает, каким способом читать страницу."""
        if self._is_clean_sparse_page(probe):
            return TextLayerVerdict(
                page_number=probe.page_number,
                decision=ExtractionMethod.TEXT_LAYER,
                score=1.0,
                hard_reject=False,
                reasons=(REASON_CLEAN_SPARSE_PAGE,),
            )

        rejections = self._hard_rejections(probe)
        if rejections:
            return TextLayerVerdict(
                page_number=probe.page_number,
                decision=ExtractionMethod.OCR,
                score=self.score(probe),
                hard_reject=True,
                reasons=rejections,
            )

        return self._decide_by_score(probe)

    def score(self, probe: TextLayerProbe) -> float:
        """Считает качество слоя в диапазоне 0..1.

        Отсутствие словарной доли исключает её компонент и перенормирует веса:
        иначе страницы без словаря упирались бы в потолок и уезжали в гибрид.
        """
        components = [
            (self.weight_density, self._density(probe)),
            (self.weight_decodable, self._decodable(probe)),
            (self.weight_encoding, self._encoding(probe)),
            (self.weight_word_length, self._word_length(probe)),
        ]
        if probe.dictionary_word_ratio is not None:
            components.append((self.weight_dictionary, self._dictionary(probe)))

        total_weight = sum(weight for weight, _ in components)
        weighted = sum(weight * value for weight, value in components)
        return weighted / total_weight

    def plan(self, probes: Sequence[TextLayerProbe]) -> DocumentExtractionPlan:
        """Собирает план извлечения по всем страницам документа."""
        verdicts = tuple(self.evaluate(probe) for probe in probes)
        needing_render = tuple(
            verdict.page_number
            for verdict in verdicts
            if verdict.decision is not ExtractionMethod.TEXT_LAYER
        )
        render_all = bool(verdicts) and (
            len(needing_render) / len(verdicts) > self.render_all_pages_ratio
        )
        return DocumentExtractionPlan(
            verdicts=verdicts,
            pages_needing_render=needing_render,
            render_all_pages=render_all,
            dominant_method=_dominant_method(verdicts),
        )

    def _is_clean_sparse_page(self, probe: TextLayerProbe) -> bool:
        # Титульные листы и разделители читаются идеально, но общий порог объёма
        # отправил бы их на OCR, а OCR почти пустой страницы вернёт пустоту.
        return (
            probe.replacement_char_count == 0
            and probe.undecodable_char_count == 0
            and probe.control_char_count == 0
            and probe.raster_area_ratio <= self.clean_page_max_raster_ratio
            and probe.char_count >= self.clean_page_min_chars
            and probe.ratio_of(probe.alnum_count) >= self.min_alnum_ratio
        )

    def _hard_rejections(self, probe: TextLayerProbe) -> tuple[str, ...]:
        checks = (
            (probe.char_count < self.min_chars_per_page, REASON_TOO_FEW_CHARS),
            (
                probe.ratio_of(probe.alnum_count) < self.min_alnum_ratio,
                REASON_LOW_ALNUM_RATIO,
            ),
            (
                probe.ratio_of(probe.replacement_char_count)
                > self.max_replacement_ratio,
                REASON_MOJIBAKE,
            ),
            (
                probe.ratio_of(probe.undecodable_char_count)
                > self.max_undecodable_ratio,
                REASON_UNDECODABLE_GLYPHS,
            ),
            (
                probe.ratio_of(probe.control_char_count) > self.max_control_ratio,
                REASON_CONTROL_CHARS,
            ),
        )
        return tuple(reason for triggered, reason in checks if triggered)

    def _decide_by_score(self, probe: TextLayerProbe) -> TextLayerVerdict:
        score = self.score(probe)
        if score >= self.accept_score:
            over_raster = probe.raster_area_ratio > self.max_raster_area_ratio
            return TextLayerVerdict(
                page_number=probe.page_number,
                decision=ExtractionMethod.HYBRID
                if over_raster
                else ExtractionMethod.TEXT_LAYER,
                score=score,
                hard_reject=False,
                reasons=(REASON_TEXT_OVER_RASTER,) if over_raster else (),
            )
        if score >= self.hybrid_score:
            return TextLayerVerdict(
                page_number=probe.page_number,
                decision=ExtractionMethod.HYBRID,
                score=score,
                hard_reject=False,
                reasons=(REASON_MEDIUM_QUALITY,),
            )
        return TextLayerVerdict(
            page_number=probe.page_number,
            decision=ExtractionMethod.OCR,
            score=score,
            hard_reject=False,
            reasons=(REASON_LOW_QUALITY_SCORE,),
        )

    def _density(self, probe: TextLayerProbe) -> float:
        return min(1.0, probe.char_count / self.dense_page_chars)

    def _dictionary(self, probe: TextLayerProbe) -> float:
        ratio = probe.dictionary_word_ratio or 0.0
        return min(1.0, ratio / self.good_dictionary_ratio)

    def _decodable(self, probe: TextLayerProbe) -> float:
        share = (
            probe.ratio_of(probe.undecodable_char_count) / self.max_undecodable_ratio
        )
        return 1.0 - min(1.0, share)

    def _encoding(self, probe: TextLayerProbe) -> float:
        share = (
            probe.ratio_of(probe.replacement_char_count) / self.max_replacement_ratio
        )
        return 1.0 - min(1.0, share)

    def _word_length(self, probe: TextLayerProbe) -> float:
        deviation = abs(probe.mean_word_length - self.ideal_word_length)
        return max(
            0.0,
            1.0 - deviation / self.ideal_word_length,
        )


def _dominant_method(verdicts: tuple[TextLayerVerdict, ...]) -> ExtractionMethod:
    if not verdicts:
        return ExtractionMethod.NONE
    counts = Counter(verdict.decision for verdict in verdicts)
    return counts.most_common(1)[0][0]
