"""Качество чанка: уверенность, неразборчивость, пригодность для поиска."""

from __future__ import annotations

import pytest

from document_worker.domain.chunking.pipeline import build_pipeline
from document_worker.domain.chunking.quality import (
    MIN_RETRIEVABLE_TOKENS,
    ChunkQualityEvaluator,
)
from document_worker.domain.value_objects.enums import ExtractionMethod, PageStatus
from document_worker.domain.value_objects.text import TextSpan
from tests.unit.domain.chunking.support import (
    FakeTokenCounter,
    default_policy,
    ocr_page,
    text_layer_page,
)

pytestmark = pytest.mark.unit

EVALUATOR = ChunkQualityEvaluator()
CONTENT = "я" * 100
ENOUGH_TOKENS = MIN_RETRIEVABLE_TOKENS


def test_computes_char_weighted_average_confidence() -> None:
    # Десять символов с уверенностью 0.3 против девяноста при 0.9 дают 0.84,
    # а не 0.6: усреднять сами величины, игнорируя их вес, нельзя.
    page = ocr_page(CONTENT, confidence=0.9, illegible=((0, 10, 0.3),))

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(0, 100), own_tokens=ENOUGH_TOKENS
    )

    assert quality.avg_confidence is not None
    assert quality.avg_confidence.value == pytest.approx(0.84)


def test_text_layer_chunk_has_none_confidence() -> None:
    page = text_layer_page(CONTENT)

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(0, 100), own_tokens=ENOUGH_TOKENS
    )

    assert quality.avg_confidence is None
    assert quality.is_retrievable is True


def test_hybrid_chunk_has_confidence() -> None:
    page = ocr_page(CONTENT, method=ExtractionMethod.HYBRID, confidence=0.8)

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(0, 100), own_tokens=ENOUGH_TOKENS
    )

    assert quality.avg_confidence is not None


def test_illegible_ratio_counts_only_structural_spans() -> None:
    # Подстрока, похожая на маркер, диапазоном не является: учёт по тексту дал бы
    # тот же дефект дважды.
    content = "НЕРАЗБОРЧИВО строка 14 " * 5
    page = ocr_page(content, confidence=0.9)

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(0, len(content)), own_tokens=ENOUGH_TOKENS
    )

    assert quality.illegible_char_ratio == 0.0
    assert quality.illegible_span_count == 0


def test_illegible_span_count_counts_only_intersecting_spans() -> None:
    page = ocr_page(CONTENT, confidence=0.9, illegible=((0, 7, 0.3),))

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(50, 100), own_tokens=ENOUGH_TOKENS
    )

    assert quality.illegible_span_count == 0
    assert quality.illegible_char_count == 0


def test_marks_chunk_fully_illegible_above_ratio_threshold() -> None:
    page = ocr_page(CONTENT, confidence=0.9, illegible=((0, 75, 0.3),))

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(0, 100), own_tokens=ENOUGH_TOKENS
    )

    assert quality.illegible_char_ratio == 0.75
    assert quality.is_fully_illegible is True
    assert quality.is_retrievable is False


def test_short_clean_chunk_is_not_called_illegible() -> None:
    # Короткий чанк без единого диапазона нечитаемым не является: он просто
    # короткий, и это решает порог токенов, а не признак неразборчивости.
    page = text_layer_page("Договор")

    quality = EVALUATOR.evaluate(page=page, span=TextSpan(0, 7), own_tokens=2)

    assert quality.is_fully_illegible is False
    assert quality.is_retrievable is False


def test_low_average_confidence_makes_chunk_not_retrievable() -> None:
    page = ocr_page(CONTENT, confidence=0.4)

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(0, 100), own_tokens=ENOUGH_TOKENS
    )

    assert quality.is_retrievable is False


def test_too_few_own_tokens_make_chunk_not_retrievable() -> None:
    page = text_layer_page(CONTENT)

    quality = EVALUATOR.evaluate(
        page=page, span=TextSpan(0, 100), own_tokens=MIN_RETRIEVABLE_TOKENS - 1
    )

    assert quality.is_retrievable is False


def test_chunk_from_fully_illegible_page_is_saved_not_dropped() -> None:
    # Устав прямо запрещает выдумывать текст, а удаление чанка ломает аудит
    # и подсчёт покрытия документа.
    page = ocr_page(
        CONTENT,
        confidence=0.3,
        illegible=((0, 90, 0.2),),
        status=PageStatus.ILLEGIBLE,
    )
    drafts = build_pipeline(default_policy(), FakeTokenCounter()).run([page])

    assert len(drafts) == 1
    assert drafts[0].text == CONTENT
    assert drafts[0].quality.illegible_span_count >= 1
    assert drafts[0].quality.is_retrievable is False
