"""Тесты политики выбора способа извлечения текста."""

from __future__ import annotations

import pytest

from document_worker.domain.policies.text_layer_quality import (
    TextLayerProbe,
    TextLayerQualityPolicy,
)
from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.domain.value_objects.paging import PageNumber

pytestmark = pytest.mark.unit

POLICY = TextLayerQualityPolicy()

# Поля зонда — чистые числа. PDF-специфики (шрифты, ToUnicode, XObject) здесь быть
# не может: домен ничего не парсит.
PDF_SPECIFIC_FIELDS = {"fonts", "to_unicode", "xobject", "cmap", "encoding", "stream"}


def _probe(  # noqa: PLR0913 — зонд страницы это набор независимых чисел
    *,
    page: int = 1,
    char_count: int = 2000,
    alnum_count: int | None = None,
    word_count: int = 300,
    replacement_char_count: int = 0,
    control_char_count: int = 0,
    undecodable_char_count: int = 0,
    raster_area_ratio: float = 0.0,
    mean_word_length: float = 6.0,
    dictionary_word_ratio: float | None = 0.9,
) -> TextLayerProbe:
    return TextLayerProbe(
        page_number=PageNumber(page),
        char_count=char_count,
        alnum_count=char_count if alnum_count is None else alnum_count,
        word_count=word_count,
        replacement_char_count=replacement_char_count,
        control_char_count=control_char_count,
        undecodable_char_count=undecodable_char_count,
        raster_area_ratio=raster_area_ratio,
        mean_word_length=mean_word_length,
        dictionary_word_ratio=dictionary_word_ratio,
    )


def test_probe_contains_no_pdf_specific_fields() -> None:
    fields = set(TextLayerProbe.__dataclass_fields__)

    assert not fields & PDF_SPECIFIC_FIELDS


def test_accepts_dense_digital_page() -> None:
    verdict = POLICY.evaluate(_probe())

    assert verdict.decision is ExtractionMethod.TEXT_LAYER
    assert not verdict.hard_reject


def test_accepts_sparse_but_clean_page_without_ocr() -> None:
    # Титульный лист: «Приложение № 3 к Договору» — 30 символов и ни одного дефекта.
    verdict = POLICY.evaluate(
        _probe(char_count=30, alnum_count=25, word_count=5, mean_word_length=5.0)
    )

    assert verdict.decision is ExtractionMethod.TEXT_LAYER
    assert verdict.reasons == ("clean_sparse_page",)
    assert verdict.score == 1.0


def test_rejects_page_with_zero_chars_and_full_page_image() -> None:
    verdict = POLICY.evaluate(
        _probe(char_count=0, alnum_count=0, word_count=0, raster_area_ratio=1.0)
    )

    assert verdict.decision is ExtractionMethod.OCR
    assert verdict.hard_reject
    assert "too_few_chars" in verdict.reasons


def test_rejects_page_with_replacement_ratio_above_threshold() -> None:
    verdict = POLICY.evaluate(
        _probe(char_count=1000, replacement_char_count=30, raster_area_ratio=0.5)
    )

    assert verdict.decision is ExtractionMethod.OCR
    assert "mojibake" in verdict.reasons


def test_rejects_page_with_undecodable_ratio_above_threshold() -> None:
    verdict = POLICY.evaluate(
        _probe(char_count=1000, undecodable_char_count=100, raster_area_ratio=0.5)
    )

    assert verdict.decision is ExtractionMethod.OCR
    assert "undecodable_glyphs" in verdict.reasons


def test_rejects_page_with_control_chars_above_threshold() -> None:
    verdict = POLICY.evaluate(
        _probe(char_count=1000, control_char_count=50, raster_area_ratio=0.5)
    )

    assert verdict.decision is ExtractionMethod.OCR
    assert "control_chars" in verdict.reasons


def test_rejects_page_with_low_alnum_ratio() -> None:
    verdict = POLICY.evaluate(
        _probe(char_count=1000, alnum_count=100, raster_area_ratio=0.5)
    )

    assert verdict.decision is ExtractionMethod.OCR
    assert "low_alnum_ratio" in verdict.reasons


def test_verdict_lists_all_triggered_reasons() -> None:
    verdict = POLICY.evaluate(
        _probe(
            char_count=1000,
            alnum_count=100,
            replacement_char_count=30,
            undecodable_char_count=100,
            control_char_count=50,
            raster_area_ratio=0.5,
        )
    )

    assert set(verdict.reasons) == {
        "low_alnum_ratio",
        "mojibake",
        "undecodable_glyphs",
        "control_chars",
    }


def test_rejects_glued_text_without_spaces() -> None:
    verdict = POLICY.evaluate(
        _probe(
            char_count=150,
            word_count=1,
            mean_word_length=150.0,
            dictionary_word_ratio=0.0,
            raster_area_ratio=0.5,
        )
    )

    assert verdict.decision is ExtractionMethod.OCR
    assert "low_quality_score" in verdict.reasons


def test_returns_hybrid_when_clean_text_overlaps_large_image_area() -> None:
    # Договор с вклеенными сканами печатей: слой хорош, но растр занимает лист.
    verdict = POLICY.evaluate(_probe(raster_area_ratio=0.8))

    assert verdict.decision is ExtractionMethod.HYBRID
    assert "text_over_raster" in verdict.reasons


def test_returns_hybrid_for_medium_quality_page() -> None:
    verdict = POLICY.evaluate(
        _probe(
            char_count=300,
            word_count=50,
            mean_word_length=6.0,
            dictionary_word_ratio=0.3,
            raster_area_ratio=0.5,
        )
    )

    assert verdict.decision is ExtractionMethod.HYBRID
    assert "medium_quality" in verdict.reasons


def test_missing_dictionary_ratio_does_not_lower_score() -> None:
    with_dictionary = POLICY.score(_probe(dictionary_word_ratio=0.9))
    without_dictionary = POLICY.score(_probe(dictionary_word_ratio=None))

    assert without_dictionary == pytest.approx(with_dictionary, abs=0.05)


def test_score_is_deterministic_for_same_probe() -> None:
    probe = _probe(char_count=300, raster_area_ratio=0.5, dictionary_word_ratio=0.5)

    assert POLICY.score(probe) == POLICY.score(probe)


def test_score_weights_sum_to_one() -> None:
    assert sum(POLICY.score_weights().values()) == pytest.approx(1.0)


@pytest.mark.parametrize("raster", [0.0, 0.5, 1.0])
def test_score_stays_within_unit_range(raster: float) -> None:
    score = POLICY.score(_probe(char_count=1, raster_area_ratio=raster))

    assert 0.0 <= score <= 1.0


def test_plan_marks_pages_needing_render() -> None:
    probes = [
        _probe(page=1),
        _probe(page=2, raster_area_ratio=0.8),
        _probe(page=3, char_count=0, alnum_count=0, word_count=0),
    ]

    plan = POLICY.plan(probes)

    assert plan.pages_needing_render == (PageNumber(2), PageNumber(3))
    assert plan.dominant_method is ExtractionMethod.TEXT_LAYER


def test_plan_renders_all_pages_when_many_need_ocr() -> None:
    probes = [
        _probe(page=1),
        _probe(page=2, char_count=0, alnum_count=0, word_count=0),
    ]

    plan = POLICY.plan(probes)

    assert plan.render_all_pages


def test_plan_keeps_render_off_for_clean_document() -> None:
    plan = POLICY.plan([_probe(page=number) for number in range(1, 11)])

    assert plan.pages_needing_render == ()
    assert not plan.render_all_pages


def test_plan_of_empty_document_has_no_pages() -> None:
    plan = POLICY.plan([])

    assert plan.verdicts == ()
    assert not plan.render_all_pages
    assert plan.dominant_method is ExtractionMethod.NONE
