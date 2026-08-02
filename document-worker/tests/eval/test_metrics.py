"""Метрики качества: сравнение эталона с тем, что выдал сервис."""

from __future__ import annotations

import pytest

from eval.metrics import (
    boundary_f1,
    cer,
    normalize_for_scoring,
    span_iou,
    spearman,
    wer,
)

pytestmark = pytest.mark.unit

MARKED = "[НЕРАЗБОРЧИВО: строка 3, confidence=0.31]"


def test_cer_of_identical_strings_is_zero() -> None:
    assert cer("Договор поставки", "Договор поставки") == 0.0


def test_cer_counts_every_wrong_character() -> None:
    # Семь символов эталона, один заменён.
    assert cer("договор", "договоp") == pytest.approx(1 / 7)


def test_cer_of_empty_reference_is_zero_when_nothing_recognized() -> None:
    assert cer("", "") == 0.0


def test_cer_of_empty_reference_is_one_when_something_recognized() -> None:
    # Делить не на что, но выдуманный текст обязан быть виден метрике.
    assert cer("", "договор") == 1.0


def test_cer_ignores_illegibility_markers() -> None:
    # Маркер — представление, а не текст: его символы не должны попадать в
    # расстояние, иначе честная пометка ухудшала бы метрику сильнее молчания.
    reference = "Стороны договорились о следующем"
    assert cer(reference, f"Стороны {MARKED}договорились о следующем") == cer(
        reference, "Стороны договорились о следующем"
    )


def test_cer_keeps_case_because_it_matters_in_legal_text() -> None:
    assert cer("Договор", "договор") > 0.0


def test_wer_counts_words_not_characters() -> None:
    reference = "предмет договора поставка товара"
    hypothesis = "предмет договора поставка тoвара"

    assert wer(reference, hypothesis) == pytest.approx(1 / 4)
    assert cer(reference, hypothesis) < wer(reference, hypothesis)


def test_normalization_collapses_whitespace() -> None:
    assert normalize_for_scoring("Договор \t\n  поставки") == "Договор поставки"


def test_normalization_removes_soft_hyphen() -> None:
    assert normalize_for_scoring("дого­вор") == "договор"


def test_normalization_unifies_quotes_and_dashes() -> None:
    assert normalize_for_scoring("«Поставщик» — сторона") == normalize_for_scoring(
        '"Поставщик" - сторона'
    )


def test_boundary_f1_tolerates_offset_within_window() -> None:
    assert boundary_f1([100, 400], [108, 393], tolerance=20) == 1.0


def test_boundary_f1_penalises_offset_beyond_window() -> None:
    assert boundary_f1([100], [140], tolerance=20) == 0.0


def test_boundary_f1_of_no_boundaries_on_both_sides_is_one() -> None:
    # Нечего искать и ничего не найдено — это согласие, а не провал.
    assert boundary_f1([], [], tolerance=20) == 1.0


def test_boundary_f1_of_missed_boundaries_is_zero() -> None:
    assert boundary_f1([100], [], tolerance=20) == 0.0
    assert boundary_f1([], [100], tolerance=20) == 0.0


def test_boundary_f1_matches_each_expected_boundary_once() -> None:
    # Две границы в окне одной эталонной — это одно попадание и один промах,
    # иначе дробление документа улучшало бы метрику.
    assert boundary_f1([100], [95, 105], tolerance=20) == pytest.approx(2 / 3)


def test_span_iou_of_identical_spans_is_one() -> None:
    assert span_iou([(10, 20)], [(10, 20)]) == 1.0


def test_span_iou_of_disjoint_spans_is_zero() -> None:
    assert span_iou([(10, 20)], [(30, 40)]) == 0.0


def test_span_iou_counts_characters_not_spans() -> None:
    assert span_iou([(10, 20)], [(15, 25)]) == pytest.approx(5 / 15)


def test_span_iou_of_nothing_expected_and_nothing_found_is_one() -> None:
    assert span_iou([], []) == 1.0


def test_span_iou_of_nothing_expected_but_something_found_is_zero() -> None:
    assert span_iou([], [(10, 20)]) == 0.0


def test_spearman_of_monotone_series_is_one() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(
        1.0
    )


def test_spearman_of_reversed_series_is_minus_one() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(
        -1.0
    )


def test_spearman_uses_ranks_not_values() -> None:
    # Связь монотонная, но не линейная: Пирсон дал бы меньше единицы.
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 800.0]) == pytest.approx(1.0)


def test_spearman_of_constant_series_is_zero() -> None:
    # Постоянная уверенность ничего не предсказывает, и корреляции нет.
    assert spearman([0.9, 0.9, 0.9], [0.1, 0.5, 0.9]) == 0.0


def test_spearman_of_too_short_series_is_zero() -> None:
    assert spearman([0.9], [0.1]) == 0.0
