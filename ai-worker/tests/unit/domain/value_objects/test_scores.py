"""Скоры и доли: числа, которые попадают в выдачу и в отчёт качества."""

from __future__ import annotations

import math

import pytest

from ai_worker.domain.errors import InvalidScore
from ai_worker.domain.value_objects.scores import Ratio, RrfScore, Score

pytestmark = pytest.mark.unit

NOT_A_NUMBER = (math.nan, math.inf, -math.inf)


@pytest.mark.parametrize("value", NOT_A_NUMBER)
def test_score_rejects_values_that_are_not_numbers(value: float) -> None:
    # NaN ломает сортировку молча: он не больше и не меньше ничего, и выдача
    # перестаёт быть упорядоченной без единой ошибки.
    with pytest.raises(InvalidScore):
        Score(value)


@pytest.mark.parametrize("value", NOT_A_NUMBER)
def test_ratio_rejects_values_that_are_not_numbers(value: float) -> None:
    with pytest.raises(InvalidScore):
        Ratio(value)


@pytest.mark.parametrize("value", [-0.001, 1.001, 2.0, -1.0])
def test_ratio_stays_within_its_range(value: float) -> None:
    with pytest.raises(InvalidScore):
        Ratio(value)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_ratio_accepts_its_boundaries(value: float) -> None:
    assert Ratio(value).value == value


def test_score_allows_any_finite_value() -> None:
    # Косинус приходит из −1..1, а логит кросс-энкодера ничем не ограничен:
    # сузить диапазон здесь значило бы отвергать честные значения.
    assert Score(-3.5).value == -3.5
    assert Score(12.75).value == 12.75


def test_scores_are_ordered() -> None:
    assert Score(0.1) < Score(0.9)
    assert max(Score(0.1), Score(0.9)) == Score(0.9)


@pytest.mark.parametrize("value", [0.0, -0.1])
def test_rrf_score_is_strictly_positive(value: float) -> None:
    # Слияние складывает 1/(k + rank) по найденным ветвям: ноль означал бы
    # попадание, не найденное ни одной из них.
    with pytest.raises(InvalidScore):
        RrfScore(value)


def test_rrf_scores_are_ordered() -> None:
    assert RrfScore(0.01) < RrfScore(0.02)


def test_ratio_of_counts_computes_the_share() -> None:
    assert Ratio.of(part=3, whole=4) == Ratio(0.75)


def test_ratio_of_nothing_is_zero_not_a_division_error() -> None:
    # Черновик без утверждений — штатный исход: обоснованность такого
    # черновика равна нулю, а не падает.
    assert Ratio.of(part=0, whole=0) == Ratio(0.0)


def test_ratio_rejects_a_part_bigger_than_the_whole() -> None:
    with pytest.raises(InvalidScore):
        Ratio.of(part=5, whole=4)


@pytest.mark.parametrize(("part", "whole"), [(-1, 4), (1, -4), (-1, -4)])
def test_ratio_rejects_negative_counts(part: int, whole: int) -> None:
    # Отрицательный счётчик означает ошибку подсчёта утверждений выше по
    # потоку, и превращать её в правдоподобную долю нельзя.
    with pytest.raises(InvalidScore):
        Ratio.of(part=part, whole=whole)
