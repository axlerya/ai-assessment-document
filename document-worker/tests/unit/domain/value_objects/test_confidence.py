"""Тесты уверенности распознавания."""

from __future__ import annotations

import math

import pytest

from document_worker.domain.errors import InvalidConfidence
from document_worker.domain.value_objects.confidence import OcrConfidence

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [1.0001, 1.5, 2.0])
def test_rejects_value_above_one(value: float) -> None:
    with pytest.raises(InvalidConfidence):
        OcrConfidence(value)


@pytest.mark.parametrize("value", [-0.0001, -1.0])
def test_rejects_negative_value(value: float) -> None:
    with pytest.raises(InvalidConfidence):
        OcrConfidence(value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nan_and_infinity(value: float) -> None:
    with pytest.raises(InvalidConfidence):
        OcrConfidence(value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.123456, 0.1235), (0.99999, 1.0), (0.30000001, 0.3)],
)
def test_rounds_to_four_decimals(raw: float, expected: float) -> None:
    assert OcrConfidence(raw).value == expected


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_accepts_boundaries(value: float) -> None:
    assert OcrConfidence(value).value == value


def test_zero_constant_is_zero() -> None:
    assert OcrConfidence.ZERO.value == 0.0


def test_is_below_compares_with_threshold() -> None:
    confidence = OcrConfidence(0.4)

    assert confidence.is_below(0.5)
    assert not confidence.is_below(0.4)


def test_confidences_are_ordered() -> None:
    assert sorted([OcrConfidence(0.9), OcrConfidence(0.1)]) == [
        OcrConfidence(0.1),
        OcrConfidence(0.9),
    ]


def test_weighted_mean_uses_char_length_weights() -> None:
    pairs = [(OcrConfidence(0.9), 10), (OcrConfidence(0.5), 90)]

    assert OcrConfidence.weighted_mean(pairs) == OcrConfidence(0.54)


def test_weighted_mean_differs_from_plain_average() -> None:
    pairs = [(OcrConfidence(1.0), 1), (OcrConfidence(0.0), 99)]

    assert OcrConfidence.weighted_mean(pairs) == OcrConfidence(0.01)


def test_weighted_mean_returns_none_for_empty_sequence() -> None:
    assert OcrConfidence.weighted_mean([]) is None


def test_weighted_mean_returns_none_for_zero_total_weight() -> None:
    assert OcrConfidence.weighted_mean([(OcrConfidence(0.9), 0)]) is None


def test_weighted_mean_rejects_negative_weight() -> None:
    with pytest.raises(InvalidConfidence):
        OcrConfidence.weighted_mean([(OcrConfidence(0.9), -1)])


def test_confidence_is_immutable() -> None:
    confidence = OcrConfidence(0.5)

    with pytest.raises(AttributeError):
        confidence.value = 0.9  # type: ignore[misc]
