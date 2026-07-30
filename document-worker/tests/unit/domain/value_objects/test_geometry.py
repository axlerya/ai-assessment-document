"""Тесты ограничивающего прямоугольника."""

from __future__ import annotations

import pytest

from document_worker.domain.errors import InvalidBoundingBox
from document_worker.domain.value_objects.geometry import BoundingBox

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("x0", "y0", "x1", "y1"),
    [
        (-0.1, 0.0, 0.5, 0.5),
        (0.0, -0.1, 0.5, 0.5),
        (0.0, 0.0, 1.1, 0.5),
        (0.0, 0.0, 0.5, 1.1),
    ],
)
def test_rejects_coordinates_outside_unit_range(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    with pytest.raises(InvalidBoundingBox):
        BoundingBox(x0, y0, x1, y1)


@pytest.mark.parametrize(
    ("x0", "y0", "x1", "y1"),
    [
        (0.5, 0.5, 0.5, 0.9),
        (0.5, 0.5, 0.9, 0.5),
        (0.9, 0.5, 0.5, 0.9),
        (0.5, 0.9, 0.9, 0.5),
    ],
)
def test_rejects_zero_or_negative_area(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    with pytest.raises(InvalidBoundingBox):
        BoundingBox(x0, y0, x1, y1)


def test_width_height_and_area() -> None:
    box = BoundingBox(0.1, 0.2, 0.5, 0.4)

    assert box.width == pytest.approx(0.4)
    assert box.height == pytest.approx(0.2)
    assert box.area == pytest.approx(0.08)


def test_from_pixels_normalizes_coordinates() -> None:
    box = BoundingBox.from_pixels(x=100, y=200, w=300, h=100, page_w=1000, page_h=2000)

    assert (box.x0, box.y0, box.x1, box.y1) == (0.1, 0.1, 0.4, 0.15)


def test_from_pixels_rejects_non_positive_page_size() -> None:
    with pytest.raises(InvalidBoundingBox):
        BoundingBox.from_pixels(x=0, y=0, w=10, h=10, page_w=0, page_h=100)


def test_to_pixels_is_inverse_of_from_pixels() -> None:
    box = BoundingBox.from_pixels(x=100, y=200, w=300, h=100, page_w=1000, page_h=2000)

    assert box.to_pixels(1000, 2000) == (100, 200, 400, 300)


def test_union_covers_both_boxes() -> None:
    first = BoundingBox(0.1, 0.1, 0.3, 0.3)
    second = BoundingBox(0.2, 0.4, 0.6, 0.5)

    assert first.union(second) == BoundingBox(0.1, 0.1, 0.6, 0.5)


def test_union_is_commutative() -> None:
    first = BoundingBox(0.1, 0.1, 0.3, 0.3)
    second = BoundingBox(0.2, 0.4, 0.6, 0.5)

    assert first.union(second) == second.union(first)


def test_intersects_is_true_for_overlapping_boxes() -> None:
    assert BoundingBox(0.1, 0.1, 0.4, 0.4).intersects(BoundingBox(0.3, 0.3, 0.6, 0.6))


def test_intersects_is_false_for_touching_boxes() -> None:
    assert not BoundingBox(0.1, 0.1, 0.3, 0.3).intersects(
        BoundingBox(0.3, 0.1, 0.6, 0.3)
    )


def test_intersects_is_false_for_disjoint_boxes() -> None:
    assert not BoundingBox(0.1, 0.1, 0.2, 0.2).intersects(
        BoundingBox(0.7, 0.7, 0.9, 0.9)
    )
