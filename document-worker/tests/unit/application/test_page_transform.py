"""Преобразование координат страницы."""

from __future__ import annotations

import pytest

from document_worker.application.dto.ocr import PageTransform
from document_worker.domain.errors import InvariantViolation

pytestmark = pytest.mark.unit


def test_non_positive_page_size_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="положительным"):
        PageTransform.identity(width_px=0, height_px=100)


def test_degenerate_matrix_is_refused() -> None:
    # Необратимая матрица не переводит бокс обратно на страницу, и подсветка
    # фрагмента указала бы оператору куда угодно.
    with pytest.raises(InvariantViolation, match="вырожденное"):
        PageTransform(
            a=0.0,
            b=0.0,
            c=0.0,
            d=0.0,
            e=0.0,
            f=0.0,
            page_width_px=100,
            page_height_px=100,
        )


def test_composed_transform_maps_a_point_through_both_steps() -> None:
    shifted = PageTransform.identity(width_px=100, height_px=100).then(
        (1.0, 0.0, 10.0, 0.0, 1.0, -5.0)
    )

    assert shifted.to_image(0.0, 0.0) == (10.0, -5.0)
    assert shifted.to_page(10.0, -5.0) == (0.0, 0.0)
