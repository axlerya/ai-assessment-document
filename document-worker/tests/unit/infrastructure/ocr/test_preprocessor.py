"""Предобработка изображения страницы.

Каждый шаг условный: «всегда всё» ухудшает чистые сканы, потому что
нейросетевой распознаватель обучен на изображениях с антиалиасингом, и
агрессивная чистка съедает признаки, на которые он опирается.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from document_worker.application.dto.ocr import (
    PageImage,
    PageTransform,
    PreprocessProfile,
)
from document_worker.infrastructure.ocr.preprocessor import (
    DESKEW_DISAGREEMENT_LIMIT_DEG,
    STEP_DESKEW_UNCERTAIN,
    STEP_DESKEWED,
    STEP_INVERTED,
    prepare_page,
)
from tests.fakes.page_images import blank_page_png, make_page_image, make_page_png

pytestmark = pytest.mark.unit

SKEW_DEG = 2.7
SKEW_TOLERANCE_DEG = 0.4
ROUNDTRIP_TOLERANCE = 0.004


def page_image(png: bytes) -> PageImage:
    """Оборачивает байты в то, что ходит между процессами."""
    with Image.open(io.BytesIO(png)) as image:
        return PageImage(png=png, width_px=image.width, height_px=image.height, dpi=150)


def prepared(
    png: bytes, profile: PreprocessProfile = PreprocessProfile.DEFAULT
) -> object:
    """Прогоняет предобработку в текущем процессе."""
    return prepare_page(page_image(png), profile)


def mean_level(png: bytes) -> float:
    with Image.open(io.BytesIO(png)) as image:
        pixels = list(image.convert("L").getdata())
    return sum(pixels) / len(pixels)


def test_clean_scan_needs_no_step_at_all() -> None:
    result = prepared(make_page_png())

    assert result.applied == ()


def test_inversion_is_detected_on_negative_image() -> None:
    # Бланк «белым по чёрному» без переворота даёт инвертированные оценки
    # контраста на всех последующих шагах.
    result = prepared(make_page_png(inverted=True))

    assert STEP_INVERTED in result.applied
    assert mean_level(result.image.png) > mean_level(make_page_png(inverted=True))


def test_deskew_estimates_known_angle_within_tolerance() -> None:
    result = prepared(make_page_png(angle=SKEW_DEG))

    assert STEP_DESKEWED in result.applied
    assert abs(result.skew_angle_deg - SKEW_DEG) <= SKEW_TOLERANCE_DEG


def test_deskew_is_skipped_on_straight_page() -> None:
    # Детектор сам терпит наклон в доли градуса, а лишний warpAffine стоит
    # качества штрихов.
    assert STEP_DESKEWED not in prepared(make_page_png()).applied


def test_deskew_is_skipped_when_estimates_disagree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Кривой поворот хуже, чем никакого: он размывает штрихи и при этом
    # не выпрямляет строки.
    from document_worker.infrastructure.ocr import preprocessor  # noqa: PLC0415

    monkeypatch.setattr(
        preprocessor,
        "_hough_angle",
        lambda _: DESKEW_DISAGREEMENT_LIMIT_DEG * 4,
    )

    result = prepared(make_page_png(angle=SKEW_DEG))

    assert STEP_DESKEW_UNCERTAIN in result.applied
    assert STEP_DESKEWED not in result.applied


def test_fast_profile_skips_deskew() -> None:
    result = prepared(make_page_png(angle=SKEW_DEG), PreprocessProfile.FAST)

    assert STEP_DESKEWED not in result.applied


def test_blank_page_is_prepared_without_error() -> None:
    result = prepared(blank_page_png())

    assert result.applied == ()
    assert result.image.width_px > 0


def test_transparent_background_is_composited_on_white() -> None:
    # Без композита прозрачный фон становится чёрным, и страница уезжает
    # в инверсию целиком.
    source = make_page_image().convert("RGBA")
    source.putalpha(0)
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    result = prepared(buffer.getvalue())

    assert mean_level(result.image.png) > 250


def test_transform_of_untouched_page_is_identity() -> None:
    result = prepared(make_page_png())

    assert result.transform.to_page(0.0, 0.0) == (0.0, 0.0)


def test_transform_roundtrip_returns_box_to_original_coordinates() -> None:
    # Известная точка, прошедшая поворот, обязана вернуться туда, откуда пришла:
    # иначе подсветка фрагмента укажет оператору не на тот кусок страницы.
    result = prepared(make_page_png(angle=SKEW_DEG))
    transform = result.transform
    x_page, y_page = 0.35, 0.42
    x_px, y_px = transform.to_image(x_page, y_page)

    back_x, back_y = transform.to_page(x_px, y_px)

    assert abs(back_x - x_page) <= ROUNDTRIP_TOLERANCE
    assert abs(back_y - y_page) <= ROUNDTRIP_TOLERANCE


def test_transform_keeps_page_size_of_the_source_render() -> None:
    source = page_image(make_page_png(angle=SKEW_DEG))

    result = prepare_page(source, PreprocessProfile.DEFAULT)

    assert result.transform.page_width_px == source.width_px
    assert result.transform.page_height_px == source.height_px


def test_identity_transform_maps_corners_to_unit_square() -> None:
    transform = PageTransform.identity(width_px=200, height_px=100)

    assert transform.to_page(200.0, 100.0) == (1.0, 1.0)
