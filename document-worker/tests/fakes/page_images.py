"""Синтетические изображения страниц для препроцессинга и распознавания.

Текст латинский: встроенный в Pillow шрифт кириллицы не содержит, а тянуть
системный означало бы разный результат на разных машинах. Проверяемая здесь
механика — боксы, уверенность, наклон, инверсия — от письменности не зависит.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Final

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Sequence

# A4 при 150 DPI: достаточно, чтобы строки были выше тридцати пикселей, и
# вчетверо быстрее полноразмерного рендера.
PAGE_WIDTH: Final[int] = 1240
PAGE_HEIGHT: Final[int] = 1754
FONT_SIZE: Final[int] = 44
LINE_STEP: Final[int] = 90
MARGIN: Final[int] = 120

DEFAULT_LINES: Final[tuple[str, ...]] = (
    "SUPPLY CONTRACT No 42",
    "The Supplier undertakes to deliver",
    "the goods within thirty days.",
)


def make_page_image(
    *,
    lines: Sequence[str] = DEFAULT_LINES,
    width: int = PAGE_WIDTH,
    height: int = PAGE_HEIGHT,
    angle: float = 0.0,
    inverted: bool = False,
) -> Image.Image:
    """Страница с текстом, при желании наклонённая или в негативе."""
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=FONT_SIZE)
    for index, line in enumerate(lines):
        draw.text((MARGIN, MARGIN + index * LINE_STEP), line, fill=0, font=font)
    if angle:
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    if inverted:
        image = Image.eval(image, lambda value: 255 - value)
    return image


def make_page_png(**kwargs: object) -> bytes:
    """То же изображение в PNG-байтах — так оно ходит между процессами."""
    return encode_png(make_page_image(**kwargs))  # type: ignore[arg-type]


def blank_page_png(*, width: int = 400, height: int = 400) -> bytes:
    """Пустая белая страница: распознавать нечего, но это не ошибка."""
    return encode_png(Image.new("L", (width, height), color=255))


def encode_png(image: Image.Image) -> bytes:
    """Кодирует изображение без метаданных — байты воспроизводимы."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()
