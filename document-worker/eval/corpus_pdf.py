"""Рендер страниц корпуса: текстовый слой и сканы с управляемой деградацией.

Всё, что влияет на байты, закреплено: формат страницы, поля, кегль, интерлиньяж,
шрифт, разрешение, углы наклона, дисперсия шума, качество JPEG и дата создания
документа. Незакреплённым не остаётся ничего — иначе `corpus_hash` перестал бы
сходиться между машинами, а вместе с ним отключилась бы защита корпуса.

Случайность здесь только псевдо: у каждого документа свой seed, и от него
зависят и шум, и наклон, и штрихи нечитаемой вставки.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from eval.corpus_text import PageContent

PAGE_WIDTH_PT: Final[float] = 595.28
PAGE_HEIGHT_PT: Final[float] = 841.89
MARGIN_PT: Final[float] = 56.0
BODY_PT: Final[int] = 11
HEADING_PT: Final[int] = 13
LINE_PT: Final[float] = 15.0
BLOCK_GAP_PT: Final[float] = 9.0
# Дата документа фиксирована: fpdf кладёт её в PDF, и «сейчас» сделало бы
# каждый прогон новым корпусом.
CREATED_AT: Final[datetime] = datetime(2026, 3, 12, 9, 0, tzinfo=UTC)

PageMode = Literal["text", "scan"]
SCRIBBLE_STROKES: Final[int] = 220
STAMP_TEXT: Final[str] = "КОПИЯ ВЕРНА"


@dataclass(frozen=True, slots=True)
class Degradation:
    """Во что превращается страница по дороге в скан."""

    dpi: int
    skew_deg: float
    noise_sigma: float
    jpeg_quality: int


@dataclass(frozen=True, slots=True)
class PagePlan:
    """Как рисовать одну страницу."""

    mode: PageMode
    degradation: Degradation | None = None
    unreadable: bool = False
    stamped: bool = False
    columns: int = 1


def render(
    path: Path,
    pages: Sequence[PageContent],
    plans: Sequence[PagePlan],
    *,
    fonts: Mapping[str, Path],
    seed: int,
) -> None:
    """Собирает PDF из страниц по их планам."""
    pdf = FPDF(unit="pt", format=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    pdf.set_creation_date(CREATED_AT)
    pdf.set_auto_page_break(auto=False)
    pdf.add_font("body", "", str(fonts["sans"]))
    pdf.add_font("body", "B", str(fonts["sans-bold"]))
    for index, (page, plan) in enumerate(zip(pages, plans, strict=True)):
        pdf.add_page()
        if plan.mode == "text":
            _draw_text_layer(pdf, page, plan)
        else:
            _place_scan(pdf, page, plan, fonts=fonts, seed=seed + index)
    pdf.output(str(path))


def _draw_text_layer(pdf: FPDF, page: PageContent, plan: PagePlan) -> None:
    if plan.columns > 1:
        _draw_columns(pdf, page, plan.columns)
        return
    pdf.set_xy(MARGIN_PT, MARGIN_PT)
    width = PAGE_WIDTH_PT - 2 * MARGIN_PT
    for block in page.blocks:
        _set_face(pdf, block.kind)
        for line in block.lines:
            pdf.set_x(MARGIN_PT)
            pdf.multi_cell(width, LINE_PT, line, align="L")
        pdf.set_y(pdf.get_y() + BLOCK_GAP_PT)


def _draw_columns(pdf: FPDF, page: PageContent, columns: int) -> None:
    gutter = 18.0
    width = (PAGE_WIDTH_PT - 2 * MARGIN_PT - gutter * (columns - 1)) / columns
    per_column = max(1, len(page.blocks) // columns + len(page.blocks) % columns)
    for column in range(columns):
        left = MARGIN_PT + column * (width + gutter)
        pdf.set_xy(left, MARGIN_PT)
        for block in page.blocks[column * per_column : (column + 1) * per_column]:
            _set_face(pdf, block.kind)
            for line in block.lines:
                pdf.set_x(left)
                pdf.multi_cell(width, LINE_PT, line, align="L")
            pdf.set_y(pdf.get_y() + BLOCK_GAP_PT)


def _set_face(pdf: FPDF, kind: str) -> None:
    if kind == "heading":
        pdf.set_font("body", "B", HEADING_PT)
    else:
        pdf.set_font("body", "", BODY_PT)


def _place_scan(
    pdf: FPDF,
    page: PageContent,
    plan: PagePlan,
    *,
    fonts: Mapping[str, Path],
    seed: int,
) -> None:
    degradation = plan.degradation
    if degradation is None:  # pragma: no cover — план скана без деградации не строится
        message = "скану нужны параметры деградации"
        raise ValueError(message)
    image = _raster(page, plan, fonts=fonts, degradation=degradation, seed=seed)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=degradation.jpeg_quality, optimize=False)
    buffer.seek(0)
    pdf.image(buffer, x=0, y=0, w=PAGE_WIDTH_PT, h=PAGE_HEIGHT_PT)


def _raster(
    page: PageContent,
    plan: PagePlan,
    *,
    fonts: Mapping[str, Path],
    degradation: Degradation,
    seed: int,
) -> Image.Image:
    scale = degradation.dpi / 72.0
    size = (round(PAGE_WIDTH_PT * scale), round(PAGE_HEIGHT_PT * scale))
    canvas = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(canvas)
    body = ImageFont.truetype(str(fonts["sans"]), round(BODY_PT * scale))
    heading = ImageFont.truetype(str(fonts["sans-bold"]), round(HEADING_PT * scale))

    bottom = _draw_blocks(draw, page, plan, body=body, heading=heading, scale=scale)
    if plan.unreadable:
        _scribble(draw, bottom, size, seed=seed, scale=scale)
    if plan.stamped:
        _stamp(draw, size, font=heading)
    return _degrade(canvas, degradation, seed=seed)


def _draw_blocks(  # noqa: PLR0913 — рисование зависит от всех этих величин
    draw: ImageDraw.ImageDraw,
    page: PageContent,
    plan: PagePlan,
    *,
    body: ImageFont.FreeTypeFont,
    heading: ImageFont.FreeTypeFont,
    scale: float,
) -> float:
    left = MARGIN_PT * scale
    width = (PAGE_WIDTH_PT - 2 * MARGIN_PT) * scale
    top = MARGIN_PT * scale
    blocks = page.blocks[:-1] if plan.unreadable else page.blocks
    for block in blocks:
        font = heading if block.kind == "heading" else body
        for line in block.lines:
            for wrapped in _wrap(line, font, width):
                draw.text((left, top), wrapped, font=font, fill=0)
                top += LINE_PT * scale
        top += BLOCK_GAP_PT * scale
    return top


def _wrap(line: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    words = line.split(" ")
    rows: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > width:
            rows.append(current)
            current = word
        else:
            current = candidate
    rows.append(current)
    return rows


def _scribble(
    draw: ImageDraw.ImageDraw,
    top: float,
    size: tuple[int, int],
    *,
    seed: int,
    scale: float,
) -> None:
    """Рисует «от руки»: линии, в которых нет ни одного настоящего глифа.

    Именно на этой вставке проверяется честность сервиса: распознать её нечем,
    и единственный правильный исход — пометить диапазон неразборчивым.
    """
    rng = np.random.default_rng(seed)
    left = MARGIN_PT * scale
    right = size[0] - MARGIN_PT * scale
    bottom = min(size[1] - MARGIN_PT * scale, top + 220 * scale)
    xs = rng.uniform(left, right, size=(SCRIBBLE_STROKES, 2))
    ys = rng.uniform(top, bottom, size=(SCRIBBLE_STROKES, 2))
    for index in range(SCRIBBLE_STROKES):
        draw.line(
            (xs[index, 0], ys[index, 0], xs[index, 1], ys[index, 1]),
            fill=90,
            width=max(1, round(scale)),
        )


def _stamp(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    *,
    font: ImageFont.FreeTypeFont,
) -> None:
    box = min(size) // 3
    left = size[0] - box - size[0] // 12
    top = size[1] // 2
    draw.ellipse((left, top, left + box, top + box // 2), outline=110, width=3)
    draw.text((left + box // 8, top + box // 6), STAMP_TEXT, font=font, fill=110)


def _degrade(
    canvas: Image.Image,
    degradation: Degradation,
    *,
    seed: int,
) -> Image.Image:
    skewed = canvas
    if degradation.skew_deg:
        skewed = canvas.rotate(
            degradation.skew_deg, resample=Image.Resampling.BICUBIC, fillcolor=255
        )
    if not degradation.noise_sigma:
        return skewed
    rng = np.random.default_rng(seed + 1)
    pixels = np.asarray(skewed, dtype=np.int16)
    noise = rng.normal(0.0, degradation.noise_sigma, size=pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8), mode="L")
