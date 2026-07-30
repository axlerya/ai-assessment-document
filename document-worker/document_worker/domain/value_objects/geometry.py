"""Ограничивающий прямоугольник в нормализованных координатах.

Координаты — доли страницы 0..1, поэтому точки PDF и пиксели рендера
сравнимы, а DPI не протекает в домен.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from document_worker.domain.errors import InvalidBoundingBox


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Прямоугольник на странице, координаты 0..1."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        """Проверяет диапазон координат и непустоту площади."""
        for name, value in (
            ("x0", self.x0),
            ("y0", self.y0),
            ("x1", self.x1),
            ("y1", self.y1),
        ):
            if not 0.0 <= value <= 1.0:
                raise InvalidBoundingBox(
                    f"координата {name} вне 0..1",
                    context={"coordinate": name, "value": value},
                )
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise InvalidBoundingBox(
                "прямоугольник вырожден",
                context={"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1},
            )

    @property
    def width(self) -> float:
        """Ширина в долях страницы."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Высота в долях страницы."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Площадь в долях страницы."""
        return self.width * self.height

    def intersects(self, other: BoundingBox) -> bool:
        """Пересекаются ли прямоугольники. Касание пересечением не считается."""
        return (
            self.x0 < other.x1
            and other.x0 < self.x1
            and self.y0 < other.y1
            and other.y0 < self.y1
        )

    def union(self, other: BoundingBox) -> BoundingBox:
        """Наименьший прямоугольник, покрывающий оба."""
        return BoundingBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def to_pixels(self, page_w: int, page_h: int) -> tuple[int, int, int, int]:
        """Переводит координаты в пиксели страницы заданного размера."""
        return (
            round(self.x0 * page_w),
            round(self.y0 * page_h),
            round(self.x1 * page_w),
            round(self.y1 * page_h),
        )

    @classmethod
    def from_pixels(  # noqa: PLR0913 — прямоугольник и размер страницы это 6 чисел
        cls,
        *,
        x: float,
        y: float,
        w: float,
        h: float,
        page_w: int,
        page_h: int,
    ) -> Self:
        """Строит прямоугольник из пиксельных координат и размеров страницы.

        Raises:
            InvalidBoundingBox: Неположительный размер страницы.
        """
        if page_w <= 0 or page_h <= 0:
            raise InvalidBoundingBox(
                "размер страницы должен быть положительным",
                context={"page_w": page_w, "page_h": page_h},
            )
        return cls(x / page_w, y / page_h, (x + w) / page_w, (y + h) / page_h)
