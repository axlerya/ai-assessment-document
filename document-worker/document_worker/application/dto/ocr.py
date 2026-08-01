"""Данные распознавания, которыми адаптеры отвечают прикладному слою.

Изображение ходит между процессами PNG-байтами, а не массивом: A4 при 300 DPI
в grayscale — это 25 МБ на каждую передачу через границу процесса, тогда как
PNG той же страницы весит меньше мегабайта.

Координаты слов нормализованы к долям страницы — той же системе, в которой их
отдаёт текстовый слой: DPI рендера и применённая предобработка за границу
адаптера не протекают.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from document_worker.domain.errors import InvariantViolation

if TYPE_CHECKING:
    from document_worker.domain.value_objects.geometry import BoundingBox


class PreprocessProfile(StrEnum):
    """Насколько тщательно готовить изображение к распознаванию."""

    DEFAULT = "default"
    # Деградированный повтор после таймаута: только то, без чего распознавание
    # заведомо не сработает.
    FAST = "fast"


class ConfidenceSource(StrEnum):
    """Откуда взялась уверенность слова.

    Деградация до уверенности строки должна быть видна в данных, а не
    угадываться по косвенным признакам.
    """

    WORD = "word"
    LINE = "line"


@dataclass(frozen=True, slots=True)
class PageImage:
    """Изображение страницы в PNG."""

    number: int
    png: bytes
    width_px: int
    height_px: int
    dpi: int


@dataclass(frozen=True, slots=True)
class PageTransform:
    """Аффинное преобразование «страница → подготовленное изображение».

    Страница здесь — исходный рендер: именно к нему привязаны нормализованные
    координаты, и именно он воспроизводится из PDF детерминированно.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    page_width_px: int
    page_height_px: int

    def __post_init__(self) -> None:
        """Проверяет обратимость и размер страницы.

        Raises:
            InvariantViolation: Размер непозитивен или матрица вырождена.
        """
        if self.page_width_px <= 0 or self.page_height_px <= 0:
            raise InvariantViolation(
                "размер страницы должен быть положительным",
                context={"width": self.page_width_px, "height": self.page_height_px},
            )
        if self.determinant == 0.0:
            raise InvariantViolation(
                "вырожденное преобразование не переводится обратно",
                context={"a": self.a, "b": self.b, "d": self.d, "e": self.e},
            )

    @classmethod
    def identity(cls, *, width_px: int, height_px: int) -> Self:
        """Преобразование страницы, которую не трогали."""
        return cls(
            a=1.0,
            b=0.0,
            c=0.0,
            d=0.0,
            e=1.0,
            f=0.0,
            page_width_px=width_px,
            page_height_px=height_px,
        )

    @property
    def determinant(self) -> float:
        """Определитель линейной части."""
        return self.a * self.e - self.b * self.d

    def then(self, matrix: tuple[float, float, float, float, float, float]) -> Self:
        """Дописывает ещё одно преобразование поверх этого."""
        a, b, c, d, e, f = matrix
        return type(self)(
            a=a * self.a + b * self.d,
            b=a * self.b + b * self.e,
            c=a * self.c + b * self.f + c,
            d=d * self.a + e * self.d,
            e=d * self.b + e * self.e,
            f=d * self.c + e * self.f + f,
            page_width_px=self.page_width_px,
            page_height_px=self.page_height_px,
        )

    def to_image(self, x_page: float, y_page: float) -> tuple[float, float]:
        """Нормализованные координаты страницы → пиксели изображения."""
        x_px = x_page * self.page_width_px
        y_px = y_page * self.page_height_px
        return (
            self.a * x_px + self.b * y_px + self.c,
            self.d * x_px + self.e * y_px + self.f,
        )

    def to_page(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Пиксели изображения → нормализованные координаты страницы."""
        shifted_x = x_px - self.c
        shifted_y = y_px - self.f
        determinant = self.determinant
        x_page = (self.e * shifted_x - self.b * shifted_y) / determinant
        y_page = (self.a * shifted_y - self.d * shifted_x) / determinant
        return (x_page / self.page_width_px, y_page / self.page_height_px)


@dataclass(frozen=True, slots=True)
class PreparedPage:
    """Изображение, готовое к распознаванию, вместе с историей подготовки."""

    image: PageImage
    transform: PageTransform
    # Какие шаги реально сработали: уходит в предупреждения страницы, потому
    # что «страница была в негативе» объясняет оператору качество результата.
    applied: tuple[str, ...]
    skew_angle_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class RecognizedWordDTO:
    """Слово, выданное распознавателем, с его уверенностью и местом."""

    text: str
    confidence: float
    bbox: BoundingBox
    line_index: int
    word_index: int
    confidence_source: ConfidenceSource


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Итог распознавания одной страницы.

    Пустой список слов — валидный результат, а не ошибка: дальше его разбирает
    политика читаемости.
    """

    words: tuple[RecognizedWordDTO, ...]
    line_count: int
    # Нужна для решения об эскалации DPI: если строка и так высокая, рост
    # разрешения распознаванию уже ничего не даёт.
    median_line_height_px: float
    engine_version: str
    elapsed_ms: int
