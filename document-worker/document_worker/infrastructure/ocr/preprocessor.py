"""Предобработка изображения страницы.

Каждый шаг условный. «Всегда всё» ухудшает чистые сканы: нейросетевой
распознаватель обучен на изображениях с антиалиасингом, и агрессивная чистка
съедает признаки, на которые он опирается. Поэтому по умолчанию в движок
уходит grayscale, а не бинаризованная картинка.

Наклон оценивается двумя независимыми способами, и при их расхождении поворот
не применяется вовсе: кривой поворот хуже, чем никакого — он размывает штрихи
и при этом не выпрямляет строки.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from document_worker.application.dto.ocr import (
    PageImage,
    PageTransform,
    PreparedPage,
    PreprocessProfile,
)
from document_worker.application.errors import CorruptedPageImageError

if TYPE_CHECKING:
    from document_worker.infrastructure.cpu.executor import CpuPool

STEP_INVERTED: Final[str] = "inverted"
STEP_DESKEWED: Final[str] = "deskewed"
STEP_DESKEW_UNCERTAIN: Final[str] = "deskew_uncertain"

# Доля тёмных пикселей, выше которой страница считается негативом.
INVERSION_DARK_SHARE: Final[float] = 0.60
# Ниже этого наклона поворот не окупается: детектор выдаёт четырёхугольники и
# сам терпит доли градуса, а лишний warpAffine стоит качества штрихов.
MIN_DESKEW_ANGLE_DEG: Final[float] = 0.5
# Выше этого — уже не наклон сканирования, а повёрнутая страница или ошибка
# оценки; такой поворот только испортит изображение.
MAX_DESKEW_ANGLE_DEG: Final[float] = 15.0
DESKEW_DISAGREEMENT_LIMIT_DEG: Final[float] = 1.0

_CONTOUR_MIN_AREA: Final[int] = 100
_CONTOUR_MIN_ASPECT: Final[float] = 3.0
_DILATE_KERNEL: Final[tuple[int, int]] = (15, 3)
_HOUGH_THRESHOLD: Final[int] = 100
_HOUGH_MAX_GAP: Final[int] = 20
_HOUGH_MAX_ANGLE_DEG: Final[float] = 15.0
_RIGHT_ANGLE_DEG: Final[float] = 45.0
_WHITE: Final[int] = 255


def prepare_page(image: PageImage, profile: PreprocessProfile) -> PreparedPage:
    """Готовит изображение к распознаванию. Выполняется в рабочем процессе.

    Raises:
        CorruptedPageImageError: Байты страницы не разбираются как изображение.
    """
    gray = _decode_on_white(image.png, number=image.number)
    applied: list[str] = []
    if _is_inverted(gray):
        gray = cv2.bitwise_not(gray)
        applied.append(STEP_INVERTED)
    transform = PageTransform.identity(
        width_px=image.width_px, height_px=image.height_px
    )
    angle = 0.0
    if profile is PreprocessProfile.DEFAULT:
        gray, transform, angle = _deskewed(gray, transform, applied)
    return PreparedPage(
        image=_encode(gray, number=image.number, dpi=image.dpi),
        transform=transform,
        applied=tuple(applied),
        skew_angle_deg=angle,
    )


@dataclass(frozen=True, slots=True)
class OpenCvImagePreprocessor:
    """Предобработка поверх OpenCV, выполняемая в пуле процессов."""

    pool: CpuPool

    async def prepare(
        self,
        image: PageImage,
        *,
        profile: PreprocessProfile,
    ) -> PreparedPage:
        """Возвращает подготовленное изображение и его преобразование."""
        return await self.pool.run(prepare_page, image, profile)


def _decode_on_white(png: bytes, *, number: int) -> np.ndarray:
    """Разбирает PNG в grayscale, положив прозрачность на белое.

    Без композита прозрачный фон становится чёрным, и все последующие оценки
    контраста инвертируются.
    """
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise CorruptedPageImageError(
            "изображение страницы не разбирается", page_number=number
        )
    if decoded.ndim == 2:  # noqa: PLR2004 — двумерный массив это уже grayscale
        return decoded
    if decoded.shape[2] == 4:  # noqa: PLR2004 — четвёртый канал это альфа
        alpha = decoded[:, :, 3:4].astype(np.float32) / _WHITE
        color = decoded[:, :, :3].astype(np.float32)
        decoded = (color * alpha + _WHITE * (1.0 - alpha)).astype(np.uint8)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)


def _is_inverted(gray: np.ndarray) -> bool:
    """Ловит негативы и бланки «белым по чёрному»."""
    threshold, _ = cv2.threshold(gray, 0, _WHITE, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_share = float(np.count_nonzero(gray < threshold)) / gray.size
    return dark_share > INVERSION_DARK_SHARE


def _deskewed(
    gray: np.ndarray,
    transform: PageTransform,
    applied: list[str],
) -> tuple[np.ndarray, PageTransform, float]:
    contour = _contour_angle(gray)
    hough = _hough_angle(gray)
    if contour is None or hough is None:
        return gray, transform, 0.0
    if abs(contour - hough) > DESKEW_DISAGREEMENT_LIMIT_DEG:
        applied.append(STEP_DESKEW_UNCERTAIN)
        return gray, transform, 0.0
    angle = (contour + hough) / 2
    if not MIN_DESKEW_ANGLE_DEG <= abs(angle) <= MAX_DESKEW_ANGLE_DEG:
        return gray, transform, angle
    rotated, matrix = _rotate(gray, angle)
    applied.append(STEP_DESKEWED)
    return rotated, transform.then(matrix), angle


def _contour_angle(gray: np.ndarray) -> float | None:
    """Медиана углов текстовых строк, взвешенная по их ширине.

    Широкая строка надёжнее короткой: у неё длиннее база, на которой меряется
    наклон.
    """
    mask = _ink_mask(gray)
    dilated = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_RECT, _DILATE_KERNEL)
    )
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    samples: list[tuple[float, float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < _CONTOUR_MIN_AREA:
            continue
        (_, _), (width, height), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(width, height), min(width, height)
        if short_side <= 0 or long_side / short_side < _CONTOUR_MIN_ASPECT:
            continue
        samples.append((_normalized(angle if width >= height else angle + 90), width))
    return _weighted_median(samples)


def _hough_angle(gray: np.ndarray) -> float | None:
    """Медиана углов прямых, найденных преобразованием Хафа."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=_HOUGH_MAX_GAP,
    )
    if lines is None:
        return None
    angles = [
        angle
        for x0, y0, x1, y1 in lines[:, 0]
        if abs(angle := float(np.degrees(np.arctan2(y1 - y0, x1 - x0))))
        < _HOUGH_MAX_ANGLE_DEG
    ]
    return float(np.median(angles)) if angles else None


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, _WHITE, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _normalized(angle: float) -> float:
    """Приводит угол минимального прямоугольника к диапазону (-45, 45]."""
    while angle > _RIGHT_ANGLE_DEG:
        angle -= 2 * _RIGHT_ANGLE_DEG
    while angle <= -_RIGHT_ANGLE_DEG:
        angle += 2 * _RIGHT_ANGLE_DEG
    return angle


def _weighted_median(samples: list[tuple[float, float]]) -> float | None:
    if not samples:
        return None
    samples.sort()
    half = sum(weight for _, weight in samples) / 2
    running = 0.0
    for angle, weight in samples:
        running += weight
        if running >= half:
            return angle
    return samples[-1][0]  # pragma: no cover — сумма весов всегда достигает половины


def _rotate(
    gray: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    height, width = gray.shape[:2]
    centre = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_WHITE,
    )
    values = tuple(float(value) for value in matrix.reshape(-1))
    return rotated, (values[0], values[1], values[2], values[3], values[4], values[5])


def _encode(gray: np.ndarray, *, number: int, dpi: int) -> PageImage:
    success, buffer = cv2.imencode(".png", gray)
    if not success:  # pragma: no cover — grayscale-массив кодируется всегда
        raise CorruptedPageImageError(
            "подготовленное изображение не кодируется", page_number=number
        )
    height, width = gray.shape[:2]
    return PageImage(
        number=number,
        png=buffer.tobytes(),
        width_px=int(width),
        height_px=int(height),
        dpi=dpi,
    )
