"""Уверенность распознавания.

Константы «достоверно» здесь нет: у текстового слоя confidence не единица,
а отсутствует — это выражается на уровне RecognizedText.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from document_worker.domain.constants import CONFIDENCE_PRECISION
from document_worker.domain.errors import InvalidConfidence

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True, order=True)
class OcrConfidence:
    """Уверенность распознавания в диапазоне 0..1."""

    value: float

    ZERO: ClassVar[OcrConfidence]

    def __post_init__(self) -> None:
        """Проверяет диапазон и округляет значение."""
        if not math.isfinite(self.value):
            raise InvalidConfidence(
                "уверенность не является конечным числом",
                context={"value": repr(self.value)},
            )
        if not 0.0 <= self.value <= 1.0:
            raise InvalidConfidence(
                "уверенность вне диапазона 0..1",
                context={"value": self.value},
            )
        object.__setattr__(self, "value", round(self.value, CONFIDENCE_PRECISION))

    def is_below(self, threshold: float) -> bool:
        """Строго ли уверенность ниже порога."""
        return self.value < threshold

    @staticmethod
    def weighted_mean(
        pairs: Sequence[tuple[OcrConfidence, int]],
    ) -> OcrConfidence | None:
        """Среднее, взвешенное по длине фрагментов.

        Возвращает None при пустом входе или нулевой сумме весов — это штатный
        случай «в документе нет OCR-страниц», а не ошибка.

        Raises:
            InvalidConfidence: Отрицательный вес.
        """
        total_weight = 0
        weighted_sum = 0.0
        for confidence, weight in pairs:
            if weight < 0:
                raise InvalidConfidence(
                    "вес фрагмента отрицателен",
                    context={"weight": weight},
                )
            total_weight += weight
            weighted_sum += confidence.value * weight
        if total_weight == 0:
            return None
        return OcrConfidence(weighted_sum / total_weight)


OcrConfidence.ZERO = OcrConfidence(0.0)
