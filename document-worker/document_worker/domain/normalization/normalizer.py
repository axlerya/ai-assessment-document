"""Конвейер нормализации текста страницы.

Порядок правил зафиксирован: замена омоглифов идёт после склейки строк, чтобы
работать с уже собранными токенами, а схлопывание пробелов — последним.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.normalization.offsets import OffsetMap
from document_worker.domain.normalization.rules import RULES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.normalization.rules import NormalizationRule
    from document_worker.domain.value_objects.enums import ExtractionMethod


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Канонический текст и карта, ведущая к нему от исходного."""

    content: str
    offsets: OffsetMap


@dataclass(frozen=True, slots=True)
class TextNormalizer:
    """Применяет правила по очереди, складывая карты смещений."""

    rules: Sequence[NormalizationRule] = RULES

    def normalize(self, text: str, *, source: ExtractionMethod) -> NormalizedText:
        """Нормализует текст страницы, полученный указанным способом."""
        content = text
        offsets = OffsetMap.identity(len(text))
        for rule in self.rules:
            if rule.applies_to_ocr_only and not source.is_ocr_based:
                continue
            content, step = rule.apply(content)
            offsets = offsets.compose(step)
        return NormalizedText(content=content, offsets=offsets)
