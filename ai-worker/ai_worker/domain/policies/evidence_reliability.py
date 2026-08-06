"""Можно ли опереть утверждение на этот фрагмент.

Скор фрагмента здесь не трогается намеренно. Понизить ранг ненадёжного чанка
означало бы замаскировать проблему: он всё равно попадёт в контекст, только
объяснить его место в выдаче станет нечем. Вместо этого политика ставит метку,
а решение принимает обоснованность утверждения: фрагмент видно оператору, но
единственной опорой утверждения он быть не может.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_worker.domain.entities.source_chunk import ChunkQuality


@dataclass(frozen=True, slots=True)
class ReliabilityVerdict:
    """Пригоден ли фрагмент как опора утверждения и почему нет."""

    reliable: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceReliabilityPolicy:
    """Порог, ниже которого фрагмент нельзя считать подтверждением."""

    min_citable_confidence: float

    def __post_init__(self) -> None:
        """Требует порог в пределах доли.

        Raises:
            ValueError: Порог вне 0..1 — сравнивать с ним уверенность
                бессмысленно.
        """
        if not 0.0 <= self.min_citable_confidence <= 1.0:
            raise ValueError(
                f"порог цитируемости {self.min_citable_confidence} вне 0..1"
            )

    def judge(self, quality: ChunkQuality) -> ReliabilityVerdict:
        """Выносит вердикт по признакам качества чанка."""
        if quality.has_illegible:
            # Неразборчивое место внутри фрагмента означает, что часть текста
            # не прочитана, а какая именно — неизвестно.
            return ReliabilityVerdict(reliable=False, reason="illegible_spans")
        confidence = quality.avg_confidence
        if confidence is not None and confidence.value < self.min_citable_confidence:
            return ReliabilityVerdict(reliable=False, reason="low_confidence")
        return ReliabilityVerdict(reliable=True)
