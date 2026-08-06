"""Оценки релевантности и доли.

Три разных типа, потому что три разных обещания. `Score` — любое конечное
число: косинус приходит из −1..1, а логит кросс-энкодера ничем не ограничен, и
сузить диапазон значило бы отвергать честные значения. `RrfScore` строго
положителен по построению суммы `1/(k + rank)`. `Ratio` — доля 0..1, в которой
хранятся обоснованность и уверенность распознавания.

Общее у всех — запрет на значения, не являющиеся числами. NaN не больше и не
меньше ничего: попав в выдачу, он ломает сортировку молча, без единой ошибки.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

from ai_worker.domain.errors import InvalidScore


def _require_finite(value: float, *, kind: str) -> None:
    if not math.isfinite(value):
        raise InvalidScore(
            f"{kind} обязана быть конечным числом",
            context={"value": repr(value), "kind": kind},
        )


@dataclass(frozen=True, slots=True, order=True)
class Score:
    """Оценка релевантности: косинус, внутреннее произведение или логит."""

    value: float

    def __post_init__(self) -> None:
        """Требует конечное число."""
        _require_finite(self.value, kind="оценка")


@dataclass(frozen=True, slots=True, order=True)
class RrfScore:
    """Оценка слияния рангов: сумма `1/(k + rank)` по найденным ветвям."""

    value: float

    def __post_init__(self) -> None:
        """Требует строго положительное конечное число."""
        _require_finite(self.value, kind="оценка слияния")
        if self.value <= 0:
            raise InvalidScore(
                "оценка слияния обязана быть положительной: ноль означал бы "
                "попадание, не найденное ни одной ветвью поиска",
                context={"value": self.value},
            )


@dataclass(frozen=True, slots=True, order=True)
class Ratio:
    """Доля в диапазоне 0..1 включительно."""

    value: float

    def __post_init__(self) -> None:
        """Требует конечное число в пределах 0..1."""
        _require_finite(self.value, kind="доля")
        if not 0.0 <= self.value <= 1.0:
            raise InvalidScore(
                "доля вне диапазона 0..1",
                context={"value": self.value},
            )

    @classmethod
    def of(cls, *, part: int, whole: int) -> Self:
        """Доля части в целом.

        Пустое целое даёт ноль, а не ошибку деления: черновик без утверждений —
        штатный исход, и обоснованность такого черновика равна нулю.

        Raises:
            InvalidScore: Часть больше целого либо одно из чисел отрицательно.
        """
        if part < 0 or whole < 0:
            raise InvalidScore(
                "доля не считается по отрицательным числам",
                context={"part": part, "whole": whole},
            )
        if part > whole:
            raise InvalidScore(
                "часть больше целого",
                context={"part": part, "whole": whole},
            )
        return cls(0.0 if whole == 0 else part / whole)
