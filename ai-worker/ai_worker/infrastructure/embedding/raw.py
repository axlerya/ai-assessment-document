"""Сырой выход модели: то, что рабочий процесс отдаёт через границу pickle.

Доменных величин здесь нет намеренно. Проверки `DenseVector` и `SparseVector`
принадлежат домену, а не рабочему процессу: собранный там объект пришлось бы
пересобирать после распаковки, и нарушение инварианта всплыло бы в чужом
процессе, где его некому объяснить.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class RawEmbedding:
    """Плотный вектор без нормировки и веса токенов без обрезки."""

    dense: tuple[float, ...]
    sparse: Mapping[int, float]
