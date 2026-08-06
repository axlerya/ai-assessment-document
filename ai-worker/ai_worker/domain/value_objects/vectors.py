"""Плотное и разреженное представления чанка.

Обе величины приходят из одного прогона `bge-m3` и хранятся рядом: плотное
ловит переформулировку, разреженное — точную лексему вроде номера договора или
суммы. Ни одно из них по отдельности не закрывает юридический документ.

Границы заданы хранилищем, а не вкусом: ширина плотного вектора — ширина
колонки `vector(1024)`, предел разреженного — предел HNSW в pgvector. Нарушить
их можно только один раз, и узнать об этом на записи, а не на построении,
значит потерять весь прогон индексации документа.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from ai_worker.domain.constants import (
    DENSE_DIMENSIONS,
    SPARSE_TOP_K,
    SPARSE_VOCABULARY_SIZE,
)
from ai_worker.domain.errors import InvalidVector

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class DenseVector:
    """Плотное представление чанка фиксированной ширины."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Сверяет ширину и требует, чтобы все значения были числами.

        Raises:
            InvalidVector: Ширина не совпадает с колонкой либо среди значений
                есть NaN или бесконечность — с ними любое расстояние
                становится NaN, и поиск молча возвращает произвольный порядок.
        """
        if len(self.values) != DENSE_DIMENSIONS:
            raise InvalidVector(
                f"ширина плотного вектора {len(self.values)} вместо {DENSE_DIMENSIONS}",
                context={"actual": len(self.values), "expected": DENSE_DIMENSIONS},
            )
        for position, value in enumerate(self.values):
            if not math.isfinite(value):
                raise InvalidVector(
                    "плотный вектор содержит значение, не являющееся числом",
                    context={"position": position, "value": repr(value)},
                )


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Разреженное представление чанка: веса токенов по возрастанию индекса.

    Порядок канонический — его требует литерал `sparsevec`, и он же избавляет
    адаптер хранилища от собственной сортировки.
    """

    weights: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        """Проверяет непустоту, предел, порядок и допустимость каждого веса.

        Raises:
            InvalidVector: Вектор пуст, длиннее предела индекса, не упорядочен
                по индексу либо содержит недопустимый индекс или вес.
        """
        if not self.weights:
            self._reject("разреженный вектор пуст", context={})
        if len(self.weights) > SPARSE_TOP_K:
            self._reject(
                f"ненулевых весов {len(self.weights)} при пределе {SPARSE_TOP_K}",
                context={"count": len(self.weights), "limit": SPARSE_TOP_K},
            )
        previous = -1
        for index, weight in self.weights:
            _validate_pair(index, weight)
            if index <= previous:
                self._reject(
                    "веса не упорядочены по возрастанию индекса",
                    context={"index": index, "previous": previous},
                )
            previous = index

    @staticmethod
    def _reject(reason: str, *, context: Mapping[str, object]) -> None:
        raise InvalidVector(reason, context=context)

    @classmethod
    def pruned(
        cls,
        weights: Mapping[int, float],
        *,
        limit: int = SPARSE_TOP_K,
    ) -> Self:
        """Оставляет самые тяжёлые веса и укладывает их в канонический порядок.

        Обрезка детерминирована: при равных весах порядок задаётся возрастанием
        индекса. Без этого правила один и тот же чанк давал бы разные векторы
        между прогонами, и оценка качества перестала бы быть воспроизводимой.
        """
        for index, weight in weights.items():
            _validate_pair(index, weight)
        heaviest = sorted(weights.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
        return cls(tuple(sorted(heaviest)))


def _validate_pair(index: int, weight: float) -> None:
    if not 0 <= index < SPARSE_VOCABULARY_SIZE:
        raise InvalidVector(
            f"индекс токена вне словаря 0..{SPARSE_VOCABULARY_SIZE - 1}",
            context={"index": index},
        )
    if not math.isfinite(weight):
        raise InvalidVector(
            "вес токена не является числом",
            context={"index": index, "weight": repr(weight)},
        )
    if weight <= 0:
        # Разреженный выход модели даёт только положительные веса; ноль означал
        # бы токен, которого в чанке нет, и занимал бы место до предела впустую.
        raise InvalidVector(
            "вес токена не положителен",
            context={"index": index, "weight": weight},
        )
