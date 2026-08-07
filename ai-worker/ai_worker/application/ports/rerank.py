"""Переупорядочивание найденного."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_worker.domain.value_objects.scores import Score


@runtime_checkable
class Reranker(Protocol):
    """Оценивает пару «запрос, фрагмент» и даёт новую оценку."""

    async def score(
        self,
        query: str,
        passages: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[Score]:
        """Возвращает оценку каждого фрагмента в порядке поступления.

        Порядок сохраняется, а не сортируется: сопоставить оценки с их
        фрагментами обязан вызывающий, у которого есть их происхождение.
        """
        ...
