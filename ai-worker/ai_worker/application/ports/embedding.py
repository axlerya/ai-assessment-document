"""Построение представлений текста.

Порт нейтрален к поставщику: параметр существует, только если его умеют все
запланированные реализации. Иначе абстракция ложная — сменить модель нельзя,
не переписав вызывающий код.

Запрос и фрагмент считаются разными вызовами намеренно: модели этого класса
добавляют к ним разные префиксы, и общий метод скрыл бы это различие.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Строит плотное и разреженное представления одним проходом модели."""

    async def embed_passages(
        self,
        texts: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[tuple[DenseVector, SparseVector]]:
        """Считает представления фрагментов документа."""
        ...

    async def embed_query(
        self,
        text: str,
        *,
        timeout_s: float,
    ) -> tuple[DenseVector, SparseVector]:
        """Считает представление пользовательского запроса."""
        ...
