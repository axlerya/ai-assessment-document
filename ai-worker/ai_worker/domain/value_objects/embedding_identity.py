"""Чем именно построен вектор.

Версия и имя модели всегда ходят вместе: версия открывает namespace, имя
объясняет, что в нём лежит. Порознь они бесполезны — версия без модели не
говорит, чем считали, а модель без версии не отличает два прогона с разной
нормализацией. Поэтому это одно значение, а не два поля рядом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from ai_worker.domain.errors import InvariantViolation

if TYPE_CHECKING:
    from ai_worker.domain.value_objects.versioning import EmbeddingVersion


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Версия эмбеддингов вместе с моделью, которая их считает."""

    version: EmbeddingVersion
    model_name: str

    def __post_init__(self) -> None:
        """Требует имя модели.

        Raises:
            InvariantViolation: Имя пусто — происхождение вектора восстановить
                будет нечем, и решить, нужна ли переиндексация, тоже.
        """
        if not self.model_name.strip():
            raise InvariantViolation(
                "версия эмбеддингов без имени модели не объясняет вектор",
                context={"version": str(self.version)},
            )

    @override
    def __str__(self) -> str:
        return f"{self.model_name}@{self.version}"
