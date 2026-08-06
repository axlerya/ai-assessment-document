"""Доказательство: чанк вместе с тем, как он был найден.

Ранги обеих ветвей хранятся рядом с итоговой оценкой не для отчётности. По ним
видно, что именно нашло фрагмент — переформулировка или точная лексема — и что
переставил реранкер. Без этого выдачу невозможно ни объяснить, ни улучшить:
любое изменение профиля поиска оценивается вслепую.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from ai_worker.domain.errors import InvariantViolation

if TYPE_CHECKING:
    from ai_worker.domain.entities.source_chunk import ChunkRef, SourceChunk
    from ai_worker.domain.value_objects.scores import RrfScore, Score

MIN_RANK = 1


@dataclass(frozen=True, slots=True)
class Evidence:
    """Найденный чанк вместе с происхождением его места в выдаче."""

    chunk: SourceChunk
    dense_rank: int | None
    dense_score: Score | None
    sparse_rank: int | None
    sparse_score: Score | None
    rrf_score: RrfScore
    rerank_score: Score | None = None

    def __post_init__(self) -> None:
        """Требует, чтобы попадание пришло хотя бы из одной ветви поиска.

        Raises:
            InvariantViolation: Ни одна ветвь не нашла чанк, ранг меньше
                единицы либо ранг пришёл без своей оценки.
        """
        self._validate_branch(
            rank=self.dense_rank, score=self.dense_score, name="dense"
        )
        self._validate_branch(
            rank=self.sparse_rank, score=self.sparse_score, name="sparse"
        )
        if self.dense_rank is None and self.sparse_rank is None:
            # Такое попадание означает склейку по неверному ключу — то есть
            # чужой чанк в выдаче.
            raise InvariantViolation(
                "доказательство не найдено ни одной ветвью поиска",
                context={"chunk_id": str(self.chunk.ref.chunk_id)},
            )

    def _validate_branch(
        self,
        *,
        rank: int | None,
        score: Score | None,
        name: str,
    ) -> None:
        if (rank is None) != (score is None):
            raise InvariantViolation(
                f"ветвь {name}: ранг и оценка обязаны идти вместе",
                context={"branch": name, "rank": rank},
            )
        if rank is not None and rank < MIN_RANK:
            raise InvariantViolation(
                f"ветвь {name}: ранг меньше {MIN_RANK}",
                context={"branch": name, "rank": rank},
            )

    @property
    def ref(self) -> ChunkRef:
        """Координаты чанка."""
        return self.chunk.ref

    @property
    def found_by_both_branches(self) -> bool:
        """Нашли ли фрагмент и плотная, и разреженная ветви."""
        return self.dense_rank is not None and self.sparse_rank is not None

    @property
    def ordering_score(self) -> float:
        """Число, по которому фрагмент занимает своё место в выдаче.

        После реранкинга — оценка кросс-энкодера, до него — слияние рангов.
        Возвращается именно число, а не значение-объект: сортировать пришлось
        бы список, где часть элементов уже переранжирована, а часть ещё нет, и
        два разных типа оценки в одном сравнении дали бы `TypeError`.
        """
        if self.rerank_score is not None:
            return self.rerank_score.value
        return self.rrf_score.value

    def reranked(self, score: Score) -> Self:
        """Записывает оценку кросс-энкодера, сохраняя исходные ранги.

        Raises:
            InvariantViolation: Оценка уже записана — второй проход означал бы,
                что порядок собран из двух разных прогонов.
        """
        if self.rerank_score is not None:
            raise InvariantViolation(
                "доказательство уже переранжировано",
                context={"chunk_id": str(self.chunk.ref.chunk_id)},
            )
        return replace(self, rerank_score=score)
