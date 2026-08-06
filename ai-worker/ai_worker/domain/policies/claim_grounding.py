"""Подтверждено ли утверждение.

Порядок проверок не случаен: он идёт от самого грубого нарушения к самому
тонкому. Утверждение без цитат — модель просто не сослалась. Цитата на чанк вне
контекста — модель сослалась на документ, которого не видела: самый
правдоподобный и потому самый опасный случай, потому что ссылка выглядит
настоящей. Только ненадёжные цитаты — сослалась честно, но на текст, который
сам прочитан неуверенно.

Политика возвращает вердикт и не строит утверждение: сущность, порождённая
политикой, рано или поздно окажется такой, которую эта же сущность обязана
отвергнуть.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_worker.domain.value_objects.enums import RejectCode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_worker.domain.entities.draft import Citation
    from ai_worker.domain.value_objects.identifiers import ChunkId


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """Попадает ли утверждение в тело черновика и почему нет."""

    supported: bool
    reject_code: RejectCode | None = None


@dataclass(frozen=True, slots=True)
class ClaimGroundingPolicy:
    """Правило, по которому утверждение считается подтверждённым."""

    def judge(
        self,
        *,
        citations: Sequence[Citation],
        context_chunk_ids: frozenset[ChunkId],
    ) -> ClaimVerdict:
        """Выносит вердикт по цитатам утверждения и составу контекста."""
        if not citations:
            return ClaimVerdict(supported=False, reject_code=RejectCode.NO_CITATION)
        outside = [
            citation
            for citation in citations
            if citation.ref.chunk_id not in context_chunk_ids
        ]
        if outside:
            return ClaimVerdict(
                supported=False, reject_code=RejectCode.CHUNK_NOT_IN_CONTEXT
            )
        if not any(citation.reliable for citation in citations):
            return ClaimVerdict(
                supported=False, reject_code=RejectCode.UNRELIABLE_EVIDENCE_ONLY
            )
        return ClaimVerdict(supported=True)
