"""Черновик со всем содержимым и история поиска.

Черновик, его утверждения и цитаты пишутся одной операцией: черновик без
цитат — это ровно то, что ТЗ считает провалом, и допускать такое промежуточное
состояние нельзя даже на время транзакции.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ai_worker.domain.constants import NS_AIWORKER
from ai_worker.infrastructure.persistence.mappers.draft import (
    draft_to_rows,
    rows_to_draft,
)
from ai_worker.infrastructure.persistence.models.draft import (
    DraftCitationRow,
    DraftClaimRow,
    DraftRow,
)
from ai_worker.infrastructure.persistence.models.retrieval import (
    RetrievalHitRow,
    RetrievalRunRow,
)
from ai_worker.infrastructure.persistence.repositories.base import (
    SqlAlchemyRepository,
    values_of,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_worker.application.dto.retrieval import RetrievalHitDTO, RetrievalRunDTO
    from ai_worker.domain.entities.draft import Draft
    from ai_worker.domain.value_objects.identifiers import DraftId


class SqlAlchemyDraftRepository(SqlAlchemyRepository):
    """Черновик вместе с утверждениями и цитатами."""

    async def save(self, draft: Draft) -> None:
        """Пишет черновик и всё его содержимое."""
        rows = draft_to_rows(draft)
        await self._execute(pg_insert(DraftRow).values(values_of(rows.draft)))
        if rows.claims:
            await self._execute(
                pg_insert(DraftClaimRow).values(
                    [values_of(claim) for claim in rows.claims]
                )
            )
        if rows.citations:
            await self._execute(
                pg_insert(DraftCitationRow).values(
                    [values_of(citation) for citation in rows.citations]
                )
            )

    async def get(self, draft_id: DraftId) -> Draft | None:
        """Читает черновик со всеми утверждениями и цитатами."""
        row = (
            (await self._execute(select(DraftRow).where(DraftRow.id == draft_id.value)))
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        claims = [
            DraftClaimRow(**mapping)
            for mapping in (
                await self._execute(
                    select(DraftClaimRow).where(
                        DraftClaimRow.draft_id == draft_id.value
                    )
                )
            )
            .mappings()
            .all()
        ]
        citations = [
            DraftCitationRow(**mapping)
            for mapping in (
                await self._execute(
                    select(DraftCitationRow).where(
                        DraftCitationRow.draft_id == draft_id.value
                    )
                )
            )
            .mappings()
            .all()
        ]
        return rows_to_draft(DraftRow(**row), claims, citations)


class SqlAlchemyRetrievalHistoryRepository(SqlAlchemyRepository):
    """История поиска: прогон и его попадания."""

    async def record(
        self,
        run: RetrievalRunDTO,
        hits: Sequence[RetrievalHitDTO],
    ) -> None:
        """Сохраняет прогон вместе с его выдачей."""
        await self._execute(pg_insert(RetrievalRunRow).values(values_of(_run_row(run))))
        if hits:
            await self._execute(
                pg_insert(RetrievalHitRow).values(
                    [values_of(_hit_row(run.run_id, hit)) for hit in hits]
                )
            )


def _run_row(run: RetrievalRunDTO) -> RetrievalRunRow:
    return RetrievalRunRow(
        id=run.run_id,
        draft_id=run.draft_id,
        document_id=run.document_id.value,
        query=run.query,
        query_hash=_query_hash(run.query),
        embedding_version=str(run.embedding_version),
        retrieval_profile=run.retrieval_profile,
        top_k=run.top_k,
        dense_candidates=run.dense_candidates,
        sparse_candidates=run.sparse_candidates,
        fused_candidates=run.fused_candidates,
        reranked=run.reranked,
        selected=run.selected,
        context_tokens=run.context_tokens,
        duration_ms=run.duration_ms,
        created_at=run.created_at,
    )


def _hit_row(run_id: uuid.UUID, hit: RetrievalHitDTO) -> RetrievalHitRow:
    return RetrievalHitRow(
        id=_hit_id(run_id, hit),
        run_id=run_id,
        chunk_id=hit.chunk_id.value,
        page_number=hit.page_number,
        dense_rank=hit.dense_rank,
        dense_score=hit.dense_score,
        sparse_rank=hit.sparse_rank,
        sparse_score=hit.sparse_score,
        rrf_score=hit.rrf_score,
        rerank_score=hit.rerank_score,
        final_rank=hit.final_rank,
        selected=hit.selected,
    )


def _hit_id(run_id: uuid.UUID, hit: RetrievalHitDTO) -> uuid.UUID:
    """Ключ попадания: прогон и чанк. Повтор записи гасится уникальностью."""
    return uuid.uuid5(NS_AIWORKER, f"{run_id}|{hit.chunk_id}")


def _query_hash(query: str) -> str:
    """Отпечаток запроса: по нему сравниваются прогоны на этапе оценки."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()
