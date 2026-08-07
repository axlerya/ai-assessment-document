"""История поиска: прогон и его попадания.

Хранится не для отчёта: по рангам обеих ветвей видно, что нашло фрагмент, а на
этих же строках считаются Recall@K, MRR и nDCG на этапе оценки качества.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from ai_worker.domain.value_objects.identifiers import ChunkId, DocumentId
    from ai_worker.domain.value_objects.versioning import EmbeddingVersion


@dataclass(frozen=True, slots=True)
class RetrievalRunDTO:
    """Один прогон поиска со счётчиками каждой ступени."""

    run_id: UUID
    draft_id: UUID | None
    document_id: DocumentId
    query: str
    embedding_version: EmbeddingVersion
    retrieval_profile: str
    top_k: int
    dense_candidates: int
    sparse_candidates: int
    fused_candidates: int
    reranked: int
    selected: int
    context_tokens: int
    duration_ms: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RetrievalHitDTO:
    """Одно попадание прогона со своими рангами и оценками."""

    chunk_id: ChunkId
    page_number: int
    rrf_score: float
    final_rank: int
    selected: bool
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
