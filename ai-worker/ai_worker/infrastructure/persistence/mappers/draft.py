"""Перевод черновика в три набора строк и обратно.

Черновик, его утверждения и их цитаты пишутся вместе и читаются вместе:
черновик без цитат — это ровно то, что ТЗ считает провалом, и разнести их по
разным операциям значило бы допустить такое состояние.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from ai_worker.domain.entities.draft import Citation, Claim, Draft
from ai_worker.domain.entities.source_chunk import ChunkRef
from ai_worker.domain.value_objects.enums import (
    ClaimSection,
    DraftStatus,
    DraftType,
    RejectCode,
)
from ai_worker.domain.value_objects.identifiers import (
    ChunkId,
    CitationId,
    ClaimId,
    CorrelationId,
    DocumentId,
    DraftId,
    PageId,
    RequestId,
)
from ai_worker.domain.value_objects.scores import Score
from ai_worker.domain.value_objects.text import QuoteSpan
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PromptVersion,
)
from ai_worker.infrastructure.persistence.models.draft import (
    DraftCitationRow,
    DraftClaimRow,
    DraftRow,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_GROUNDEDNESS_QUANT = Decimal("0.001")


@dataclass(frozen=True, slots=True)
class DraftRows:
    """Все строки одного черновика."""

    draft: DraftRow
    claims: tuple[DraftClaimRow, ...]
    citations: tuple[DraftCitationRow, ...]


def draft_to_rows(draft: Draft) -> DraftRows:
    """Готовит черновик со всем содержимым к записи."""
    claims = tuple(_claim_to_row(claim, draft_id=draft.id) for claim in draft.claims)
    citations = tuple(
        _citation_to_row(citation, draft_id=draft.id)
        for claim in draft.claims
        for citation in claim.citations
    )
    return DraftRows(draft=_draft_to_row(draft), claims=claims, citations=citations)


def rows_to_draft(
    row: DraftRow,
    claims: Sequence[DraftClaimRow],
    citations: Sequence[DraftCitationRow],
) -> Draft:
    """Восстанавливает черновик из его строк."""
    # Документа у строки цитаты нет: он один на весь черновик, и хранить его
    # в каждой цитате значило бы завести ещё одну точку рассинхрона.
    document_id = DocumentId(row.document_id)
    by_claim: dict[ClaimId, list[Citation]] = {}
    for citation in citations:
        by_claim.setdefault(ClaimId(citation.claim_id), []).append(
            _row_to_citation(citation, document_id=document_id)
        )
    restored = tuple(
        _row_to_claim(claim, by_claim.get(ClaimId(claim.id), []))
        for claim in sorted(claims, key=lambda claim: claim.claim_index)
    )
    return Draft(
        id=DraftId(row.id),
        request_id=RequestId(row.request_id),
        document_id=DocumentId(row.document_id),
        draft_type=DraftType(row.draft_type),
        query=row.query,
        status=DraftStatus(row.status),
        claims=restored,
        model_name=row.model_name,
        prompt_version=PromptVersion.parse(row.prompt_version),
        retrieval_profile=row.retrieval_profile,
        embedding_version=EmbeddingVersion.parse(row.embedding_version),
        chunking_version=ChunkingVersion.parse(row.chunking_version),
        evidence_total=row.evidence_total,
        correlation_id=None
        if row.correlation_id is None
        else CorrelationId(row.correlation_id),
    )


def _draft_to_row(draft: Draft) -> DraftRow:
    return DraftRow(
        id=draft.id.value,
        request_id=draft.request_id.value,
        document_id=draft.document_id.value,
        draft_type=draft.draft_type.value,
        query=draft.query,
        status=draft.status.value,
        # Тело собрано доменом из подтверждённых утверждений и хранится
        # готовым: пересборка потребителем была бы второй версией правил.
        body=draft.body,
        model_name=draft.model_name,
        prompt_version=str(draft.prompt_version),
        retrieval_profile=draft.retrieval_profile,
        embedding_version=str(draft.embedding_version),
        chunking_version=str(draft.chunking_version),
        claims_total=draft.claims_total,
        claims_grounded=draft.claims_grounded,
        claims_unsupported=draft.claims_unsupported,
        evidence_total=draft.evidence_total,
        groundedness=Decimal(str(draft.groundedness.value)).quantize(
            _GROUNDEDNESS_QUANT
        ),
        correlation_id=None
        if draft.correlation_id is None
        else draft.correlation_id.value,
    )


def _claim_to_row(claim: Claim, *, draft_id: DraftId) -> DraftClaimRow:
    return DraftClaimRow(
        id=claim.id.value,
        draft_id=draft_id.value,
        claim_index=claim.index,
        section=claim.section.value,
        text=claim.text,
        supported=claim.supported,
        reject_code=None if claim.reject_code is None else claim.reject_code.value,
    )


def _row_to_claim(row: DraftClaimRow, citations: Sequence[Citation]) -> Claim:
    return Claim(
        id=ClaimId(row.id),
        index=row.claim_index,
        section=ClaimSection(row.section),
        text=row.text,
        citations=tuple(sorted(citations, key=lambda citation: citation.span.start)),
        supported=row.supported,
        reject_code=None if row.reject_code is None else RejectCode(row.reject_code),
    )


def _citation_to_row(citation: Citation, *, draft_id: DraftId) -> DraftCitationRow:
    return DraftCitationRow(
        id=citation.id.value,
        claim_id=citation.claim_id.value,
        draft_id=draft_id.value,
        chunk_id=citation.ref.chunk_id.value,
        page_id=citation.ref.page_id.value,
        page_number=citation.ref.page_number,
        quote=citation.quote,
        quote_start=citation.span.start,
        quote_end=citation.span.end,
        retrieval_score=citation.retrieval_score.value,
        rerank_score=citation.rerank_score.value,
        reliable=citation.reliable,
    )


def _row_to_citation(row: DraftCitationRow, *, document_id: DocumentId) -> Citation:
    return Citation(
        id=CitationId(row.id),
        claim_id=ClaimId(row.claim_id),
        ref=ChunkRef(
            chunk_id=ChunkId(row.chunk_id),
            document_id=document_id,
            page_id=PageId(row.page_id),
            page_number=row.page_number,
        ),
        quote=row.quote,
        span=QuoteSpan(start=row.quote_start, end=row.quote_end),
        retrieval_score=Score(row.retrieval_score),
        rerank_score=Score(row.rerank_score),
        reliable=row.reliable,
    )
