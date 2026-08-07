"""Фабрики доменных объектов для тестов.

Значения по умолчанию правдоподобны, но не значимы: тест, которому важно
конкретное поле, задаёт его явно, и по вызову видно, что именно он проверяет.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ai_worker.domain.constants import DENSE_DIMENSIONS
from ai_worker.domain.entities.chunk_embedding import ChunkEmbedding
from ai_worker.domain.entities.draft import Citation, Claim, Draft
from ai_worker.domain.entities.source_chunk import ChunkQuality, ChunkRef, SourceChunk
from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
from ai_worker.domain.value_objects.enums import (
    ClaimSection,
    DraftType,
    ExtractionMethod,
)
from ai_worker.domain.value_objects.identifiers import (
    ChunkId,
    ClaimId,
    DocumentId,
    DraftId,
    PageId,
    RequestId,
)
from ai_worker.domain.value_objects.scores import Ratio, Score
from ai_worker.domain.value_objects.text import QuoteSpan
from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PromptVersion,
)

if TYPE_CHECKING:
    from ai_worker.domain.value_objects.enums import RejectCode
    from ai_worker.domain.value_objects.identifiers import CorrelationId

CHUNK_TEXT = (
    "Договор поставки № 12/АБ от 3 марта 2024 года заключён между ООО «Вектор» "
    "и АО «Полюс» на сумму 1 250 000 рублей."
)
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
QUOTE = "Договор поставки № 12/АБ"
EMBEDDING = EmbeddingIdentity(
    version=EmbeddingVersion(1, 0, 0), model_name="BAAI/bge-m3"
)


def make_ref(
    *,
    page_number: int = 3,
    document_id: DocumentId | None = None,
) -> ChunkRef:
    """Ссылка на чанк вместе с координатами цитирования."""
    return ChunkRef(
        chunk_id=ChunkId.generate(),
        document_id=document_id or DocumentId.generate(),
        page_id=PageId.generate(),
        page_number=page_number,
    )


def make_quality(
    *,
    method: ExtractionMethod = ExtractionMethod.TEXT_LAYER,
    confidence: float | None = None,
    illegible_span_count: int = 0,
) -> ChunkQuality:
    """Признаки надёжности чанка, скопированные из document-worker."""
    return ChunkQuality(
        extraction_method=method,
        avg_confidence=None if confidence is None else Ratio(confidence),
        illegible_span_count=illegible_span_count,
    )


def make_chunk(
    *,
    text: str = CHUNK_TEXT,
    ref: ChunkRef | None = None,
    quality: ChunkQuality | None = None,
    token_count: int = 42,
    chunking_version: ChunkingVersion | None = None,
    heading_path: tuple[str, ...] = ("Предмет договора",),
    document_id: DocumentId | None = None,
) -> SourceChunk:
    """Чанк документа в том виде, в каком его отдаёт document-worker."""
    return SourceChunk(
        ref=ref or make_ref(document_id=document_id),
        quality=quality or make_quality(),
        text=text,
        token_count=token_count,
        chunking_version=chunking_version or ChunkingVersion(1, 0, 0),
        heading_path=heading_path,
    )


def make_dense(value: float = 0.01) -> DenseVector:
    """Плотный вектор нужной ширины."""
    return DenseVector(tuple(value for _ in range(DENSE_DIMENSIONS)))


def make_sparse(weights: dict[int, float] | None = None) -> SparseVector:
    """Разреженный вектор из нескольких весов."""
    return SparseVector.pruned(weights or {7: 0.9, 19: 0.4, 101: 0.15})


def make_embedding(
    *,
    chunk: SourceChunk | None = None,
    embedding: EmbeddingIdentity | None = None,
) -> ChunkEmbedding:
    """Эмбеддинг чанка в первой версии."""
    return ChunkEmbedding.of(
        chunk=chunk or make_chunk(),
        embedding=embedding or EMBEDDING,
        dense=make_dense(),
        sparse=make_sparse(),
    )


def make_citation(
    *,
    claim_id: ClaimId,
    chunk: SourceChunk | None = None,
    reliable: bool = True,
    document_id: DocumentId | None = None,
) -> Citation:
    """Цитата на первые слова текста чанка."""
    return Citation.for_quote(
        claim_id=claim_id,
        chunk=chunk or make_chunk(document_id=document_id),
        span=QuoteSpan(start=0, end=len(QUOTE)),
        quote=QUOTE,
        retrieval_score=Score(0.9),
        rerank_score=Score(2.1),
        reliable=reliable,
    )


def make_claim(
    *,
    draft_id: DraftId,
    index: int = 0,
    section: ClaimSection = ClaimSection.DOCUMENTS,
    text: str = "Стороны заключили договор поставки № 12/АБ.",
    citations: tuple[Citation, ...] | None = None,
    supported: bool = True,
    reject_code: RejectCode | None = None,
    document_id: DocumentId | None = None,
) -> Claim:
    """Утверждение черновика вместе со своей цитатой."""
    claim_id = ClaimId.deterministic(draft_id=draft_id, claim_index=index)
    return Claim(
        id=claim_id,
        index=index,
        section=section,
        text=text,
        citations=(
            citations
            if citations is not None
            else (make_citation(claim_id=claim_id, document_id=document_id),)
        ),
        supported=supported,
        reject_code=reject_code,
    )


def draft_id_for(
    request_id: RequestId,
    prompt_version: PromptVersion | None = None,
) -> DraftId:
    """Ключ черновика, который построит `Draft.assembled`.

    Утверждениям он нужен заранее: их собственные ключи выводятся из него, а
    сам черновик собирается уже вокруг готовых утверждений.
    """
    return DraftId.deterministic(
        request_id=request_id, prompt_version=prompt_version or PromptVersion(1, 0, 0)
    )


def make_draft(
    *,
    request_id: RequestId | None = None,
    claims: tuple[Claim, ...] | None = None,
    prompt_version: PromptVersion | None = None,
    correlation_id: CorrelationId | None = None,
    document_id: DocumentId | None = None,
) -> Draft:
    """Черновик с одним подтверждённым утверждением."""
    version = prompt_version or PromptVersion(1, 0, 0)
    request = request_id or RequestId.generate()
    identity = draft_id_for(request, version)
    document = document_id or DocumentId.generate()
    return Draft.assembled(
        request_id=request,
        document_id=document,
        draft_type=DraftType.CASE_FACT_SUMMARY,
        query="Собери сводку фактов по делу",
        claims=(
            claims
            if claims is not None
            else (make_claim(draft_id=identity, document_id=document),)
        ),
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        prompt_version=version,
        retrieval_profile="hybrid-rrf-v1",
        embedding_version=EmbeddingVersion(1, 0, 0),
        chunking_version=ChunkingVersion(1, 0, 0),
        evidence_total=7,
        correlation_id=correlation_id,
    )
