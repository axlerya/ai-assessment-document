"""Мапперы: строка и сущность описывают одно и то же.

Round-trip проверяется полем за полем, а не выборочно. Потерянное при переводе
поле не видно ни одному другому тесту: сущность собирается, база принимает
строку, и расхождение всплывает через недели — на цитате, которая указывает не
туда, или на переиндексации, которая не понимает, что уже сделана.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime

import pytest

from ai_worker.domain.entities.document_index import DocumentIndex
from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
from ai_worker.domain.value_objects.enums import (
    ClaimSection,
    DraftType,
    ExtractionMethod,
    RejectCode,
    SourceStatus,
)
from ai_worker.domain.value_objects.identifiers import (
    ClaimId,
    CorrelationId,
    DocumentId,
    EventId,
    RequestId,
)
from ai_worker.domain.value_objects.source_snapshot import SourceSnapshot
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PipelineVersion,
    PromptVersion,
)
from ai_worker.infrastructure.persistence.mappers.draft import (
    draft_to_rows,
    rows_to_draft,
)
from ai_worker.infrastructure.persistence.mappers.embedding import (
    embedding_to_row,
    row_to_embedding,
)
from ai_worker.infrastructure.persistence.mappers.index import (
    index_to_row,
    row_to_index,
)
from tests.factories import (
    draft_id_for,
    make_chunk,
    make_claim,
    make_draft,
    make_embedding,
    make_quality,
)

pytestmark = pytest.mark.unit

STARTED = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
FINISHED = datetime(2026, 8, 7, 12, 5, tzinfo=UTC)
EMBEDDING = EmbeddingIdentity(
    version=EmbeddingVersion(1, 0, 0), model_name="BAAI/bge-m3"
)


def _running_index() -> DocumentIndex:
    return DocumentIndex.pending(
        document_id=DocumentId.generate(),
        embedding=EMBEDDING,
        source=SourceSnapshot(
            pipeline_version=PipelineVersion(1, 0, 0),
            chunking_version=ChunkingVersion(1, 0, 0),
            status=SourceStatus.PARTIALLY_PROCESSED,
        ),
        source_event_id=EventId.generate(),
        correlation_id=CorrelationId.generate(),
    ).start(at=STARTED)


def _assert_same(left: object, right: object) -> None:
    for field in fields(left):  # type: ignore[arg-type]
        assert getattr(left, field.name) == getattr(right, field.name), field.name


def test_embedding_round_trip_preserves_every_field() -> None:
    embedding = make_embedding(
        chunk=make_chunk(
            quality=make_quality(method=ExtractionMethod.OCR, confidence=0.874),
            heading_path=("Раздел 1", "Предмет договора"),
        )
    )

    restored = row_to_embedding(embedding_to_row(embedding))

    _assert_same(embedding, restored)


def test_embedding_round_trip_keeps_vectors_intact() -> None:
    # Вектор — единственное, что нельзя восстановить по остальным полям: его
    # порча означает молча испорченный поиск, а не ошибку.
    embedding = make_embedding()

    restored = row_to_embedding(embedding_to_row(embedding))

    assert restored.dense == embedding.dense
    assert restored.sparse == embedding.sparse


def test_embedding_of_a_text_layer_chunk_keeps_absent_confidence() -> None:
    # `None` и `0` здесь означают разное: отсутствие уверенности и полное её
    # отсутствие как значение.
    embedding = make_embedding(chunk=make_chunk(quality=make_quality()))

    restored = row_to_embedding(embedding_to_row(embedding))

    assert restored.quality.avg_confidence is None


def test_index_round_trip_preserves_every_field() -> None:
    index = _running_index().complete(
        chunks_total=10, chunks_embedded=9, chunks_failed=1, at=FINISHED
    )

    restored = row_to_index(index_to_row(index))

    _assert_same(index, restored)


def test_failed_index_round_trip_keeps_its_reason() -> None:
    index = _running_index().fail(code="no_chunks", message="чанков нет", at=FINISHED)

    restored = row_to_index(index_to_row(index))

    assert restored.failure_code == "no_chunks"
    assert restored.failure_message == "чанков нет"


def test_index_moments_survive_with_their_zone() -> None:
    # Потерянная зона превращается в тихий сдвиг на часы при чтении из базы.
    index = _running_index()

    restored = row_to_index(index_to_row(index))

    assert restored.started_at == STARTED
    assert restored.started_at is not None
    assert restored.started_at.tzinfo is not None


def test_draft_round_trip_preserves_every_field() -> None:
    draft = make_draft()

    rows = draft_to_rows(draft)
    restored = rows_to_draft(rows.draft, rows.claims, rows.citations)

    _assert_same(draft, restored)


def test_draft_round_trip_keeps_rejected_claims() -> None:
    # Отклонённые утверждения — материал разбора: потеряв их при сохранении,
    # мы теряем и след попытки додумать факт.
    request_id = RequestId.generate()
    document_id = DocumentId.generate()
    draft_id = draft_id_for(request_id)
    rejected = make_claim(
        draft_id=draft_id,
        document_id=document_id,
        index=1,
        supported=False,
        reject_code=RejectCode.QUOTE_NOT_FOUND,
        citations=(),
    )
    draft = make_draft(
        request_id=request_id,
        document_id=document_id,
        claims=(
            make_claim(draft_id=draft_id, document_id=document_id),
            rejected,
        ),
    )

    rows = draft_to_rows(draft)
    restored = rows_to_draft(rows.draft, rows.claims, rows.citations)

    assert restored.claims_unsupported == 1
    assert restored.claims[1].reject_code is RejectCode.QUOTE_NOT_FOUND


def test_draft_round_trip_keeps_citations_with_their_claims() -> None:
    draft = make_draft()

    rows = draft_to_rows(draft)
    restored = rows_to_draft(rows.draft, rows.claims, rows.citations)

    original = draft.claims[0].citations[0]
    citation = restored.claims[0].citations[0]
    assert citation.id == original.id
    assert citation.quote == original.quote
    assert citation.span == original.span
    assert citation.ref == original.ref
    assert citation.reliable == original.reliable


def test_draft_round_trip_keeps_claim_order() -> None:
    # Порядок — часть черновика: разделы собираются по нему, и перестановка
    # меняет текст, который увидит оператор.
    request_id = RequestId.generate()
    document_id = DocumentId.generate()
    draft_id = draft_id_for(request_id)
    claims = tuple(
        make_claim(
            draft_id=draft_id,
            index=index,
            section=section,
            document_id=document_id,
        )
        for index, section in enumerate(
            (ClaimSection.DATES, ClaimSection.PARTIES, ClaimSection.AMOUNTS)
        )
    )
    draft = make_draft(request_id=request_id, document_id=document_id, claims=claims)

    rows = draft_to_rows(draft)
    restored = rows_to_draft(rows.draft, rows.claims, rows.citations)

    assert [claim.index for claim in restored.claims] == [0, 1, 2]
    assert [claim.section for claim in restored.claims] == [
        ClaimSection.DATES,
        ClaimSection.PARTIES,
        ClaimSection.AMOUNTS,
    ]


def test_draft_row_carries_the_counters_the_event_needs() -> None:
    # Счётчики считаются доменом и переносятся в строку: пересчёт в SQL был бы
    # вторым источником одной величины.
    draft = make_draft()

    row = draft_to_rows(draft).draft

    assert row.claims_total == draft.claims_total
    assert row.claims_grounded == draft.claims_grounded
    assert row.claims_unsupported == draft.claims_unsupported


def test_draft_row_stores_the_rendered_body() -> None:
    # Тело собирается доменом из подтверждённых утверждений; хранится готовым,
    # чтобы потребитель не пересобирал его своей версией правил.
    draft = make_draft()

    assert draft_to_rows(draft).draft.body == draft.body


def test_draft_round_trip_keeps_correlation_and_request() -> None:
    request_id = RequestId.generate()
    correlation_id = CorrelationId.generate()
    draft = make_draft(request_id=request_id, correlation_id=correlation_id)

    rows = draft_to_rows(draft)
    restored = rows_to_draft(rows.draft, rows.claims, rows.citations)

    assert restored.request_id == request_id
    assert restored.correlation_id == correlation_id


def test_draft_round_trip_keeps_its_provenance() -> None:
    # По этим четырём величинам черновик воспроизводится: без них нельзя
    # объяснить, почему два прогона по одному документу разошлись.
    draft = make_draft(prompt_version=PromptVersion(2, 1, 0))

    rows = draft_to_rows(draft)
    restored = rows_to_draft(rows.draft, rows.claims, rows.citations)

    assert restored.prompt_version == PromptVersion(2, 1, 0)
    assert restored.embedding_version == EmbeddingVersion(1, 0, 0)
    assert restored.chunking_version == ChunkingVersion(1, 0, 0)
    assert restored.retrieval_profile == "hybrid-rrf-v1"
    assert restored.draft_type is DraftType.CASE_FACT_SUMMARY


def test_claim_identifiers_survive_the_round_trip() -> None:
    request_id = RequestId.generate()
    document_id = DocumentId.generate()
    draft = make_draft(
        request_id=request_id,
        document_id=document_id,
        claims=(
            make_claim(draft_id=draft_id_for(request_id), document_id=document_id),
        ),
    )

    rows = draft_to_rows(draft)
    restored = rows_to_draft(rows.draft, rows.claims, rows.citations)

    assert restored.claims[0].id == ClaimId.deterministic(
        draft_id=draft.id, claim_index=0
    )


def test_sparse_vector_is_read_back_from_a_raw_literal() -> None:
    # Сырой запрос отдаёт `sparsevec` строкой: индексы в ней одномерные, и
    # смещение на единицу здесь — единственное место, где домен и хранилище
    # расходятся в нумерации.
    embedding = make_embedding()
    row = embedding_to_row(embedding)
    row.sparse = row.sparse.to_text()

    assert row_to_embedding(row).sparse == embedding.sparse
