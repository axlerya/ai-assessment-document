"""Идентификаторы: типы не взаимозаменяемы, ключи детерминированы."""

from __future__ import annotations

import uuid

import pytest

from ai_worker.domain.constants import NS_AIWORKER
from ai_worker.domain.errors import InvalidIdentifier
from ai_worker.domain.value_objects.identifiers import (
    ChunkId,
    CitationId,
    ClaimId,
    CorrelationId,
    DocumentId,
    DraftId,
    EmbeddingId,
    EventId,
    IndexId,
    PageId,
    RequestId,
    RetrievalRunId,
)
from ai_worker.domain.value_objects.versioning import EmbeddingVersion, PromptVersion

pytestmark = pytest.mark.unit

ALL_IDENTIFIERS = (
    DocumentId,
    ChunkId,
    PageId,
    EmbeddingId,
    IndexId,
    DraftId,
    ClaimId,
    CitationId,
    RequestId,
    RetrievalRunId,
    EventId,
    CorrelationId,
)

# Namespace document-worker записан здесь литералом намеренно: это чужая
# граница, и импортировать её пакет нельзя. Совпадение namespace сделало бы
# детерминированные ключи двух сервисов пересекающимися.
NS_DOCWORKER_LITERAL = uuid.UUID("6f1c0f8e-6a1e-5b2a-9f3c-2d4e5a6b7c81")


@pytest.mark.parametrize("identifier_class", ALL_IDENTIFIERS)
def test_nil_uuid_is_rejected(identifier_class: type) -> None:
    with pytest.raises(InvalidIdentifier):
        identifier_class(uuid.UUID(int=0))


@pytest.mark.parametrize("identifier_class", ALL_IDENTIFIERS)
def test_identifier_parses_and_prints_its_canonical_form(
    identifier_class: type,
) -> None:
    raw = "0f4a1d3c-2b5e-4a6f-8c9d-1e2f3a4b5c6d"

    assert str(identifier_class.parse(raw)) == raw


@pytest.mark.parametrize("identifier_class", ALL_IDENTIFIERS)
def test_garbage_is_not_an_identifier(identifier_class: type) -> None:
    with pytest.raises(InvalidIdentifier):
        identifier_class.parse("не-uuid")


def test_identifier_types_are_not_interchangeable() -> None:
    # Один и тот же UUID под разными типами — разные значения: подстановку
    # ловит mypy, а равенство не должно её маскировать в рантайме.
    value = uuid.uuid4()

    assert DraftId(value) != ClaimId(value)
    assert ChunkId(value) != PageId(value)


def test_namespace_does_not_collide_with_the_other_service() -> None:
    assert NS_AIWORKER != NS_DOCWORKER_LITERAL


def test_embedding_id_repeats_for_the_same_chunk_and_version() -> None:
    chunk_id = ChunkId.generate()
    version = EmbeddingVersion(1, 0, 0)

    first = EmbeddingId.deterministic(chunk_id=chunk_id, embedding_version=version)
    second = EmbeddingId.deterministic(chunk_id=chunk_id, embedding_version=version)

    assert first == second


def test_embedding_id_changes_with_the_version() -> None:
    # Иначе эмбеддинги двух моделей легли бы в одну строку и повторная
    # индексация новой версией погасилась бы как дубль.
    chunk_id = ChunkId.generate()

    first = EmbeddingId.deterministic(
        chunk_id=chunk_id, embedding_version=EmbeddingVersion(1, 0, 0)
    )
    second = EmbeddingId.deterministic(
        chunk_id=chunk_id, embedding_version=EmbeddingVersion(2, 0, 0)
    )

    assert first != second


def test_index_id_is_determined_by_document_and_version() -> None:
    document_id = DocumentId.generate()
    version = EmbeddingVersion(1, 0, 0)

    first = IndexId.deterministic(document_id=document_id, embedding_version=version)
    second = IndexId.deterministic(document_id=document_id, embedding_version=version)

    assert first == second
    assert first != IndexId.deterministic(
        document_id=DocumentId.generate(), embedding_version=version
    )


def test_draft_id_is_determined_by_request_and_prompt_version() -> None:
    request_id = RequestId.generate()

    first = DraftId.deterministic(
        request_id=request_id, prompt_version=PromptVersion(1, 0, 0)
    )
    second = DraftId.deterministic(
        request_id=request_id, prompt_version=PromptVersion(1, 0, 0)
    )
    other_prompt = DraftId.deterministic(
        request_id=request_id, prompt_version=PromptVersion(1, 1, 0)
    )

    assert first == second
    # Новая версия промпта даёт новый черновик по тому же запросу: иначе
    # сравнить две версии на одном документе было бы нечем.
    assert first != other_prompt


def test_claim_id_is_determined_by_draft_and_position() -> None:
    draft_id = DraftId.generate()

    first = ClaimId.deterministic(draft_id=draft_id, claim_index=0)
    second = ClaimId.deterministic(draft_id=draft_id, claim_index=0)

    assert first == second
    assert first != ClaimId.deterministic(draft_id=draft_id, claim_index=1)


def test_citation_id_is_determined_by_claim_chunk_and_offset() -> None:
    claim_id = ClaimId.generate()
    chunk_id = ChunkId.generate()

    first = CitationId.deterministic(
        claim_id=claim_id, chunk_id=chunk_id, quote_start=10
    )
    second = CitationId.deterministic(
        claim_id=claim_id, chunk_id=chunk_id, quote_start=10
    )

    assert first == second
    assert first != CitationId.deterministic(
        claim_id=claim_id, chunk_id=chunk_id, quote_start=11
    )


def test_indexing_event_id_repeats_for_the_same_document_and_version() -> None:
    document_id = DocumentId.generate()
    version = EmbeddingVersion(1, 0, 0)

    first = EventId.for_indexing(document_id=document_id, embedding_version=version)
    second = EventId.for_indexing(document_id=document_id, embedding_version=version)

    assert first == second


def test_draft_event_id_separates_success_from_failure() -> None:
    request_id = RequestId.generate()
    version = PromptVersion(1, 0, 0)

    generated = EventId.for_draft(
        request_id=request_id, prompt_version=version, event_type="draft.generated"
    )
    failed = EventId.for_draft(
        request_id=request_id, prompt_version=version, event_type="draft.failed"
    )

    assert generated != failed


def test_deterministic_keys_live_in_the_service_namespace() -> None:
    chunk_id = ChunkId.generate()
    version = EmbeddingVersion(1, 0, 0)

    embedding_id = EmbeddingId.deterministic(
        chunk_id=chunk_id, embedding_version=version
    )

    expected = uuid.uuid5(NS_AIWORKER, f"{chunk_id}|{version}")
    assert embedding_id.value == expected
