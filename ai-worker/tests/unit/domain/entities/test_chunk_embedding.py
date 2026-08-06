"""Проекция чанка и его эмбеддинг."""

from __future__ import annotations

import pytest

from ai_worker.domain.entities.chunk_embedding import ChunkEmbedding
from ai_worker.domain.entities.source_chunk import ChunkQuality, ChunkRef
from ai_worker.domain.errors import InvariantViolation
from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
from ai_worker.domain.value_objects.enums import ExtractionMethod
from ai_worker.domain.value_objects.hashing import ContentHash
from ai_worker.domain.value_objects.identifiers import (
    ChunkId,
    DocumentId,
    EmbeddingId,
    PageId,
)
from ai_worker.domain.value_objects.scores import Ratio
from ai_worker.domain.value_objects.versioning import ChunkingVersion, EmbeddingVersion
from tests.factories import make_chunk, make_dense, make_sparse

pytestmark = pytest.mark.unit

EMBEDDING_VERSION = EmbeddingVersion(1, 0, 0)
EMBEDDING = EmbeddingIdentity(version=EMBEDDING_VERSION, model_name="BAAI/bge-m3")


def test_page_number_starts_at_one() -> None:
    with pytest.raises(InvariantViolation):
        ChunkRef(
            chunk_id=ChunkId.generate(),
            document_id=DocumentId.generate(),
            page_id=PageId.generate(),
            page_number=0,
        )


def test_text_layer_chunk_has_no_confidence() -> None:
    # Уверенность текстового слоя и уверенность распознавания — величины
    # разной природы: смешав их, мы навсегда портим любой агрегат качества.
    with pytest.raises(InvariantViolation):
        ChunkQuality(
            extraction_method=ExtractionMethod.TEXT_LAYER,
            avg_confidence=Ratio(1.0),
            illegible_span_count=0,
        )


@pytest.mark.parametrize("method", [ExtractionMethod.OCR, ExtractionMethod.HYBRID])
def test_recognized_chunk_must_carry_confidence(method: ExtractionMethod) -> None:
    # Без неё потребитель не отличит надёжный фрагмент от мусора.
    with pytest.raises(InvariantViolation):
        ChunkQuality(
            extraction_method=method,
            avg_confidence=None,
            illegible_span_count=0,
        )


def test_illegible_span_count_is_not_negative() -> None:
    with pytest.raises(InvariantViolation):
        ChunkQuality(
            extraction_method=ExtractionMethod.TEXT_LAYER,
            avg_confidence=None,
            illegible_span_count=-1,
        )


def test_blank_chunk_text_is_not_indexable() -> None:
    with pytest.raises(InvariantViolation):
        make_chunk(text="   \n ")


def test_chunk_knows_its_own_content_hash() -> None:
    chunk = make_chunk(text="Договор № 12/АБ")

    assert chunk.content_hash == ContentHash.sha256_of("Договор № 12/АБ")


def test_content_hash_rejects_anything_but_lowercase_sha256() -> None:
    for raw in ("ABC", "z" * 64, "0" * 63, ""):
        with pytest.raises(InvariantViolation):
            ContentHash(raw)


def test_embedding_is_built_from_its_chunk() -> None:
    chunk = make_chunk()

    embedding = ChunkEmbedding.of(
        chunk=chunk,
        embedding=EMBEDDING,
        dense=make_dense(),
        sparse=make_sparse(),
    )

    assert embedding.ref == chunk.ref
    assert embedding.quality == chunk.quality
    assert embedding.content_hash == chunk.content_hash
    assert embedding.token_count == chunk.token_count


def test_embedding_key_is_determined_by_chunk_and_version() -> None:
    chunk = make_chunk()

    embedding = ChunkEmbedding.of(
        chunk=chunk,
        embedding=EMBEDDING,
        dense=make_dense(),
        sparse=make_sparse(),
    )

    assert embedding.id == EmbeddingId.deterministic(
        chunk_id=chunk.ref.chunk_id, embedding_version=EMBEDDING_VERSION
    )


def test_embedding_without_a_model_name_is_meaningless() -> None:
    # Имя модели — часть происхождения вектора: без него нельзя ни объяснить
    # выдачу, ни решить, нужна ли переиндексация. Версия и модель — одно
    # значение, поэтому проверка живёт там, а не в каждом её потребителе.
    with pytest.raises(InvariantViolation):
        EmbeddingIdentity(version=EMBEDDING_VERSION, model_name="  ")


def test_embedding_is_identified_by_its_key_not_by_its_vectors() -> None:
    chunk = make_chunk()
    first = ChunkEmbedding.of(
        chunk=chunk,
        embedding=EMBEDDING,
        dense=make_dense(0.01),
        sparse=make_sparse(),
    )
    second = ChunkEmbedding.of(
        chunk=chunk,
        embedding=EMBEDDING,
        dense=make_dense(0.02),
        sparse=make_sparse(),
    )

    assert first == second
    assert len({first, second}) == 1


def test_embedding_carries_the_chunking_version_of_its_source() -> None:
    # Без неё нельзя воспроизвести, по какому корпусу строился индекс.
    chunk = make_chunk(chunking_version=ChunkingVersion(2, 0, 0))

    embedding = ChunkEmbedding.of(
        chunk=chunk,
        embedding=EMBEDDING,
        dense=make_dense(),
        sparse=make_sparse(),
    )

    assert embedding.chunking_version == ChunkingVersion(2, 0, 0)
