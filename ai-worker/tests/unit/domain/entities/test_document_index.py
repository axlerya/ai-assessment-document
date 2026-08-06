"""Состояние индексации документа: переходы и идемпотентное завершение."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_worker.domain.entities.document_index import DocumentIndex
from ai_worker.domain.errors import InvalidStatusTransition, InvariantViolation
from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
from ai_worker.domain.value_objects.enums import IndexStatus, SourceStatus
from ai_worker.domain.value_objects.identifiers import DocumentId, IndexId
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PipelineVersion,
)

pytestmark = pytest.mark.unit

STARTED = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
FINISHED = datetime(2026, 8, 7, 12, 5, tzinfo=UTC)
EMBEDDING_VERSION = EmbeddingVersion(1, 0, 0)
EMBEDDING = EmbeddingIdentity(version=EMBEDDING_VERSION, model_name="BAAI/bge-m3")


def _pending() -> DocumentIndex:
    return DocumentIndex.pending(
        document_id=DocumentId.generate(),
        embedding=EMBEDDING,
        chunking_version=ChunkingVersion(1, 0, 0),
        pipeline_version=PipelineVersion(1, 0, 0),
        source_status=SourceStatus.PROCESSED,
    )


def _running() -> DocumentIndex:
    return _pending().start(at=STARTED)


def test_key_is_determined_by_document_and_embedding_version() -> None:
    document_id = DocumentId.generate()

    index = DocumentIndex.pending(
        document_id=document_id,
        embedding=EMBEDDING,
        chunking_version=ChunkingVersion(1, 0, 0),
        pipeline_version=PipelineVersion(1, 0, 0),
        source_status=SourceStatus.PROCESSED,
    )

    assert index.id == IndexId.deterministic(
        document_id=document_id, embedding_version=EMBEDDING_VERSION
    )


def test_fresh_index_is_pending() -> None:
    assert _pending().status is IndexStatus.PENDING


def test_start_moves_to_indexing_and_remembers_when() -> None:
    started = _pending().start(at=STARTED)

    assert started.status is IndexStatus.INDEXING
    assert started.started_at == STARTED


def test_completion_records_the_counters() -> None:
    finished = _running().complete(
        chunks_total=10, chunks_embedded=9, chunks_failed=1, at=FINISHED
    )

    assert finished.status is IndexStatus.INDEXED
    assert (finished.chunks_total, finished.chunks_embedded) == (10, 9)
    assert finished.finished_at == FINISHED


def test_counters_of_a_finished_index_must_add_up() -> None:
    # Расхождение означает потерянный или посчитанный дважды чанк, и заметить
    # его позже, чем здесь, будет уже нечем.
    with pytest.raises(InvariantViolation):
        _running().complete(
            chunks_total=10, chunks_embedded=4, chunks_failed=1, at=FINISHED
        )


def test_document_without_a_single_embedded_chunk_is_not_indexed() -> None:
    # Иначе retrieval нашёл бы по такому документу ровно ничего и молча вернул
    # пустой контекст, а документ считался бы готовым.
    with pytest.raises(InvariantViolation):
        _running().complete(
            chunks_total=5, chunks_embedded=0, chunks_failed=5, at=FINISHED
        )


def test_failure_records_its_code() -> None:
    failed = _running().fail(code="no_chunks", message="чанков нет", at=FINISHED)

    assert failed.status is IndexStatus.FAILED
    assert failed.failure_code == "no_chunks"
    assert failed.finished_at == FINISHED


def test_failure_without_a_code_is_useless_for_triage() -> None:
    with pytest.raises(InvariantViolation):
        _running().fail(code="  ", message="что-то пошло не так", at=FINISHED)


def test_index_terminal_transition_is_idempotent_no_op() -> None:
    # Протухший лиз допускает второго воркера: оба дойдут до завершения, и
    # второй не имеет права ни упасть, ни переписать чужой результат.
    finished = _running().complete(
        chunks_total=10, chunks_embedded=10, chunks_failed=0, at=FINISHED
    )

    again = finished.complete(
        chunks_total=10, chunks_embedded=10, chunks_failed=0, at=FINISHED
    )

    assert again is finished


def test_failing_an_already_indexed_document_is_refused() -> None:
    # Это не повтор, а противоречие: корректно проиндексированный документ
    # нельзя пометить отказом.
    finished = _running().complete(
        chunks_total=10, chunks_embedded=10, chunks_failed=0, at=FINISHED
    )

    with pytest.raises(InvalidStatusTransition):
        finished.fail(code="internal_error", message="поздно", at=FINISHED)


def test_repeated_failure_is_idempotent_no_op() -> None:
    failed = _running().fail(code="no_chunks", message="чанков нет", at=FINISHED)

    assert failed.fail(code="no_chunks", message="чанков нет", at=FINISHED) is failed


def test_completion_without_start_is_refused() -> None:
    with pytest.raises(InvalidStatusTransition):
        _pending().complete(
            chunks_total=1, chunks_embedded=1, chunks_failed=0, at=FINISHED
        )


def test_second_start_is_refused() -> None:
    with pytest.raises(InvalidStatusTransition):
        _running().start(at=STARTED)


def test_finish_cannot_precede_start() -> None:
    with pytest.raises(InvariantViolation):
        _running().complete(
            chunks_total=1,
            chunks_embedded=1,
            chunks_failed=0,
            at=datetime(2026, 8, 7, 11, 0, tzinfo=UTC),
        )


def test_moments_without_a_zone_are_refused() -> None:
    # Наивное время в базе с `timestamptz` превращается в тихий сдвиг на часы.
    with pytest.raises(InvariantViolation):
        _pending().start(at=datetime(2026, 8, 7, 12, 0))  # noqa: DTZ001


def test_terminal_statuses_are_recognized() -> None:
    finished = _running().complete(
        chunks_total=1, chunks_embedded=1, chunks_failed=0, at=FINISHED
    )

    assert finished.is_terminal
    assert not _running().is_terminal
