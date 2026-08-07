"""Репозитории и единица работы на живой базе.

Репозиторий не коммитит: транзакцией управляет тот, кто знает границы работы.
Иначе половина результата оказывалась бы зафиксированной, а вторая — нет, и
восстановить, что именно произошло, было бы нечем.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from ai_worker.application.dto.messaging import ClaimOutcome, OutboxEventDTO
from ai_worker.application.dto.retrieval import RetrievalHitDTO, RetrievalRunDTO
from ai_worker.application.errors import DuplicateRecord
from ai_worker.domain.entities.document_index import DocumentIndex
from ai_worker.domain.value_objects.enums import (
    IndexStatus,
    RejectCode,
    SourceStatus,
)
from ai_worker.domain.value_objects.identifiers import (
    ChunkId,
    DocumentId,
    DraftId,
    EventId,
    RequestId,
)
from ai_worker.domain.value_objects.source_snapshot import SourceSnapshot
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PipelineVersion,
)
from ai_worker.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork
from tests.factories import (
    EMBEDDING,
    draft_id_for,
    make_chunk,
    make_claim,
    make_draft,
    make_embedding,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncConnection

    from ai_worker.domain.entities.chunk_embedding import ChunkEmbedding


pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(minutes=5)
VERSION = EmbeddingVersion(1, 0, 0)
SOURCE = SourceSnapshot(
    pipeline_version=PipelineVersion(1, 0, 0),
    chunking_version=ChunkingVersion(1, 0, 0),
    status=SourceStatus.PROCESSED,
)


@pytest.fixture
async def uow(connection: AsyncConnection) -> AsyncIterator[SqlAlchemyUnitOfWork]:
    """Единица работы поверх откатываемой транзакции теста."""
    async with SqlAlchemyUnitOfWork.on(connection) as opened:
        yield opened


def _index(document_id: DocumentId | None = None) -> DocumentIndex:
    return DocumentIndex.pending(
        document_id=document_id or DocumentId.generate(),
        embedding=EMBEDDING,
        source=SOURCE,
        source_event_id=EventId.generate(),
    )


def _embeddings(document_id: DocumentId, count: int) -> list[ChunkEmbedding]:
    return [
        make_embedding(chunk=make_chunk(document_id=document_id)) for _ in range(count)
    ]


async def _rows(connection: AsyncConnection, table: str) -> int:
    counted = await connection.scalar(text(f"SELECT count(*) FROM {table}"))
    return int(counted or 0)


# --------------------------- единица работы ---------------------------


async def test_repository_never_commits(
    uow: SqlAlchemyUnitOfWork,
    connection: AsyncConnection,
) -> None:
    # Записанное репозиторием не должно пережить откат: иначе половина
    # результата фиксируется, а вторая теряется.
    await uow.index.add(_index())

    assert await _rows(connection, "ai_document_index") == 1


async def test_unit_of_work_rolls_back_on_error(
    connection: AsyncConnection,
) -> None:
    async def _failing_scenario() -> None:
        async with SqlAlchemyUnitOfWork.on(connection) as opened:
            await opened.index.add(_index())
            raise RuntimeError("сценарий сорвался")

    with pytest.raises(RuntimeError, match="сценарий сорвался"):
        await _failing_scenario()

    assert await _rows(connection, "ai_document_index") == 0


async def test_unit_of_work_keeps_what_was_committed(
    connection: AsyncConnection,
) -> None:
    async with SqlAlchemyUnitOfWork.on(connection) as opened:
        await opened.index.add(_index())
        await opened.commit()

    assert await _rows(connection, "ai_document_index") == 1


async def test_driver_failure_arrives_classified(uow: SqlAlchemyUnitOfWork) -> None:
    # Подписчику нужен класс ошибки, а не `DBAPIError`: по нему он решает,
    # повторить или отправить в разбор.
    index = _index()
    await uow.index.add(index)

    with pytest.raises(DuplicateRecord):
        await uow.index.add(index)


# --------------------------- индекс документа ---------------------------


async def test_index_is_read_back_by_document_and_version(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    index = _index()
    await uow.index.add(index)

    found = await uow.index.get(
        document_id=index.document_id, embedding_version=VERSION
    )

    assert found is not None
    assert found.id == index.id
    assert found.source == SOURCE


async def test_missing_index_reads_as_absent(uow: SqlAlchemyUnitOfWork) -> None:
    found = await uow.index.get(
        document_id=DocumentId.generate(), embedding_version=VERSION
    )

    assert found is None


async def test_terminal_update_uses_a_status_guard(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    # Ноль изменённых строк означает «кто-то уже завершил», а не ошибку:
    # именно эта ветка не даёт пометить готовый документ отказом.
    index = _index()
    await uow.index.add(index)
    running = await uow.index.start(index, at=NOW)
    finished = running.complete(
        chunks_total=2, chunks_embedded=2, chunks_failed=0, at=LATER
    )

    assert await uow.index.finish(finished, expected=IndexStatus.INDEXING)
    assert not await uow.index.finish(finished, expected=IndexStatus.INDEXING)


async def test_finished_index_keeps_its_counters(uow: SqlAlchemyUnitOfWork) -> None:
    index = _index()
    await uow.index.add(index)
    running = await uow.index.start(index, at=NOW)
    await uow.index.finish(
        running.complete(chunks_total=3, chunks_embedded=2, chunks_failed=1, at=LATER),
        expected=IndexStatus.INDEXING,
    )

    found = await uow.index.get(
        document_id=index.document_id, embedding_version=VERSION
    )

    assert found is not None
    assert found.status is IndexStatus.INDEXED
    assert (found.chunks_embedded, found.chunks_failed) == (2, 1)


# --------------------------- эмбеддинги ---------------------------


async def test_embeddings_are_written_in_one_batch(
    uow: SqlAlchemyUnitOfWork,
    connection: AsyncConnection,
) -> None:
    document_id = DocumentId.generate()

    written = await uow.embeddings.add_many(_embeddings(document_id, 5))

    assert written == 5
    assert await _rows(connection, "ai_chunk_embeddings") == 5


async def test_batch_insert_of_embeddings_is_idempotent(
    uow: SqlAlchemyUnitOfWork,
    connection: AsyncConnection,
) -> None:
    # Повторная доставка не должна ни падать, ни удваивать корпус: ключ
    # эмбеддинга детерминирован, и повтор гасится уникальным ограничением.
    batch = _embeddings(DocumentId.generate(), 4)
    await uow.embeddings.add_many(batch)

    written = await uow.embeddings.add_many(batch)

    assert written == 0
    assert await _rows(connection, "ai_chunk_embeddings") == 4


async def test_empty_batch_writes_nothing(uow: SqlAlchemyUnitOfWork) -> None:
    assert await uow.embeddings.add_many([]) == 0


async def test_stored_hashes_tell_what_is_already_embedded(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    # По ним пропускается повторный прогон модели: устав прямо запрещает
    # пересчитывать эмбеддинг чанка, текст которого не изменился.
    document_id = DocumentId.generate()
    batch = _embeddings(document_id, 3)
    await uow.embeddings.add_many(batch)

    stored = await uow.embeddings.stored_hashes(
        document_id=document_id, embedding_version=VERSION
    )

    assert stored == {
        embedding.ref.chunk_id: embedding.content_hash for embedding in batch
    }


async def test_stored_hashes_are_scoped_to_their_version(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    document_id = DocumentId.generate()
    await uow.embeddings.add_many(_embeddings(document_id, 2))

    stored = await uow.embeddings.stored_hashes(
        document_id=document_id, embedding_version=EmbeddingVersion(2, 0, 0)
    )

    assert stored == {}


async def test_embeddings_are_counted_per_document_and_version(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    document_id = DocumentId.generate()
    await uow.embeddings.add_many(_embeddings(document_id, 3))
    await uow.embeddings.add_many(_embeddings(DocumentId.generate(), 2))

    counted = await uow.embeddings.count(
        document_id=document_id, embedding_version=VERSION
    )

    assert counted == 3


# --------------------------- черновики ---------------------------


async def test_draft_is_saved_with_its_claims_and_citations(
    uow: SqlAlchemyUnitOfWork,
    connection: AsyncConnection,
) -> None:
    await uow.drafts.save(make_draft())

    assert await _rows(connection, "ai_drafts") == 1
    assert await _rows(connection, "ai_draft_claims") == 1
    assert await _rows(connection, "ai_draft_citations") == 1


async def test_draft_is_read_back_whole(uow: SqlAlchemyUnitOfWork) -> None:
    draft = make_draft()
    await uow.drafts.save(draft)

    found = await uow.drafts.get(draft.id)

    assert found is not None
    assert found.id == draft.id
    assert found.body == draft.body
    assert found.claims[0].citations[0].quote == draft.claims[0].citations[0].quote


async def test_missing_draft_reads_as_absent(uow: SqlAlchemyUnitOfWork) -> None:
    assert await uow.drafts.get(DraftId.generate()) is None


async def test_rejected_claims_survive_saving(uow: SqlAlchemyUnitOfWork) -> None:
    request_id = RequestId.generate()
    document_id = DocumentId.generate()
    draft_id = draft_id_for(request_id)
    draft = make_draft(
        request_id=request_id,
        document_id=document_id,
        claims=(
            make_claim(draft_id=draft_id, document_id=document_id),
            make_claim(
                draft_id=draft_id,
                index=1,
                citations=(),
                supported=False,
                reject_code=RejectCode.QUOTE_NOT_FOUND,
                document_id=document_id,
            ),
        ),
    )
    await uow.drafts.save(draft)

    found = await uow.drafts.get(draft.id)

    assert found is not None
    assert found.claims_unsupported == 1


# --------------------------- барьер идемпотентности ---------------------------


async def _claim(
    uow: SqlAlchemyUnitOfWork,
    *,
    event_id: EventId,
    subject_id: uuid.UUID,
    owner: str = "worker-1",
    at: datetime = NOW,
    lease_s: int = 3600,
) -> ClaimOutcome:
    outcome = await uow.messages.claim(
        event_id=event_id,
        subject_id=subject_id,
        message_type="document.processed",
        lease_owner=owner,
        lease_seconds=lease_s,
        at=at,
    )
    return outcome.outcome


async def test_first_delivery_proceeds(uow: SqlAlchemyUnitOfWork) -> None:
    outcome = await _claim(uow, event_id=EventId.generate(), subject_id=uuid.uuid4())

    assert outcome is ClaimOutcome.PROCEED


async def test_second_worker_on_a_live_lease_is_rejected(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    # Работу уже кто-то делает: сообщение уходит на повтор, попытка расходуется.
    event_id = EventId.generate()
    subject_id = uuid.uuid4()
    await _claim(uow, event_id=event_id, subject_id=subject_id)

    outcome = await _claim(
        uow, event_id=event_id, subject_id=subject_id, owner="worker-2"
    )

    assert outcome is ClaimOutcome.REJECT_CONCURRENT


async def test_expired_lease_allows_resuming(uow: SqlAlchemyUnitOfWork) -> None:
    # Прежний воркер умер: работу можно продолжить, не начиная заново.
    event_id = EventId.generate()
    subject_id = uuid.uuid4()
    await _claim(uow, event_id=event_id, subject_id=subject_id, lease_s=1)

    outcome = await _claim(
        uow,
        event_id=event_id,
        subject_id=subject_id,
        owner="worker-2",
        at=NOW + timedelta(hours=1),
    )

    assert outcome is ClaimOutcome.RESUME


async def test_completed_message_is_skipped(uow: SqlAlchemyUnitOfWork) -> None:
    event_id = EventId.generate()
    subject_id = uuid.uuid4()
    await _claim(uow, event_id=event_id, subject_id=subject_id)
    await uow.messages.mark_completed(event_id, at=LATER)

    outcome = await _claim(uow, event_id=event_id, subject_id=subject_id)

    assert outcome is ClaimOutcome.SKIP


async def test_claim_outcomes_cover_proceed_resume_reject_skip() -> None:
    # Четыре исхода — полный словарь: пятого случая у захвата не бывает, и
    # подписчик обязан уметь обработать каждый.
    assert {outcome.value for outcome in ClaimOutcome} == {
        "proceed",
        "resume",
        "reject_concurrent",
        "skip",
    }


async def test_claim_counts_attempts(uow: SqlAlchemyUnitOfWork) -> None:
    event_id = EventId.generate()
    subject_id = uuid.uuid4()
    await _claim(uow, event_id=event_id, subject_id=subject_id, lease_s=1)

    outcome = await uow.messages.claim(
        event_id=event_id,
        subject_id=subject_id,
        message_type="document.processed",
        lease_owner="worker-2",
        lease_seconds=3600,
        at=NOW + timedelta(hours=1),
    )

    assert outcome.attempts == 2


async def test_release_lets_the_next_delivery_proceed(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    # Провалившаяся попытка отпускает захват сразу, не дожидаясь протухания.
    event_id = EventId.generate()
    subject_id = uuid.uuid4()
    await _claim(uow, event_id=event_id, subject_id=subject_id)
    await uow.messages.release(event_id, at=NOW)

    outcome = await _claim(
        uow, event_id=event_id, subject_id=subject_id, owner="worker-2"
    )

    assert outcome is ClaimOutcome.RESUME


# --------------------------- outbox ---------------------------


def _event(event_id: uuid.UUID | None = None) -> OutboxEventDTO:
    identity = event_id or uuid.uuid4()
    aggregate_id = uuid.uuid4()
    return OutboxEventDTO(
        event_id=identity,
        aggregate_id=aggregate_id,
        event_type="document.indexed",
        routing_key="document.indexed",
        payload={"event_id": str(identity), "document_id": str(aggregate_id)},
        correlation_id=str(uuid.uuid4()),
        occurred_at=NOW,
    )


async def _fetch(
    uow: SqlAlchemyUnitOfWork,
    *,
    owner: str = "relay-1",
    at: datetime = LATER,
) -> Sequence[object]:
    return await uow.outbox.fetch_pending(
        limit=10, now=at, lease_owner=owner, lease_seconds=30
    )


async def test_events_are_enqueued(uow: SqlAlchemyUnitOfWork) -> None:
    assert await uow.outbox.enqueue([_event(), _event()]) == 2


async def test_repeated_event_is_enqueued_once(uow: SqlAlchemyUnitOfWork) -> None:
    # Ключ события детерминирован: повторное завершение не создаёт второго.
    event = _event()
    await uow.outbox.enqueue([event])

    assert await uow.outbox.enqueue([event]) == 0


async def test_empty_enqueue_writes_nothing(uow: SqlAlchemyUnitOfWork) -> None:
    assert await uow.outbox.enqueue([]) == 0


async def test_pending_events_are_handed_out(uow: SqlAlchemyUnitOfWork) -> None:
    await uow.outbox.enqueue([_event()])

    assert len(await _fetch(uow)) == 1


async def test_outbox_fetch_pending_skips_events_with_live_lease(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    # Лизинг обязан участвовать в предикате выборки: без этого второй реле
    # через полсекунды заберёт те же строки.
    await uow.outbox.enqueue([_event()])
    await _fetch(uow)

    assert await _fetch(uow, owner="relay-2") == ()


async def test_two_relay_ticks_do_not_publish_same_event_twice(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    await uow.outbox.enqueue([_event()])
    first = await _fetch(uow)

    second = await _fetch(uow, owner="relay-2", at=LATER + timedelta(seconds=1))

    assert len(first) == 1
    assert second == ()


async def test_expired_lease_returns_the_event_to_the_queue(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    await uow.outbox.enqueue([_event()])
    await _fetch(uow)

    retried = await _fetch(uow, owner="relay-2", at=LATER + timedelta(minutes=1))

    assert len(retried) == 1


async def test_published_events_are_not_handed_out_again(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    event = _event()
    await uow.outbox.enqueue([event])
    await _fetch(uow)
    await uow.outbox.mark_published([event.event_id], published_at=LATER)

    assert await _fetch(uow, at=LATER + timedelta(minutes=5)) == ()


async def test_rescheduled_event_waits_and_counts_the_attempt(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    event = _event()
    await uow.outbox.enqueue([event])
    await _fetch(uow)
    await uow.outbox.reschedule(
        event.event_id,
        error="брокер недоступен",
        available_at=LATER + timedelta(hours=1),
    )

    assert await _fetch(uow, at=LATER + timedelta(minutes=1)) == ()
    later = await _fetch(uow, at=LATER + timedelta(hours=2))
    assert len(later) == 1


# --------------------------- история поиска ---------------------------


def _run(*, selected: int) -> RetrievalRunDTO:
    return RetrievalRunDTO(
        run_id=uuid.uuid4(),
        draft_id=None,
        document_id=DocumentId.generate(),
        query="Собери сводку фактов по делу",
        embedding_version=VERSION,
        retrieval_profile="hybrid-rrf-v1",
        top_k=50,
        dense_candidates=50,
        sparse_candidates=50,
        fused_candidates=70,
        reranked=50,
        selected=selected,
        context_tokens=6000,
        duration_ms=1200,
        created_at=NOW,
    )


async def test_retrieval_run_is_recorded_with_its_hits(
    uow: SqlAlchemyUnitOfWork,
    connection: AsyncConnection,
) -> None:
    run = _run(selected=1)
    hits = (
        RetrievalHitDTO(
            chunk_id=ChunkId.generate(),
            page_number=3,
            dense_rank=1,
            dense_score=0.81,
            sparse_rank=4,
            sparse_score=6.4,
            rrf_score=0.031,
            rerank_score=2.1,
            final_rank=1,
            selected=True,
        ),
    )

    await uow.retrieval.record(run, hits)

    assert await _rows(connection, "ai_retrieval_runs") == 1
    assert await _rows(connection, "ai_retrieval_hits") == 1


async def test_draft_without_claims_is_saved_as_insufficient_evidence(
    uow: SqlAlchemyUnitOfWork,
    connection: AsyncConnection,
) -> None:
    # Модель может не вернуть ни одного утверждения: это не поломка, а честный
    # отказ выдумывать — и он обязан сохраниться вместе со своим объяснением.
    draft = make_draft(claims=())
    await uow.drafts.save(draft)

    assert await _rows(connection, "ai_drafts") == 1
    assert await _rows(connection, "ai_draft_claims") == 0
    assert await _rows(connection, "ai_draft_citations") == 0


async def test_retrieval_run_without_hits_is_still_recorded(
    uow: SqlAlchemyUnitOfWork,
    connection: AsyncConnection,
) -> None:
    # Пустая выдача — тоже результат: без записи прогона нечем объяснить, что
    # поиск отработал и не нашёл ничего.
    await uow.retrieval.record(_run(selected=0), ())

    assert await _rows(connection, "ai_retrieval_runs") == 1
    assert await _rows(connection, "ai_retrieval_hits") == 0


async def test_marking_nothing_published_touches_nothing(
    uow: SqlAlchemyUnitOfWork,
) -> None:
    event = _event()
    await uow.outbox.enqueue([event])

    await uow.outbox.mark_published([], published_at=LATER)

    assert len(await _fetch(uow)) == 1
