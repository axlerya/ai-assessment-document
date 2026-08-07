"""Сценарий индексации документа: границы транзакций и идемпотентность."""

from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from ai_worker.application.dto.commands import IndexDocumentCommand
from ai_worker.application.dto.messaging import ClaimOutcome
from ai_worker.application.errors import PermanentError, TransientError
from ai_worker.application.services.message_claim import (
    MESSAGE_TYPE,
    MessageClaimService,
)
from ai_worker.application.use_cases.index_document import IndexDocument
from ai_worker.domain.embedding.policy import DEFAULT_EMBEDDING_POLICY
from ai_worker.domain.value_objects.enums import IndexStatus
from ai_worker.domain.value_objects.identifiers import DocumentId, EventId
from ai_worker.domain.value_objects.versioning import EmbeddingVersion
from tests.integration.use_cases.conftest import (
    NOW,
    FakeEmbeddings,
    FrozenClock,
    processing_config,
)
from tests.support.document_corpus import (
    CHUNK_TEXTS,
    add_chunks,
    page_of,
    seed_document,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncConnection

    from ai_worker.application.config import ProcessingConfig
    from ai_worker.infrastructure.persistence.read_model.processed_chunks import (
        SqlAlchemyProcessedChunkReader,
    )
    from ai_worker.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork

pytestmark = pytest.mark.integration

type UnitOfWorkFactory = Callable[[], SqlAlchemyUnitOfWork]


def _use_case(
    *,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
    provider: FakeEmbeddings,
    config: ProcessingConfig | None = None,
    clock: FrozenClock | None = None,
) -> IndexDocument:
    prepared = config or processing_config()
    ticking = clock or FrozenClock(datetime.fromisoformat(NOW))
    return IndexDocument(
        claims=MessageClaimService(
            uow_factory=uow_factory, reader=reader, clock=ticking, config=prepared
        ),
        uow_factory=uow_factory,
        reader=reader,
        embeddings=provider,
        clock=ticking,
        config=prepared,
    )


def _command(
    document_id: uuid.UUID, *, event_id: EventId | None = None
) -> IndexDocumentCommand:
    return IndexDocumentCommand(
        event_id=event_id or EventId.generate(),
        document_id=DocumentId(document_id),
        correlation_id=None,
        occurred_at=datetime.fromisoformat(NOW),
    )


async def _count(
    connection: AsyncConnection, table: str, document_id: uuid.UUID
) -> int:
    row = (
        await connection.execute(
            text(f"SELECT count(*) AS total FROM {table} WHERE document_id = :id"),
            {"id": document_id},
        )
    ).one()
    return int(row.total)


async def _index_row(
    connection: AsyncConnection, document_id: uuid.UUID
) -> tuple[str, int, int, int]:
    row = (
        await connection.execute(
            text(
                "SELECT status, chunks_total, chunks_embedded, chunks_failed"
                " FROM ai_document_index WHERE document_id = :id"
            ),
            {"id": document_id},
        )
    ).one()
    return (
        str(row.status),
        int(row.chunks_total),
        int(row.chunks_embedded),
        int(row.chunks_failed),
    )


async def _outbox(
    connection: AsyncConnection, document_id: uuid.UUID
) -> list[dict[str, object]]:
    rows = await connection.execute(
        text(
            "SELECT event_id, event_type, routing_key, payload, correlation_id"
            " FROM ai_outbox_events WHERE aggregate_id = :id ORDER BY id"
        ),
        {"id": document_id},
    )
    return [dict(row._mapping) for row in rows]


async def test_document_is_indexed_end_to_end_within_the_use_case(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    provider = FakeEmbeddings(connection=indexing_connection)

    result = await _use_case(reader=reader, uow_factory=uow_factory, provider=provider)(
        _command(document_id)
    )

    assert result.outcome is ClaimOutcome.PROCEED
    assert result.status is IndexStatus.INDEXED
    assert (result.chunks_total, result.chunks_embedded, result.chunks_failed) == (
        3,
        3,
        0,
    )
    assert await _count(indexing_connection, "ai_chunk_embeddings", document_id) == 3
    assert await _index_row(indexing_connection, document_id) == ("indexed", 3, 3, 0)


async def test_terminal_transaction_writes_state_and_event_atomically(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Статус, событие и отметка сообщения — одна транзакция (ADR-0006): иначе
    # бывает документ, помеченный готовым, о котором никто не узнал.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    command = _command(document_id)

    await _use_case(reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings())(
        command
    )

    events = await _outbox(indexing_connection, document_id)
    marked = (
        await indexing_connection.execute(
            text("SELECT status FROM ai_processed_messages WHERE event_id = :id"),
            {"id": command.event_id.value},
        )
    ).one()
    assert len(events) == 1
    assert events[0]["event_type"] == "document.indexed"
    assert str(marked.status) == "completed"


async def test_nothing_is_written_when_the_terminal_transaction_fails(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Отказ на записи события обязан откатить и статус: «готов, но не
    # опубликован» — это состояние, которого не бывает.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)

    def broken() -> SqlAlchemyUnitOfWork:
        unit = uow_factory()
        unit.outbox = _RefusingOutbox()  # type: ignore[assignment]
        return unit

    with pytest.raises(RuntimeError, match="outbox"):
        await _use_case(reader=reader, uow_factory=broken, provider=FakeEmbeddings())(
            _command(document_id)
        )

    assert await _index_row(indexing_connection, document_id) == ("indexing", 0, 0, 0)


class _RefusingOutbox:
    """Outbox, который не принимает событий."""

    async def enqueue(self, events: object) -> int:
        del events
        msg = "outbox недоступен"
        raise RuntimeError(msg)


async def test_second_delivery_creates_no_duplicate_embeddings(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    use_case = _use_case(
        reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings()
    )

    await use_case(_command(document_id))
    await use_case(_command(document_id))

    assert await _count(indexing_connection, "ai_chunk_embeddings", document_id) == 3


async def test_second_delivery_publishes_no_second_event(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    use_case = _use_case(
        reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings()
    )

    await use_case(_command(document_id))
    second = await use_case(_command(document_id))

    assert second.outcome is ClaimOutcome.SKIP
    assert len(await _outbox(indexing_connection, document_id)) == 1


async def test_second_delivery_does_not_run_the_model_again(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Устав прямо это требует: не пересчитывать эмбеддинги без изменения
    # текста или версии модели.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    provider = FakeEmbeddings()
    use_case = _use_case(reader=reader, uow_factory=uow_factory, provider=provider)

    await use_case(_command(document_id))
    calls_after_first = len(provider.batches)
    await use_case(_command(document_id))

    assert len(provider.batches) == calls_after_first


async def test_chunk_with_unchanged_hash_is_not_re_embedded(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Прерванный прогон возобновляется с того места, где встал: уже
    # построенные эмбеддинги в модель повторно не идут.
    document_id = uuid.uuid4()
    await seed_document(
        indexing_connection, document_id=document_id, texts=CHUNK_TEXTS[:2]
    )
    provider = FakeEmbeddings()
    await _use_case(reader=reader, uow_factory=uow_factory, provider=provider)(
        _command(document_id)
    )

    await add_chunks(
        indexing_connection,
        document_id=document_id,
        page_id=await page_of(indexing_connection, document_id),
        texts=[CHUNK_TEXTS[2]],
    )
    resumed = FakeEmbeddings()
    await _use_case(reader=reader, uow_factory=uow_factory, provider=resumed)(
        _command(document_id)
    )

    assert resumed.embedded_texts == ()


async def test_version_bump_creates_a_parallel_namespace(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Смена мажорной части открывает новый namespace: старые эмбеддинги
    # остаются, переиндексация идёт рядом (ADR-0004).
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    await _use_case(reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings())(
        _command(document_id)
    )

    next_policy = dataclasses.replace(
        DEFAULT_EMBEDDING_POLICY, version=EmbeddingVersion(2, 0, 0)
    )
    await _use_case(
        reader=reader,
        uow_factory=uow_factory,
        provider=FakeEmbeddings(),
        config=processing_config(policy=next_policy),
    )(_command(document_id))

    assert await _count(indexing_connection, "ai_chunk_embeddings", document_id) == 6
    versions = await indexing_connection.execute(
        text("SELECT count(*) AS total FROM ai_document_index WHERE document_id = :id"),
        {"id": document_id},
    )
    assert int(versions.one().total) == 2


async def test_failed_chunk_does_not_fail_the_document(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Один нечитаемый фрагмент не имеет права стоить всего документа.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    provider = FakeEmbeddings(unindexable=frozenset({CHUNK_TEXTS[1]}))

    result = await _use_case(reader=reader, uow_factory=uow_factory, provider=provider)(
        _command(document_id)
    )

    assert result.status is IndexStatus.INDEXED
    assert (result.chunks_embedded, result.chunks_failed) == (2, 1)
    assert await _count(indexing_connection, "ai_chunk_embeddings", document_id) == 2


async def test_document_with_all_chunks_failed_is_marked_failed(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Иначе поиск нашёл бы по документу ровно ничего, а документ считался бы
    # готовым.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    provider = FakeEmbeddings(unindexable=frozenset(CHUNK_TEXTS))

    result = await _use_case(reader=reader, uow_factory=uow_factory, provider=provider)(
        _command(document_id)
    )

    assert result.status is IndexStatus.FAILED
    assert await _index_row(indexing_connection, document_id) == ("failed", 0, 0, 0)
    assert await _outbox(indexing_connection, document_id) == []


async def test_no_unit_of_work_is_open_during_embedding(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Инференс занимает минуты: держать на это время открытую транзакцию —
    # то же, от чего отказался document-worker (ADR-0006).
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    provider = FakeEmbeddings(connection=indexing_connection)

    await _use_case(reader=reader, uow_factory=uow_factory, provider=provider)(
        _command(document_id)
    )

    assert provider.units_open
    assert not any(provider.units_open)


async def test_texts_go_to_the_model_in_batches_of_the_configured_size(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    provider = FakeEmbeddings()

    await _use_case(
        reader=reader,
        uow_factory=uow_factory,
        provider=provider,
        config=processing_config(batch_size=2),
    )(_command(document_id))

    assert [len(batch) for batch in provider.batches] == [2, 1]


async def test_highest_chunking_version_is_indexed(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Версия чанкования в событии не приходит и читается из базы (ADR-0008);
    # брать первую попавшуюся значило бы индексировать устаревший корпус.
    document_id = uuid.uuid4()
    await seed_document(
        indexing_connection, document_id=document_id, texts=CHUNK_TEXTS[:1]
    )
    await add_chunks(
        indexing_connection,
        document_id=document_id,
        page_id=await page_of(indexing_connection, document_id),
        texts=CHUNK_TEXTS[1:],
        chunking_version="2.0.0",
    )
    provider = FakeEmbeddings()

    result = await _use_case(reader=reader, uow_factory=uow_factory, provider=provider)(
        _command(document_id)
    )

    assert result.chunks_total == 2
    assert set(provider.embedded_texts) == set(CHUNK_TEXTS[1:])


async def test_event_payload_carries_its_own_event_id(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Проверка базы требует `payload ->> 'event_id' = event_id` (ADR-0007):
    # потребитель дедуплицирует по телу, а не по заголовку.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)

    await _use_case(reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings())(
        _command(document_id)
    )

    (event,) = await _outbox(indexing_connection, document_id)
    payload = event["payload"]
    assert isinstance(payload, dict)
    assert payload["event_id"] == str(event["event_id"])
    assert payload["document_id"] == str(document_id)
    assert payload["embedding_version"] == str(DEFAULT_EMBEDDING_POLICY.version)
    assert payload["chunking_version"] == "1.0.0"
    assert payload["model_name"] == DEFAULT_EMBEDDING_POLICY.model_name
    assert payload["chunks_embedded"] == 3


async def test_document_without_a_row_is_a_transient_error(
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Сообщение обогнало коммит соседнего сервиса: повтор через несколько
    # секунд его увидит.
    with pytest.raises(TransientError, match="документ"):
        await _use_case(
            reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings()
        )(_command(uuid.uuid4()))


async def test_document_without_chunks_is_a_permanent_error(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id, texts=())

    with pytest.raises(PermanentError, match="чанк"):
        await _use_case(
            reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings()
        )(_command(document_id))


async def test_message_held_by_another_worker_is_rejected(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Попытка расходуется намеренно: иначе живой лиз зависшего воркера гонял
    # бы сообщение по первой ступени без предела.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    command = _command(document_id)
    await indexing_connection.execute(
        text(
            "INSERT INTO ai_processed_messages (event_id, subject_id, message_type,"
            " status, lease_owner, lease_expires_at, attempts)"
            " VALUES (:event_id, :subject_id, :message_type, 'in_progress',"
            " 'worker-2', now() + interval '30 minutes', 1)"
        ),
        {
            "event_id": command.event_id.value,
            "subject_id": document_id,
            "message_type": MESSAGE_TYPE,
        },
    )

    with pytest.raises(TransientError, match="другим воркером"):
        await _use_case(
            reader=reader, uow_factory=uow_factory, provider=FakeEmbeddings()
        )(command)


async def test_failed_run_releases_its_claim(
    indexing_connection: AsyncConnection,
    reader: SqlAlchemyProcessedChunkReader,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Следующая доставка получает возобновление немедленно, а не через
    # таймаут лиза.
    document_id = uuid.uuid4()
    await seed_document(indexing_connection, document_id=document_id)
    command = _command(document_id)

    def broken() -> SqlAlchemyUnitOfWork:
        unit = uow_factory()
        unit.outbox = _RefusingOutbox()  # type: ignore[assignment]
        return unit

    with pytest.raises(RuntimeError):
        await _use_case(reader=reader, uow_factory=broken, provider=FakeEmbeddings())(
            command
        )

    row = (
        await indexing_connection.execute(
            text(
                "SELECT status, lease_expires_at FROM ai_processed_messages"
                " WHERE event_id = :id"
            ),
            {"id": command.event_id.value},
        )
    ).one()
    assert str(row.status) == "pending"
