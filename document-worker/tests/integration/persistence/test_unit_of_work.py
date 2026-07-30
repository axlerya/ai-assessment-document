"""Транзакционная граница: единственная точка коммита на всю обработку.

Коммит внутри репозитория опубликовал бы событие раньше, чем записан статус
документа, и оставил бы отметку об обработке без её результата.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from document_worker.application.dto.results import MessageClaimDTO, OutboxEventDTO
from document_worker.application.errors import DuplicateRecordError
from document_worker.application.ports.unit_of_work import UnitOfWork
from document_worker.domain.value_objects.identifiers import EventId
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.unit_of_work import (
    NestedUnitOfWorkError,
    SqlAlchemyUnitOfWork,
)
from tests.factories import (
    CHUNKING_VERSION,
    NOW,
    PIPELINE_VERSION,
    make_document,
    make_ocr_page,
    new_correlation_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.document_page import DocumentPage

pytestmark = pytest.mark.integration

EVENT_TYPE = "document.processed"


def _event(document: Document, *, occurred_at: datetime = NOW) -> OutboxEventDTO:
    event_id = uuid.uuid4()
    return OutboxEventDTO(
        event_id=event_id,
        aggregate_id=document.id.value,
        event_type=EVENT_TYPE,
        routing_key=EVENT_TYPE,
        payload={"event_id": str(event_id), "document_id": str(document.id)},
        correlation_id=new_correlation_id(),
        occurred_at=occurred_at,
    )


def _claim(document: Document) -> MessageClaimDTO:
    return MessageClaimDTO(
        event_id=EventId.generate(),
        document_id=document.id,
        correlation_id=new_correlation_id(),
        pipeline_version=PIPELINE_VERSION,
        message_type="document.process.requested",
        lease_owner="worker-1",
        lease_expires_at=NOW,
        claimed_at=NOW,
    )


@contextmanager
def _commit_counter() -> Iterator[list[int]]:
    """Считает коммиты сессии: репозиторий не имеет права коммитить сам."""
    commits = [0]

    def count(_session: object) -> None:
        commits[0] += 1

    event.listen(Session, "after_commit", count)
    try:
        yield commits
    finally:
        event.remove(Session, "after_commit", count)


async def _fails_midway(
    session_factory: async_sessionmaker[AsyncSession],
    document: Document,
) -> None:
    """Пишет страницу и событие, после чего падает: писаться не должно ничего."""
    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        await uow.pages.add(make_ocr_page(document))
        await uow.outbox.enqueue([_event(document)])
        raise RuntimeError


async def _adds_page_twice(
    session_factory: async_sessionmaker[AsyncSession],
    page: DocumentPage,
) -> None:
    """Вторая страница с тем же id конфликтует по первичному ключу."""
    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        await uow.pages.add(page)
        await uow.pages.add(replace(page, number=2))


async def _seed_document(session: AsyncSession, document: Document) -> None:
    session.add(document_to_row(document))
    await session.commit()


async def _count(session: AsyncSession, table: str) -> int:
    result = await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608 — имя таблицы задаёт тест
    return int(result.scalar_one())


def test_unit_of_work_satisfies_its_port(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    assert isinstance(SqlAlchemyUnitOfWork(session_factory), UnitOfWork)


async def test_commit_persists_page_and_outbox_atomically(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    document = make_document()
    await _seed_document(session, document)

    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        await uow.pages.add(make_ocr_page(document))
        await uow.outbox.enqueue([_event(document)])
        await uow.commit()

    assert await _count(session, "document_pages") == 1
    assert await _count(session, "outbox_events") == 1


async def test_exception_rolls_back_everything_including_outbox(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    document = make_document()
    await _seed_document(session, document)

    with pytest.raises(RuntimeError):
        await _fails_midway(session_factory, document)

    assert await _count(session, "document_pages") == 0
    assert await _count(session, "outbox_events") == 0


async def test_exit_without_commit_rolls_back(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    document = make_document()
    await _seed_document(session, document)

    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        await uow.pages.add(make_ocr_page(document))

    assert await _count(session, "document_pages") == 0


async def test_repository_does_not_commit_on_its_own(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    document = make_document()
    await _seed_document(session, document)

    with _commit_counter() as commits:
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.pages.add(make_ocr_page(document))
            await uow.chunks.count(document.id, CHUNKING_VERSION)
            await uow.outbox.enqueue([_event(document)])
            await uow.commit()

    assert commits[0] == 1


async def test_nested_unit_of_work_raises(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # Вложенная сессия взяла бы второе соединение из пула, невидимое для
    # первого: дедлок сам с собой.
    unit = SqlAlchemyUnitOfWork(session_factory)

    async with unit:
        with pytest.raises(NestedUnitOfWorkError):
            await unit.__aenter__()


async def test_unit_of_work_is_reusable_after_exit(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    unit = SqlAlchemyUnitOfWork(session_factory)
    async with unit:
        pass

    async with unit as reopened:
        assert reopened is unit


async def test_commit_outside_the_block_is_an_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    unit = SqlAlchemyUnitOfWork(session_factory)
    async with unit:
        pass

    with pytest.raises(NestedUnitOfWorkError):
        await unit.commit()


async def test_savepoint_keeps_transaction_usable_after_conflict(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    # В PostgreSQL любая ошибка переводит транзакцию в aborted, поэтому там,
    # где конфликт ожидаем, он обязан быть накрыт точкой отката.
    document = make_document()
    await _seed_document(session, document)
    page = make_ocr_page(document, number=1)

    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        await uow.pages.add(page)
        with pytest.raises(DuplicateRecordError):
            async with uow.savepoint():
                await uow.pages.add(replace(page, number=2))
        await uow.outbox.enqueue([_event(document)])
        await uow.commit()

    assert await _count(session, "outbox_events") == 1
    assert await _count(session, "document_pages") == 1


async def test_conflict_is_translated_to_duplicate_record_error(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    # Ожидаемые дубли гасит ON CONFLICT, поэтому долетевший конфликт означает
    # нарушенный инвариант, а не повторную доставку.
    document = make_document()
    await _seed_document(session, document)
    page = make_ocr_page(document, number=1)

    with pytest.raises(DuplicateRecordError):
        await _adds_page_twice(session_factory, page)


async def test_flush_does_not_commit(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    document = make_document()
    await _seed_document(session, document)

    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        await uow.pages.add(make_ocr_page(document))
        await uow.flush()

    assert await _count(session, "document_pages") == 0


async def test_rollback_discards_pending_changes(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    document = make_document()
    await _seed_document(session, document)

    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        await uow.pages.add(make_ocr_page(document))
        await uow.rollback()
        await uow.commit()

    assert await _count(session, "document_pages") == 0


async def test_all_six_repositories_share_one_transaction(
    session_factory: async_sessionmaker[AsyncSession],
    session: AsyncSession,
) -> None:
    document = make_document()
    await _seed_document(session, document)

    async with SqlAlchemyUnitOfWork(session_factory) as uow:
        assert await uow.documents.get(document.id) is not None
        await uow.pages.count(document.id, PIPELINE_VERSION)
        await uow.chunks.count(document.id, CHUNKING_VERSION)
        await uow.jobs.get(document.id, PIPELINE_VERSION)
        await uow.messages.try_claim(_claim(document))
        await uow.outbox.enqueue([_event(document)])
        await uow.commit()

    assert await _count(session, "processed_messages") == 1
    assert await _count(session, "outbox_events") == 1
