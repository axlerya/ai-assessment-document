"""Репозиторий документов: чтение, блокировка и терминальный переход."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event, text

from document_worker.domain.value_objects.enums import DocumentStatus, ProcessingStage
from document_worker.domain.value_objects.identifiers import DocumentId
from document_worker.domain.value_objects.storage import Checksum
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.repositories.documents import (
    SqlAlchemyDocumentRepository,
)
from tests.factories import NOW, PIPELINE_VERSION, make_document

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

pytestmark = pytest.mark.integration

LATER = NOW + timedelta(minutes=5)


async def _persisted(session: AsyncSession, **kwargs: object) -> DocumentId:
    document = make_document(**kwargs)  # type: ignore[arg-type]
    session.add(document_to_row(document))
    await session.flush()
    return document.id


@contextmanager
def _recorded(connection: AsyncConnection) -> Iterator[list[str]]:
    """Собирает SQL, отправленный в базу за время блока."""
    statements: list[str] = []
    engine = connection.sync_connection.engine  # type: ignore[union-attr]

    def record(*args: object) -> None:
        # Сигнатура события SQLAlchemy: conn, cursor, statement, parameters,
        # context, executemany — из них нужен только текст запроса.
        statements.append(str(args[2]))

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


async def test_add_and_get_returns_equal_document(session: AsyncSession) -> None:
    document = make_document()
    session.add(document_to_row(document))
    await session.flush()
    repository = SqlAlchemyDocumentRepository(session)

    restored = await repository.get(document.id)

    assert restored == replace(document, stats=None)


async def test_get_returns_none_for_unknown_document(session: AsyncSession) -> None:
    repository = SqlAlchemyDocumentRepository(session)

    assert await repository.get(DocumentId.generate()) is None


async def test_acquire_returns_the_document(session: AsyncSession) -> None:
    document_id = await _persisted(session)
    repository = SqlAlchemyDocumentRepository(session)

    acquired = await repository.acquire(document_id)

    assert acquired is not None
    assert acquired.id == document_id


async def test_acquire_locks_row_for_update(
    session: AsyncSession,
    connection: AsyncConnection,
) -> None:
    # Захват без блокировки допустил бы второго воркера на тот же документ
    # между чтением статуса и его записью.
    document_id = await _persisted(session)
    repository = SqlAlchemyDocumentRepository(session)

    with _recorded(connection) as statements:
        await repository.acquire(document_id)

    assert any("FOR UPDATE" in statement for statement in statements)


async def test_start_processing_moves_document_and_sets_version(
    session: AsyncSession,
) -> None:
    document_id = await _persisted(session)
    repository = SqlAlchemyDocumentRepository(session)

    started = await repository.start_processing(
        document_id, pipeline_version=PIPELINE_VERSION, at=LATER
    )

    assert started
    document = await repository.get(document_id)
    assert document is not None
    assert document.status is DocumentStatus.PROCESSING
    assert document.pipeline_version == PIPELINE_VERSION
    assert document.processing_started_at == LATER


async def test_start_processing_of_already_processing_document_returns_false(
    session: AsyncSession,
) -> None:
    document_id = await _persisted(session, status=DocumentStatus.PROCESSING)
    repository = SqlAlchemyDocumentRepository(session)

    started = await repository.start_processing(
        document_id, pipeline_version=PIPELINE_VERSION, at=LATER
    )

    assert not started


async def test_finish_writes_terminal_result(session: AsyncSession) -> None:
    document_id = await _persisted(session, status=DocumentStatus.PROCESSING)
    repository = SqlAlchemyDocumentRepository(session)
    document = await repository.get(document_id)
    assert document is not None
    document.status = DocumentStatus.PROCESSED
    document.page_count = 3
    # Контрольная сумма появляется после скачивания файла, и схема требует её
    # у успешно обработанного документа.
    document.source = replace(document.source, checksum=Checksum.sha256_of(b"pdf"))
    document.processed_at = LATER
    document.updated_at = LATER

    applied = await repository.finish(document, expected=DocumentStatus.PROCESSING)

    assert applied
    stored = await repository.get(document_id)
    assert stored is not None
    assert stored.status is DocumentStatus.PROCESSED
    assert stored.page_count == 3
    assert stored.source.checksum == Checksum.sha256_of(b"pdf")


async def test_finish_with_wrong_expected_status_changes_nothing(
    session: AsyncSession,
) -> None:
    # Ноль строк — это «кто-то уже завершил документ», а не ошибка: транзакция
    # коммитится без изменений, сообщение подтверждается.
    document_id = await _persisted(session, status=DocumentStatus.PROCESSING)
    repository = SqlAlchemyDocumentRepository(session)
    document = await repository.get(document_id)
    assert document is not None
    document.status = DocumentStatus.PROCESSED
    document.processed_at = LATER

    applied = await repository.finish(document, expected=DocumentStatus.PENDING)

    assert not applied
    stored = await repository.get(document_id)
    assert stored is not None
    assert stored.status is DocumentStatus.PROCESSING


async def test_finish_writes_failure_triple(session: AsyncSession) -> None:
    document_id = await _persisted(session, status=DocumentStatus.PROCESSING)
    repository = SqlAlchemyDocumentRepository(session)
    document = await repository.get(document_id)
    assert document is not None
    document.status = DocumentStatus.FAILED
    document.failure_code = "corrupted_document"
    document.failure_message = "файл не читается"
    document.failure_stage = ProcessingStage.VALIDATION
    document.processed_at = LATER

    applied = await repository.finish(document, expected=DocumentStatus.PROCESSING)

    assert applied
    stored = await repository.get(document_id)
    assert stored is not None
    assert stored.failure_stage is ProcessingStage.VALIDATION
    assert stored.failure_code == "corrupted_document"


async def test_finish_increments_row_version(session: AsyncSession) -> None:
    document_id = await _persisted(session, status=DocumentStatus.PROCESSING)
    repository = SqlAlchemyDocumentRepository(session)
    document = await repository.get(document_id)
    assert document is not None
    document.status = DocumentStatus.FAILED
    document.failure_code = "timeout"
    document.failure_message = "не уложились"
    document.failure_stage = ProcessingStage.OCR
    document.processed_at = LATER

    await repository.finish(document, expected=DocumentStatus.PROCESSING)

    assert await _row_version(session, document_id.value) == 1


async def _row_version(session: AsyncSession, document_id: uuid.UUID) -> int:
    result = await session.execute(
        text("SELECT version FROM documents WHERE id = :id"), {"id": document_id}
    )
    return int(result.scalar_one())
