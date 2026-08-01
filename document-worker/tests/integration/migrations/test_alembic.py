"""Тесты миграций: вверх, вниз и обратно, на пустой и заполненной базе."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import alembic_config

if TYPE_CHECKING:
    from sqlalchemy import TextClause
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration

CHARTER_TABLES = {
    "documents",
    "document_pages",
    "document_chunks",
    "document_illegible_spans",
    "processing_jobs",
    "processed_messages",
    "outbox_events",
}

CONSTRAINT_PREFIXES = ("pk__", "uq__", "ck__", "fk__", "ix__")
POSTGRES_IDENTIFIER_LIMIT = 63

_TABLES_SQL = text(
    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
)
# contype 'n' это NOT NULL: с PostgreSQL 18 они попадают в каталог с
# автоименем и к конвенции отношения не имеют.
_CONSTRAINTS_SQL = text(
    "SELECT conname FROM pg_constraint c"
    " JOIN pg_namespace n ON n.oid = c.connamespace"
    " LEFT JOIN pg_class t ON t.oid = c.conrelid"
    " WHERE n.nspname = 'public' AND c.contype <> 'n'"
    " AND coalesce(t.relname, '') <> 'alembic_version'",
)
_INDEXES_SQL = text(
    "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'",
)
_ENUMS_SQL = text(
    "SELECT typname FROM pg_type t"
    " JOIN pg_namespace n ON n.oid = t.typnamespace"
    " WHERE t.typtype = 'e' AND n.nspname = 'public'",
)


async def _upgrade(dsn: str, revision: str = "head") -> None:
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), revision)


async def _downgrade(dsn: str, revision: str) -> None:
    await asyncio.to_thread(command.downgrade, alembic_config(dsn), revision)


async def _rows(dsn: str, statement: TextClause) -> list[tuple[str, ...]]:
    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(statement)
            return [tuple(str(value) for value in row) for row in result]
    finally:
        await engine.dispose()


async def test_upgrade_head_creates_all_charter_tables(empty_database: str) -> None:
    await _upgrade(empty_database)

    tables = {row[0] for row in await _rows(empty_database, _TABLES_SQL)}

    assert tables >= CHARTER_TABLES


async def test_downgrade_base_drops_everything_except_alembic_version(
    empty_database: str,
) -> None:
    await _upgrade(empty_database)

    await _downgrade(empty_database, "base")

    tables = {row[0] for row in await _rows(empty_database, _TABLES_SQL)}
    assert tables <= {"alembic_version"}


async def test_upgrade_downgrade_upgrade_roundtrip(empty_database: str) -> None:
    revisions = list(
        ScriptDirectory.from_config(alembic_config(empty_database)).walk_revisions()
    )

    await _upgrade(empty_database)
    await _downgrade(empty_database, "base")
    await _upgrade(empty_database)

    tables = {row[0] for row in await _rows(empty_database, _TABLES_SQL)}
    assert tables >= CHARTER_TABLES
    assert revisions, "в проекте нет ни одной миграции"


async def test_downgrade_with_existing_data_does_not_fail_on_foreign_keys(
    empty_database: str,
) -> None:
    await _upgrade(empty_database)
    engine = create_async_engine(empty_database)
    document_id = uuid.uuid4()
    try:
        async with engine.begin() as connection:
            await _insert_document(connection, document_id)
            await _insert_page(connection, document_id, uuid.uuid4())
    finally:
        await engine.dispose()

    await _downgrade(empty_database, "base")

    tables = {row[0] for row in await _rows(empty_database, _TABLES_SQL)}
    assert tables <= {"alembic_version"}


async def test_no_native_enum_types_are_created(empty_database: str) -> None:
    await _upgrade(empty_database)

    enums = await _rows(empty_database, _ENUMS_SQL)

    assert not enums, "словарные значения задаются varchar + CHECK"


async def test_all_constraint_names_follow_convention(empty_database: str) -> None:
    await _upgrade(empty_database)

    names = [row[0] for row in await _rows(empty_database, _CONSTRAINTS_SQL)]

    wrong = [name for name in names if not name.startswith(CONSTRAINT_PREFIXES)]
    assert not wrong, f"имена вне конвенции: {wrong}"


async def test_all_constraint_names_fit_63_bytes(empty_database: str) -> None:
    await _upgrade(empty_database)

    names = [row[0] for row in await _rows(empty_database, _CONSTRAINTS_SQL)]
    names += [row[0] for row in await _rows(empty_database, _INDEXES_SQL)]

    too_long = [
        name for name in names if len(name.encode()) > POSTGRES_IDENTIFIER_LIMIT
    ]
    assert not too_long, f"PostgreSQL молча усечёт эти имена: {too_long}"


async def test_partial_indexes_have_predicates(empty_database: str) -> None:
    await _upgrade(empty_database)

    indexes = await _rows(empty_database, _INDEXES_SQL)
    partial = {
        name: definition
        for name, definition in indexes
        if name
        in {
            "ix__documents__stale_processing",
            "ix__outbox_events__unpublished",
            "ix__processed_messages__stale",
            "uq__processing_jobs__active",
        }
    }

    assert len(partial) == 4, f"частичные индексы не созданы: {sorted(partial)}"
    without_predicate = [name for name, sql in partial.items() if " WHERE " not in sql]
    assert not without_predicate


async def test_single_alembic_head(empty_database: str) -> None:
    script = ScriptDirectory.from_config(alembic_config(empty_database))

    assert len(script.get_heads()) == 1


async def _insert_document(connection: AsyncConnection, document_id: uuid.UUID) -> None:
    await connection.execute(
        text(
            "INSERT INTO documents (id, bucket, object_key, declared_mime_type,"
            " declared_size_bytes, correlation_id) VALUES (:id, 'documents',"
            " 'a/source.pdf', 'application/pdf', 1024, 'trace-0000-0001')"
        ),
        {"id": document_id},
    )


async def _insert_page(
    connection: AsyncConnection,
    document_id: uuid.UUID,
    page_id: uuid.UUID,
) -> None:
    await connection.execute(
        text(
            "INSERT INTO document_pages (id, document_id, pipeline_version,"
            " page_number, status, extraction_method, text, text_length)"
            " VALUES (:id, :document_id, '1.0.0', 1, 'extracted', 'text_layer',"
            " 'договор', 7)"
        ),
        {"id": page_id, "document_id": document_id},
    )
