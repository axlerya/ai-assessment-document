"""Соглашения схемы: имена, длина, отсутствие собственных типов."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration

NAME_PATTERN = re.compile(r"^(pk|uq|ck|fk|ix)__[a-z0-9_]+$")
MAX_IDENTIFIER_BYTES = 63

OWN_TABLES = (
    "ai_document_index",
    "ai_chunk_embeddings",
    "ai_drafts",
    "ai_draft_claims",
    "ai_draft_citations",
    "ai_retrieval_runs",
    "ai_retrieval_hits",
    "ai_processed_messages",
    "ai_outbox_events",
)


async def _constraint_names(connection: AsyncConnection) -> list[str]:
    rows = await connection.execute(
        text(
            "SELECT conname FROM pg_constraint c"
            " JOIN pg_class t ON t.oid = c.conrelid"
            " JOIN pg_namespace n ON n.oid = t.relnamespace"
            " WHERE n.nspname = 'public' AND t.relname LIKE 'ai\\_%'"
        )
    )
    return [row[0] for row in rows]


async def _index_names(connection: AsyncConnection) -> list[str]:
    rows = await connection.execute(
        text(
            "SELECT indexname FROM pg_indexes"
            " WHERE schemaname = 'public' AND tablename LIKE 'ai\\_%'"
        )
    )
    return [row[0] for row in rows]


async def test_all_constraint_names_follow_convention(
    connection: AsyncConnection,
) -> None:
    names = await _constraint_names(connection)

    assert names, "у схемы нет ни одного ограничения"
    offenders = [name for name in names if NAME_PATTERN.match(name) is None]
    assert not offenders, f"имена вне конвенции: {offenders}"


async def test_all_index_names_follow_convention(connection: AsyncConnection) -> None:
    names = await _index_names(connection)

    offenders = [name for name in names if NAME_PATTERN.match(name) is None]
    assert not offenders, f"имена индексов вне конвенции: {offenders}"


async def test_all_names_fit_63_bytes(connection: AsyncConnection) -> None:
    # PostgreSQL молча усекает длинные имена, после чего `drop_constraint` в
    # downgrade не находит объект.
    names = [*await _constraint_names(connection), *await _index_names(connection)]

    too_long = [name for name in names if len(name.encode()) > MAX_IDENTIFIER_BYTES]
    assert not too_long, f"имена длиннее {MAX_IDENTIFIER_BYTES} байт: {too_long}"


async def test_no_native_enum_types_are_created(connection: AsyncConnection) -> None:
    # `ALTER TYPE ... ADD VALUE` необратим, поэтому downgrade такой миграции
    # физически неисполним. Словари — varchar + CHECK.
    created = await connection.execute(
        text(
            "SELECT typname FROM pg_type t"
            " JOIN pg_namespace n ON n.oid = t.typnamespace"
            " WHERE n.nspname = 'public' AND t.typtype = 'e'"
        )
    )

    assert not [row[0] for row in created]


@pytest.mark.parametrize("table", OWN_TABLES)
async def test_every_table_carries_the_service_prefix(table: str) -> None:
    assert table.startswith("ai_")


async def test_no_foreign_keys_point_outside_the_service(
    connection: AsyncConnection,
) -> None:
    # ADR-0021: ссылка на чужую таблицу сделала бы схему неприменимой к пустой
    # базе и непроверяемой без миграций соседнего сервиса.
    rows = await connection.execute(
        text(
            "SELECT c.conname, rt.relname FROM pg_constraint c"
            " JOIN pg_class t ON t.oid = c.conrelid"
            " JOIN pg_class rt ON rt.oid = c.confrelid"
            " WHERE c.contype = 'f' AND t.relname LIKE 'ai\\_%'"
        )
    )

    outside = [
        f"{name} → {target}" for name, target in rows if not target.startswith("ai_")
    ]
    assert not outside, f"внешние ключи наружу сервиса: {outside}"


async def test_vector_columns_have_their_indexes(connection: AsyncConnection) -> None:
    # Без индекса обе ветви поиска вырождаются в последовательный проход по
    # всем эмбеддингам корпуса.
    rows = await connection.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'ai_chunk_embeddings'")
    )
    definitions = " ".join(row[0] for row in rows)

    assert "hnsw" in definitions
    assert "vector_cosine_ops" in definitions
    assert "sparsevec_ip_ops" in definitions
