"""Миграции: обратимость, изоляция цепочки и отсутствие чужих объектов."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ai_worker.infrastructure.persistence.metadata import TARGET_METADATA
from tests.conftest import (
    alembic_config,
    create_database,
    drop_database,
    dsn_for,
    unique_database_name,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.integration

VERSION_TABLE = "alembic_version_ai_worker"
EXPECTED_TABLES = frozenset(
    {
        "ai_document_index",
        "ai_chunk_embeddings",
        "ai_drafts",
        "ai_draft_claims",
        "ai_draft_citations",
        "ai_retrieval_runs",
        "ai_retrieval_hits",
        "ai_processed_messages",
        "ai_outbox_events",
    }
)


@pytest_asyncio.fixture(loop_scope="session")
async def empty_database(base_dsn: str) -> AsyncIterator[str]:
    """Одноразовая пустая база: миграции проверяются с чистого листа."""
    name = unique_database_name("ai_worker_migr")
    await create_database(base_dsn, name)
    try:
        yield dsn_for(base_dsn, name)
    finally:
        await drop_database(base_dsn, name)


async def _tables(dsn: str) -> set[str]:
    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            return {row[0] for row in rows}
    finally:
        await engine.dispose()


async def test_migration_applies_to_an_empty_database(empty_database: str) -> None:
    # ADR-0021: схема самодостаточна. Внешние ключи на таблицы document-worker
    # сделали бы этот накат невозможным без чужих миграций.
    await asyncio.to_thread(command.upgrade, alembic_config(empty_database), "head")

    assert (await _tables(empty_database)) >= EXPECTED_TABLES


async def test_upgrade_and_downgrade_are_reversible(empty_database: str) -> None:
    config = alembic_config(empty_database)

    await asyncio.to_thread(command.upgrade, config, "head")
    await asyncio.to_thread(command.downgrade, config, "base")

    left = await _tables(empty_database)
    assert not (EXPECTED_TABLES & left), f"после отката остались таблицы: {left}"


async def test_version_table_is_isolated_from_document_worker(
    empty_database: str,
) -> None:
    # Обе цепочки живут в одной базе: общая `alembic_version` означала бы, что
    # `upgrade head` одного сервиса считает ревизии другого своими.
    await asyncio.to_thread(command.upgrade, alembic_config(empty_database), "head")

    tables = await _tables(empty_database)
    assert VERSION_TABLE in tables
    assert "alembic_version" not in tables


def test_single_alembic_head() -> None:
    # Две головы означают, что `upgrade head` неоднозначен и падает.
    heads = ScriptDirectory.from_config(alembic_config("")).get_heads()

    assert len(heads) == 1, f"голов миграций должно быть одна, есть {heads}"


async def test_schema_matches_models(empty_database: str) -> None:
    # Расхождение модели и миграции не видно ни одному тесту репозитория: они
    # работают через ORM и просто не замечают колонку, которой нет в базе.
    await asyncio.to_thread(command.upgrade, alembic_config(empty_database), "head")
    engine = create_async_engine(empty_database)
    try:
        async with engine.connect() as connection:
            diff = await connection.run_sync(_diff_against_models)
    finally:
        await engine.dispose()

    assert not diff, f"схема разошлась с моделями: {diff}"


def _diff_against_models(connection: object) -> list[object]:
    context = MigrationContext.configure(
        connection,  # type: ignore[arg-type]
        opts={"target_metadata": TARGET_METADATA, "compare_type": True},
    )
    return list(compare_metadata(context, TARGET_METADATA))


async def test_autogenerate_produces_no_operations_on_document_tables(
    empty_database: str,
) -> None:
    """Автогенерация не должна видеть чужие таблицы.

    Если `target_metadata` захватит их, первый же `--autogenerate` предложит их
    удалить — и однажды кто-нибудь согласится.
    """
    await asyncio.to_thread(command.upgrade, alembic_config(empty_database), "head")
    engine = create_async_engine(empty_database)
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text("CREATE TABLE documents (id uuid PRIMARY KEY)")
            )
            await connection.commit()
            diff = await connection.run_sync(_diff_against_models)
    finally:
        await engine.dispose()

    mentions = [entry for entry in diff if "documents" in str(entry)]
    assert not mentions, f"автогенерация трогает чужие таблицы: {mentions}"


async def test_vector_extension_is_created(empty_database: str) -> None:
    await asyncio.to_thread(command.upgrade, alembic_config(empty_database), "head")
    engine = create_async_engine(empty_database)
    try:
        async with engine.connect() as connection:
            installed = await connection.scalar(
                text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            )
    finally:
        await engine.dispose()

    assert installed == 1


async def test_downgrade_keeps_the_vector_extension(empty_database: str) -> None:
    # Расширение может быть нужно другим схемам той же базы: снимать его при
    # откате своей миграции значит ломать соседей.
    config = alembic_config(empty_database)
    await asyncio.to_thread(command.upgrade, config, "head")
    await asyncio.to_thread(command.downgrade, config, "base")

    engine = create_async_engine(empty_database)
    try:
        async with engine.connect() as connection:
            installed = await connection.scalar(
                text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            )
    finally:
        await engine.dispose()

    assert installed == 1
