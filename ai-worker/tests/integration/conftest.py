"""Схема на живой базе: одна на прогон, откат транзакции между тестами."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from alembic import command
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tests.conftest import (
    alembic_config,
    create_database,
    drop_database,
    dsn_for,
    unique_database_name,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def schema_dsn(base_dsn: str) -> AsyncIterator[str]:
    """Своя база со накаченной схемой на весь прогон."""
    name = unique_database_name("ai_worker_schema")
    await create_database(base_dsn, name)
    dsn = dsn_for(base_dsn, name)
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")
    try:
        yield dsn
    finally:
        await drop_database(base_dsn, name)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def engine(schema_dsn: str) -> AsyncIterator[AsyncEngine]:
    """Движок поверх базы со схемой."""
    created = create_async_engine(schema_dsn, pool_pre_ping=True)
    try:
        yield created
    finally:
        await created.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def connection(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """Соединение во внешней транзакции: она откатывается после теста.

    Так каждый тест видит пустую схему, не пересоздавая её: накат миграций
    занимает секунды, а тестов ограничений десятки.
    """
    async with engine.connect() as opened:
        transaction = await opened.begin()
        try:
            yield opened
        finally:
            await transaction.rollback()


@pytest_asyncio.fixture(loop_scope="session")
async def session_factory(
    connection: AsyncConnection,
) -> AsyncIterator[async_sessionmaker[object]]:
    """Фабрика сессий поверх откатываемой транзакции теста."""
    yield async_sessionmaker(bind=connection, expire_on_commit=False)  # type: ignore[arg-type]


@pytest.fixture
def fresh_uuid() -> uuid.UUID:
    """Случайный идентификатор для строки теста."""
    return uuid.uuid4()
