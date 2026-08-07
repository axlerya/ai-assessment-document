"""База, в которой живут схемы обоих сервисов, как в бою."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import create_database, drop_database, dsn_for, unique_database_name
from tests.support.foreign_schema import apply_foreign_schema, skip_unless_supported

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncConnection


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def shared_dsn(base_dsn: str) -> AsyncIterator[str]:
    """База со схемой соседнего сервиса."""
    skip_unless_supported()
    name = unique_database_name("ai_worker_shared")
    await create_database(base_dsn, name)
    dsn = dsn_for(base_dsn, name)
    await asyncio.to_thread(apply_foreign_schema, dsn)
    try:
        yield dsn
    finally:
        await drop_database(base_dsn, name)


@pytest_asyncio.fixture(loop_scope="session")
async def foreign_connection(shared_dsn: str) -> AsyncIterator[AsyncConnection]:
    """Соединение с базой обоих сервисов во внешней транзакции теста."""
    engine = create_async_engine(shared_dsn)
    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                yield connection
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()
