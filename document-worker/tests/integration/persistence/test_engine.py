"""Движок базы: пул, таймауты и фабрика единиц работы.

Таймаут выражения — последняя защита от транзакции, повисшей на блокировке:
без него занятая строка держит соединение до перезапуска процесса.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from document_worker.infrastructure.persistence.engine import (
    build_engine,
    build_session_factory,
    build_unit_of_work_factory,
)
from document_worker.infrastructure.persistence.unit_of_work import (
    apply_transaction_settings,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.integration

POOL_SIZE = 2
STATEMENT_TIMEOUT_MS = 300
NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)


@pytest.fixture
async def engine(migrated_engine: AsyncEngine) -> AsyncIterator[AsyncEngine]:
    """Свой движок поверх той же базы, что и остальные тесты."""
    built = build_engine(
        str(migrated_engine.url.render_as_string(hide_password=False)),
        pool_size=POOL_SIZE,
        max_overflow=0,
        pool_timeout_s=5.0,
    )
    try:
        yield built
    finally:
        await built.dispose()


async def test_statement_timeout_stops_a_hanging_transaction(
    engine: AsyncEngine,
) -> None:
    # Транзакция, повисшая на блокировке, иначе держит соединение пула до
    # перезапуска процесса.
    session_factory = build_session_factory(engine)

    async with session_factory() as session:
        _ = await session.begin()
        await apply_transaction_settings(
            session, statement_timeout_ms=STATEMENT_TIMEOUT_MS, read_only=False
        )
        with pytest.raises(DBAPIError):
            await session.execute(text("SELECT pg_sleep(2)"))


async def test_statement_timeout_does_not_leak_into_the_next_transaction(
    engine: AsyncEngine,
) -> None:
    # Значение ставится на транзакцию: уехав в соединение, оно осталось бы на
    # следующем, кто возьмёт его из пула.
    session_factory = build_session_factory(engine)

    async with session_factory() as first:
        _ = await first.begin()
        await apply_transaction_settings(
            first, statement_timeout_ms=STATEMENT_TIMEOUT_MS, read_only=False
        )
    async with session_factory() as second:
        _ = await second.begin()
        applied = await second.execute(text("SHOW statement_timeout"))
        assert applied.scalar_one() != f"{STATEMENT_TIMEOUT_MS}ms"


async def test_read_only_transaction_refuses_to_write(engine: AsyncEngine) -> None:
    # Отчётная транзакция, случайно что-то записавшая, — это молчаливая порча
    # данных; база обязана отказать сама.
    session_factory = build_session_factory(engine)

    async with session_factory() as session:
        _ = await session.begin()
        await apply_transaction_settings(
            session, statement_timeout_ms=5_000, read_only=True
        )
        with pytest.raises(DBAPIError, match="read-only"):
            await session.execute(text("CREATE TABLE probe (id int)"))


async def test_unit_of_work_factory_produces_working_units(
    engine: AsyncEngine,
) -> None:
    factory = build_unit_of_work_factory(build_session_factory(engine))

    async with factory(statement_timeout_ms=5_000, read_only=True) as uow:
        assert await uow.outbox.oldest_pending_age_s(now=NOW) is None
