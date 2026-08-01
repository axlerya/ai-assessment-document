"""Фикстуры, специфичные для интеграционных тестов.

Контейнеры и общие помощники живут в корневом conftest: те же PostgreSQL,
RabbitMQ и MinIO нужны и сквозным сценариям.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from alembic import command
from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from document_worker.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork
from tests.conftest import _create_database, _drop_database, _dsn_for, alembic_config

if TYPE_CHECKING:
    from tests.conftest import Management

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

    from document_worker.application.ports.unit_of_work import (
        UnitOfWork,
        UnitOfWorkFactory,
    )


@pytest_asyncio.fixture(loop_scope="session")
async def empty_database(base_dsn: str) -> AsyncIterator[str]:
    """Пустая база под один тест: миграции применяет сам тест."""
    name = f"docworker_{uuid.uuid4().hex[:12]}"
    await _create_database(base_dsn, name)
    try:
        yield _dsn_for(base_dsn, name)
    finally:
        await _drop_database(base_dsn, name)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def migrated_engine(base_dsn: str) -> AsyncIterator[AsyncEngine]:
    """База со схемой, накатанной один раз на весь прогон."""
    name = f"docworker_schema_{uuid.uuid4().hex[:8]}"
    await _create_database(base_dsn, name)
    dsn = _dsn_for(base_dsn, name)
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")
    engine = create_async_engine(dsn)
    try:
        yield engine
    finally:
        await engine.dispose()
        await _drop_database(base_dsn, name)


@pytest_asyncio.fixture(loop_scope="session")
async def connection(migrated_engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """Соединение в транзакции, которая всегда откатывается."""
    async with migrated_engine.connect() as active:
        transaction = await active.begin()
        try:
            yield active
        finally:
            await transaction.rollback()


@pytest.fixture
def session_factory(connection: AsyncConnection) -> async_sessionmaker[AsyncSession]:
    """Фабрика сессий поверх откатываемой транзакции теста.

    `create_savepoint` обязателен: без него `commit()` в коде под тестом
    зафиксировал бы внешнюю транзакцию и утёк бы в следующий тест.
    """
    return async_sessionmaker(
        bind=connection,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )


@pytest_asyncio.fixture(loop_scope="session")
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Сессия для тестов, которым нужны ORM-модели, а не голый SQL."""
    async with session_factory() as active:
        yield active


@pytest.fixture
def isolated_vhost(management: Management) -> Iterator[str]:
    """Свой vhost на тест: очереди у тестов доставки одноимённые."""
    name = f"test-{uuid.uuid4().hex[:12]}"
    management.create_vhost(name)
    try:
        yield name
    finally:
        management.delete_vhost(name)


@pytest_asyncio.fixture
async def broker(rabbitmq_url: str) -> AsyncIterator[RabbitBroker]:
    """Подключённый брокер; топологию объявляет сам тест."""
    connected = RabbitBroker(rabbitmq_url)
    await connected.connect()
    try:
        yield connected
    finally:
        await connected.stop()


@pytest.fixture
def uow_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> UnitOfWorkFactory:
    """Фабрика единиц работы поверх откатываемой транзакции теста.

    Таймаут выражения задаётся движком, а тестовая транзакция уже открыта,
    поэтому здесь он не применяется.
    """

    def factory(
        *,
        statement_timeout_ms: int,
        read_only: bool = False,
    ) -> AbstractAsyncContextManager[UnitOfWork]:
        del statement_timeout_ms, read_only
        return SqlAlchemyUnitOfWork(session_factory)

    return factory
