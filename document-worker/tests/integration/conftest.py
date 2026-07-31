"""Общая инфраструктура интеграционных тестов: контейнер PostgreSQL и миграции."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from testcontainers.community.postgres import PostgresContainer

from document_worker.infrastructure.cpu.executor import CpuPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

SERVICE_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_IMAGE = "postgres:18-alpine"


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Поднимает PostgreSQL один раз на весь прогон."""
    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
def base_dsn(postgres_container: PostgresContainer) -> str:
    """DSN административной базы контейнера."""
    return str(postgres_container.get_connection_url())


def _dsn_for(base: str, database: str) -> str:
    head, _, _ = base.rpartition("/")
    return f"{head}/{database}"


def alembic_config(dsn: str) -> Config:
    """Конфигурация Alembic, направленная на указанную базу."""
    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(SERVICE_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", dsn)
    return config


async def _create_database(base: str, name: str) -> None:
    engine = create_async_engine(base, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        await engine.dispose()


async def _drop_database(base: str, name: str) -> None:
    engine = create_async_engine(base, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            )
    finally:
        await engine.dispose()


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


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def cpu_pool() -> AsyncIterator[CpuPool]:
    """Пул процессов под синхронные вызовы PDF-библиотек.

    Контекст `spawn` поднимает интерпретатор заново, поэтому пул один на весь
    прогон: иначе каждый тест платил бы за старт процесса больше, чем за разбор.
    """
    async with CpuPool(max_workers=2) as pool:
        yield pool
