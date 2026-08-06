"""PostgreSQL с pgvector на весь прогон.

Образ именно с расширением: `sparsevec` и HNSW по нему появились в pgvector
0.7, и на обычном `postgres` первая же миграция не применится.

Живёт в корне тестов, а не в conftest интеграции: conftest каталога виден
только своему поддереву, а тот же контейнер понадобится сквозным сценариям.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.postgres import PostgresContainer

if TYPE_CHECKING:
    from collections.abc import Iterator

SERVICE_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_IMAGE = "pgvector/pgvector:0.8.6-pg18"


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Поднимает PostgreSQL с pgvector один раз на весь прогон."""
    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
def base_dsn(postgres_container: PostgresContainer) -> str:
    """DSN административной базы контейнера."""
    return str(postgres_container.get_connection_url())


def dsn_for(base: str, database: str) -> str:
    """Подменяет имя базы в DSN."""
    head, _, _ = base.rpartition("/")
    return f"{head}/{database}"


def alembic_config(dsn: str) -> Config:
    """Конфигурация Alembic, направленная на указанную базу."""
    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(SERVICE_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", dsn)
    return config


async def create_database(base: str, name: str) -> None:
    """Создаёт базу с указанным именем."""
    engine = create_async_engine(base, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        await engine.dispose()


async def drop_database(base: str, name: str) -> None:
    """Удаляет базу вместе со всеми соединениями."""
    engine = create_async_engine(base, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            )
    finally:
        await engine.dispose()


def unique_database_name(prefix: str) -> str:
    """Имя одноразовой базы для теста."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def run_sync(coroutine: object) -> object:
    """Выполняет корутину в новом цикле событий."""
    return asyncio.run(coroutine)  # type: ignore[arg-type]
