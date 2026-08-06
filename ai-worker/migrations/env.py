"""Точка входа Alembic: асинхронный движок, своя таблица версий.

Таблица версий отделена от `alembic_version` намеренно: обе цепочки живут в
одной базе, и общая таблица означала бы, что `upgrade head` одного сервиса
считает ревизии другого своими и пытается их откатить.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import async_engine_from_config

from ai_worker.infrastructure.persistence.metadata import (
    TARGET_METADATA,
    include_object,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

config = context.config

# DSN приходит либо из конфигурации (так делают тесты), либо из окружения.
_dsn = config.get_main_option("sqlalchemy.url") or os.environ.get("DATABASE__DSN", "")
config.set_main_option("sqlalchemy.url", _dsn)

VERSION_TABLE = "alembic_version_ai_worker"
STATEMENT_TIMEOUT = "300s"


def _configure(connection: Connection) -> None:
    connection.execute(text(f"SET statement_timeout = '{STATEMENT_TIMEOUT}'"))
    # Закрываем неявную транзакцию от SET: иначе миграция окажется вложенной
    # в неё и откатится при закрытии соединения.
    connection.commit()
    context.configure(
        connection=connection,
        target_metadata=TARGET_METADATA,
        version_table=VERSION_TABLE,
        include_object=include_object,
        transaction_per_migration=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_configure)
    await connectable.dispose()


def _run_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=TARGET_METADATA,
        version_table=VERSION_TABLE,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    _run_offline()
else:
    asyncio.run(_run_online())
