"""Схема document-worker в тестовой базе.

Она накатывается его собственными миграциями, а не копией DDL в фикстуре.
Копия была бы вторым источником истины для чужой схемы: она разошлась бы с
оригиналом молча, и read-model проверялся бы против того, чего в бою нет.

Цена — тестам ai-worker нужно синхронизированное окружение соседнего сервиса.
Это ровно та зависимость, которая есть и при развёртывании: миграции
document-worker выполняются раньше (ADR-0001).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import create_database, drop_database, dsn_for, unique_database_name

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncConnection

DOCUMENT_WORKER = Path(__file__).resolve().parents[4] / "document-worker"


def _apply_foreign_schema(dsn: str) -> None:
    """Накатывает миграции document-worker его собственным окружением."""
    # Путь до `uv` разрешается явно: без него окружение соседнего сервиса не
    # поднять, и сказать об этом прямо честнее, чем упасть внутри запуска.
    executable = shutil.which("uv")
    if executable is None:  # pragma: no cover — на машине разработчика uv есть
        pytest.skip("для схемы document-worker нужен uv в PATH")
    completed = subprocess.run(  # noqa: S603
        [
            executable,
            "run",
            "--directory",
            str(DOCUMENT_WORKER),
            "alembic",
            "upgrade",
            "head",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ | {"POSTGRES__DSN": dsn},
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(
            "схема document-worker не накатилась — нужно его окружение "
            f"(`uv sync --directory {DOCUMENT_WORKER}`):\n{completed.stderr[-800:]}"
        )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def shared_dsn(base_dsn: str) -> AsyncIterator[str]:
    """База, в которой живут схемы обоих сервисов, как в бою."""
    if sys.platform not in {"win32", "linux", "darwin"}:  # pragma: no cover
        pytest.skip("нет способа запустить чужие миграции")
    name = unique_database_name("ai_worker_shared")
    await create_database(base_dsn, name)
    dsn = dsn_for(base_dsn, name)
    await asyncio.to_thread(_apply_foreign_schema, dsn)
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
