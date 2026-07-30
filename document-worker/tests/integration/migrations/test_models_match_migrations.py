"""Сверка ORM-моделей с накатанной схемой.

Модели и миграция описывают одну и ту же схему двумя независимыми текстами.
Разойтись они могут молча: запрос уйдёт по колонке, которой в базе нет, и это
выяснится в рантайме.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from document_worker.infrastructure.persistence.base import NAMING_CONVENTION
from document_worker.infrastructure.persistence.metadata import TARGET_METADATA

if TYPE_CHECKING:
    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration


def _summarize(entry: object) -> str:
    """Короткая строка вместо полного repr таблицы: иначе diff нечитаем."""
    if isinstance(entry, list):
        return "; ".join(_summarize(item) for item in entry)
    if not isinstance(entry, tuple):
        return str(entry)
    parts = [str(entry[0])]
    parts += [
        getattr(item, "name", None) or str(item)
        for item in entry[1:]
        if item is not None
    ]
    return " ".join(parts)


def _diff(connection: Connection) -> list[str]:
    context = MigrationContext.configure(
        connection,
        opts={
            "compare_type": True,
            "compare_server_default": True,
            "target_metadata": TARGET_METADATA,
        },
    )
    return [_summarize(entry) for entry in compare_metadata(context, TARGET_METADATA)]


async def test_models_metadata_matches_migrations(connection: AsyncConnection) -> None:
    diff = await connection.run_sync(_diff)

    assert diff == [], f"модели разошлись с миграцией: {diff}"


def test_naming_convention_is_shared_by_metadata() -> None:
    assert TARGET_METADATA.naming_convention == NAMING_CONVENTION
