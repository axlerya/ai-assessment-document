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

from document_worker.infrastructure.persistence.base import (
    NAMING_CONVENTION,
    Base,
)

if TYPE_CHECKING:
    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration


def _diff(connection: Connection) -> list[tuple[object, ...]]:
    context = MigrationContext.configure(
        connection,
        opts={
            "compare_type": True,
            "compare_server_default": True,
            "target_metadata": Base.metadata,
        },
    )
    return compare_metadata(context, Base.metadata)


async def test_models_metadata_matches_migrations(connection: AsyncConnection) -> None:
    diff = await connection.run_sync(_diff)

    assert diff == [], f"модели разошлись с миграцией: {diff}"


def test_naming_convention_is_shared_by_metadata() -> None:
    assert Base.metadata.naming_convention == NAMING_CONVENTION
