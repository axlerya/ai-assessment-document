"""Обвязка стенда оценки: своя база со схемой и настройки без брокера.

Очередей стенду не нужно: он зовёт `ProcessDocument` напрямую. База нужна
настоящая — страницы и чанки сервис пишет только в неё, и читать их больше
неоткуда.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from alembic import command

from document_worker.infrastructure.config.settings import AppSettings
from tests.conftest import _create_database, _drop_database, _dsn_for, alembic_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# Хранилище настоящему стенду не нужно, но настройки обязаны быть валидными.
UNUSED_S3 = {
    "endpoint_url": "http://storage.invalid:9000",
    "access_key": "eval",
    "secret_key": "eval-secret",
    "default_bucket": "corpus",
}


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def eval_dsn(base_dsn: str) -> AsyncIterator[str]:
    """Отдельная база со схемой на весь прогон стенда."""
    name = f"docworker_eval_{uuid.uuid4().hex[:8]}"
    await _create_database(base_dsn, name)
    dsn = _dsn_for(base_dsn, name)
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")
    try:
        yield dsn
    finally:
        await _drop_database(base_dsn, name)


@pytest.fixture(scope="session")
def eval_settings(
    eval_dsn: str,
    model_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> AppSettings:
    """Настройки, которых хватает обработке документа без брокера."""
    return AppSettings.model_validate(
        {
            "database": {"dsn": eval_dsn},
            "rabbit": {"url": "amqp://guest:guest@broker.invalid:5672/"},
            "s3": UNUSED_S3,
            "processing": {"temp_dir": str(tmp_path_factory.mktemp("eval-work"))},
            "ocr": {"model_dir": str(model_dir)},
        }
    )
