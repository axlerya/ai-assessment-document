"""MinIO для тестов S3-адаптера.

Контейнер поднимается один раз на прогон: на Windows его старт заметно дороже
самих тестов.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from document_worker.infrastructure.storage.s3_object_storage import S3ObjectStorage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from minio import Minio

    from document_worker.infrastructure.storage.s3_object_storage import S3Config

SOURCE_BUCKET = "documents"


@pytest.fixture(scope="session")
def source_bucket(minio_client: Minio) -> str:
    """Бакет с исходными документами."""
    if not minio_client.bucket_exists(SOURCE_BUCKET):
        minio_client.make_bucket(SOURCE_BUCKET)
    return SOURCE_BUCKET


@pytest_asyncio.fixture(loop_scope="session")
async def storage(s3_config: S3Config) -> AsyncIterator[S3ObjectStorage]:
    """Адаптер, направленный на поднятый MinIO."""
    async with S3ObjectStorage(s3_config) as adapter:
        yield adapter


@pytest.fixture
def object_key() -> str:
    """Уникальный ключ, чтобы тесты не мешали друг другу."""
    return f"{uuid.uuid4().hex}/source.pdf"
