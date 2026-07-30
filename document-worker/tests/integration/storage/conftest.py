"""MinIO для тестов S3-адаптера.

Контейнер поднимается один раз на прогон: на Windows его старт заметно дороже
самих тестов.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from testcontainers.community.minio import MinioContainer

from document_worker.infrastructure.storage.s3_object_storage import (
    S3Config,
    S3ObjectStorage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

SOURCE_BUCKET = "documents"


@pytest.fixture(scope="session")
def minio_container() -> Iterator[MinioContainer]:
    """Поднимает MinIO один раз на весь прогон."""
    with MinioContainer() as container:
        yield container


@pytest.fixture(scope="session")
def s3_config(minio_container: MinioContainer) -> S3Config:
    """Параметры подключения к поднятому MinIO."""
    config = minio_container.get_config()
    return S3Config(
        endpoint_url=f"http://{config['endpoint']}",
        region="us-east-1",
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        connect_timeout_s=5.0,
        read_timeout_s=30.0,
        max_attempts=1,
    )


@pytest.fixture(scope="session")
def source_bucket(minio_container: MinioContainer) -> str:
    """Бакет с исходными документами."""
    client = minio_container.get_client()
    if not client.bucket_exists(SOURCE_BUCKET):
        client.make_bucket(SOURCE_BUCKET)
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
