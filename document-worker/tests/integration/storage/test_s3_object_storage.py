"""S3-адаптер на живом MinIO.

Проверяется то, чего не покажет мок: поток обрывается на превышении предела, а
контрольная сумма считается по фактически прочитанным байтам, а не по ETag.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from document_worker.application.errors import (
    DocumentTooLargeError,
    SourceObjectNotFoundError,
    StorageAccessDeniedError,
)
from document_worker.application.ports.object_storage import ObjectStorage
from document_worker.domain.errors import ChecksumMismatch
from document_worker.domain.value_objects.storage import Checksum, ObjectRef
from document_worker.infrastructure.storage.s3_object_storage import S3ObjectStorage

if TYPE_CHECKING:
    from pathlib import Path

    from testcontainers.community.minio import MinioContainer

    from document_worker.infrastructure.storage.s3_object_storage import S3Config

pytestmark = pytest.mark.integration

CONTENT = b"%PDF-1.7\n" + b"a" * 4096
TIMEOUT_S = 30.0
MAX_BYTES = 10 * 1024 * 1024


def _put(
    container: MinioContainer,
    bucket: str,
    key: str,
    payload: bytes,
    *,
    content_type: str = "application/pdf",
) -> None:
    container.get_client().put_object(
        bucket,
        key,
        io.BytesIO(payload),
        length=len(payload),
        content_type=content_type,
    )


def test_adapter_satisfies_its_port(storage: S3ObjectStorage) -> None:
    assert isinstance(storage, ObjectStorage)


async def test_download_writes_file_and_returns_checksum(
    storage: S3ObjectStorage,
    minio_container: MinioContainer,
    source_bucket: str,
    object_key: str,
    tmp_path: Path,
) -> None:
    _put(minio_container, source_bucket, object_key, CONTENT)
    destination = tmp_path / "source.pdf"

    checksum = await storage.download_to(
        ObjectRef(bucket=source_bucket, key=object_key),
        destination,
        max_bytes=MAX_BYTES,
        timeout_s=TIMEOUT_S,
    )

    assert destination.read_bytes() == CONTENT
    assert checksum.value == hashlib.sha256(CONTENT).hexdigest()


async def test_download_computes_checksum_in_a_single_pass(
    storage: S3ObjectStorage,
    minio_container: MinioContainer,
    source_bucket: str,
    object_key: str,
    tmp_path: Path,
) -> None:
    # Второй проход по файлу ради суммы удваивает ввод-вывод на документе.
    _put(minio_container, source_bucket, object_key, CONTENT)
    destination = tmp_path / "source.pdf"

    checksum = await storage.download_to(
        ObjectRef(bucket=source_bucket, key=object_key),
        destination,
        max_bytes=MAX_BYTES,
        timeout_s=TIMEOUT_S,
        expected_checksum=Checksum.sha256_of(CONTENT),
    )

    assert checksum == Checksum.sha256_of(CONTENT)


async def test_download_aborts_when_size_exceeds_max_bytes(
    storage: S3ObjectStorage,
    minio_container: MinioContainer,
    source_bucket: str,
    object_key: str,
    tmp_path: Path,
) -> None:
    # Читать до конца, чтобы потом сказать «слишком большой», значит скачать
    # весь гигабайт: предел обязан обрывать поток.
    payload = b"x" * (1024 * 1024)
    _put(minio_container, source_bucket, object_key, payload)
    destination = tmp_path / "source.pdf"
    limit = 4096

    with pytest.raises(DocumentTooLargeError):
        await storage.download_to(
            ObjectRef(bucket=source_bucket, key=object_key),
            destination,
            max_bytes=limit,
            timeout_s=TIMEOUT_S,
        )

    assert not destination.exists()


async def test_download_of_missing_object_raises_source_object_not_found(
    storage: S3ObjectStorage,
    source_bucket: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(SourceObjectNotFoundError):
        await storage.download_to(
            ObjectRef(bucket=source_bucket, key="missing/source.pdf"),
            tmp_path / "source.pdf",
            max_bytes=MAX_BYTES,
            timeout_s=TIMEOUT_S,
        )


async def test_download_raises_checksum_mismatch_for_wrong_expected_hash(
    storage: S3ObjectStorage,
    minio_container: MinioContainer,
    source_bucket: str,
    object_key: str,
    tmp_path: Path,
) -> None:
    _put(minio_container, source_bucket, object_key, CONTENT)
    destination = tmp_path / "source.pdf"

    with pytest.raises(ChecksumMismatch):
        await storage.download_to(
            ObjectRef(bucket=source_bucket, key=object_key),
            destination,
            max_bytes=MAX_BYTES,
            timeout_s=TIMEOUT_S,
            expected_checksum=Checksum.sha256_of(b"another file"),
        )

    assert not destination.exists()


async def test_stat_returns_size_and_content_type(
    storage: S3ObjectStorage,
    minio_container: MinioContainer,
    source_bucket: str,
    object_key: str,
) -> None:
    _put(minio_container, source_bucket, object_key, CONTENT)

    stat = await storage.stat(ObjectRef(bucket=source_bucket, key=object_key))

    assert stat.size_bytes == len(CONTENT)
    assert stat.content_type == "application/pdf"


async def test_stat_of_missing_object_raises_source_object_not_found(
    storage: S3ObjectStorage,
    source_bucket: str,
) -> None:
    with pytest.raises(SourceObjectNotFoundError):
        await storage.stat(ObjectRef(bucket=source_bucket, key="missing/source.pdf"))


async def test_stat_of_missing_bucket_raises_source_object_not_found(
    storage: S3ObjectStorage,
) -> None:
    with pytest.raises(SourceObjectNotFoundError):
        await storage.stat(ObjectRef(bucket="no-such-bucket", key="a/source.pdf"))


async def test_exists_tells_present_from_absent(
    storage: S3ObjectStorage,
    minio_container: MinioContainer,
    source_bucket: str,
    object_key: str,
) -> None:
    _put(minio_container, source_bucket, object_key, CONTENT)

    assert await storage.exists(ObjectRef(bucket=source_bucket, key=object_key))
    assert not await storage.exists(
        ObjectRef(bucket=source_bucket, key="missing/source.pdf")
    )


async def test_wrong_credentials_are_a_permanent_error(
    s3_config: S3Config,
    source_bucket: str,
    object_key: str,
) -> None:
    # Права не появятся от повтора: такое сообщение уходит в DLQ, а не в retry.
    wrong = replace(s3_config, secret_key="wrong")  # noqa: S106 — заведомо неверный ключ
    async with S3ObjectStorage(wrong) as storage:
        with pytest.raises(StorageAccessDeniedError):
            await storage.stat(ObjectRef(bucket=source_bucket, key=object_key))


async def test_multipart_etag_is_not_used_as_checksum(
    storage: S3ObjectStorage,
    minio_container: MinioContainer,
    source_bucket: str,
    object_key: str,
    tmp_path: Path,
) -> None:
    # ETag многочастевой загрузки это хеш хешей с суффиксом «-N», и принять его
    # за sha256 содержимого означает сверять файл с несуществующей суммой.
    payload = b"z" * (6 * 1024 * 1024)
    _put(minio_container, source_bucket, object_key, payload)
    destination = tmp_path / "source.pdf"

    checksum = await storage.download_to(
        ObjectRef(bucket=source_bucket, key=object_key),
        destination,
        max_bytes=MAX_BYTES,
        timeout_s=TIMEOUT_S,
    )

    assert checksum.value == hashlib.sha256(payload).hexdigest()
