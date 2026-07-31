"""Фейки хранилища ведут себя как настоящий адаптер.

Фейк, расходящийся с оригиналом, зелёный на тестах и красный в бою: тут он
сверяется с теми же правилами, что проверены на MinIO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.application.errors import (
    ChecksumMismatchError,
    DocumentTooLargeError,
    SourceObjectNotFoundError,
    StorageUnavailableError,
)
from document_worker.application.ports.object_storage import ObjectStorage
from document_worker.domain.value_objects.storage import Checksum, ObjectRef
from tests.fakes.storage import FlakyObjectStorage, InMemoryObjectStorage

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

REF = ObjectRef(bucket="documents", key="a/source.pdf")
CONTENT = b"%PDF-1.7\n" + b"a" * 512
MAX_BYTES = 1024 * 1024
TIMEOUT_S = 5.0


@pytest.fixture
def storage() -> InMemoryObjectStorage:
    fake = InMemoryObjectStorage()
    fake.put(REF, CONTENT)
    return fake


def test_fakes_satisfy_the_storage_port(storage: InMemoryObjectStorage) -> None:
    assert isinstance(storage, ObjectStorage)
    assert isinstance(
        FlakyObjectStorage(storage, fail_times=0, error=RuntimeError()), ObjectStorage
    )


async def test_download_writes_the_payload_and_returns_its_checksum(
    storage: InMemoryObjectStorage,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "source.pdf"

    checksum = await storage.download_to(
        REF, destination, max_bytes=MAX_BYTES, timeout_s=TIMEOUT_S
    )

    assert destination.read_bytes() == CONTENT
    assert checksum == Checksum.sha256_of(CONTENT)


async def test_download_of_missing_object_raises_the_same_error(
    storage: InMemoryObjectStorage,
    tmp_path: Path,
) -> None:
    with pytest.raises(SourceObjectNotFoundError):
        await storage.download_to(
            ObjectRef(bucket="documents", key="missing.pdf"),
            tmp_path / "source.pdf",
            max_bytes=MAX_BYTES,
            timeout_s=TIMEOUT_S,
        )


async def test_download_respects_the_size_limit(
    storage: InMemoryObjectStorage,
    tmp_path: Path,
) -> None:
    with pytest.raises(DocumentTooLargeError):
        await storage.download_to(
            REF, tmp_path / "source.pdf", max_bytes=16, timeout_s=TIMEOUT_S
        )


async def test_download_checks_the_expected_checksum(
    storage: InMemoryObjectStorage,
    tmp_path: Path,
) -> None:
    with pytest.raises(ChecksumMismatchError):
        await storage.download_to(
            REF,
            tmp_path / "source.pdf",
            max_bytes=MAX_BYTES,
            timeout_s=TIMEOUT_S,
            expected_checksum=Checksum.sha256_of(b"another file"),
        )


async def test_stat_and_exists_tell_present_from_absent(
    storage: InMemoryObjectStorage,
) -> None:
    stat = await storage.stat(REF)

    assert stat.size_bytes == len(CONTENT)
    assert await storage.exists(REF)
    assert not await storage.exists(ObjectRef(bucket="documents", key="missing.pdf"))


async def test_flaky_storage_fails_the_given_number_of_times(
    storage: InMemoryObjectStorage,
) -> None:
    flaky = FlakyObjectStorage(
        storage, fail_times=2, error=StorageUnavailableError("хранилище недоступно")
    )

    for _ in range(2):
        with pytest.raises(StorageUnavailableError):
            await flaky.stat(REF)
    stat = await flaky.stat(REF)

    assert stat.size_bytes == len(CONTENT)
    assert flaky.attempts == 3


async def test_flaky_storage_passes_downloads_through_after_recovery(
    storage: InMemoryObjectStorage,
    tmp_path: Path,
) -> None:
    flaky = FlakyObjectStorage(
        storage, fail_times=1, error=StorageUnavailableError("хранилище недоступно")
    )
    destination = tmp_path / "source.pdf"
    with pytest.raises(StorageUnavailableError):
        await flaky.download_to(
            REF, destination, max_bytes=MAX_BYTES, timeout_s=TIMEOUT_S
        )

    checksum = await flaky.download_to(
        REF, destination, max_bytes=MAX_BYTES, timeout_s=TIMEOUT_S
    )

    assert checksum == Checksum.sha256_of(CONTENT)
    assert storage.downloads == [REF]


async def test_flaky_storage_passes_exists_through(
    storage: InMemoryObjectStorage,
) -> None:
    flaky = FlakyObjectStorage(storage, fail_times=0, error=RuntimeError())

    assert await flaky.exists(REF)
