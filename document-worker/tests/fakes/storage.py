"""Фейки хранилища для тестов, которым не нужен MinIO.

Не моки: поведение настоящее, только данные в памяти. Мок бы подтвердил, что
метод позвали, а не что скачанное совпало с положенным.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from document_worker.application.errors import (
    DocumentTooLargeError,
    SourceObjectNotFoundError,
)
from document_worker.application.ports.object_storage import ObjectStatDTO
from document_worker.domain.errors import ChecksumMismatch
from document_worker.domain.value_objects.storage import Checksum, ChecksumAlgorithm

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.application.ports.object_storage import ObjectStorage
    from document_worker.domain.value_objects.storage import ObjectRef


class InMemoryObjectStorage:
    """Хранилище в памяти с той же семантикой, что у S3-адаптера."""

    def __init__(self, content_type: str = "application/pdf") -> None:
        """Создаёт пустое хранилище."""
        self._objects: dict[tuple[str, str], bytes] = {}
        self._content_type = content_type
        self.downloads: list[ObjectRef] = []

    def put(self, ref: ObjectRef, payload: bytes) -> None:
        """Кладёт объект в хранилище."""
        self._objects[(ref.bucket, ref.key)] = payload

    async def stat(self, ref: ObjectRef) -> ObjectStatDTO:
        """Метаданные объекта."""
        payload = self._payload_of(ref)
        return ObjectStatDTO(
            size_bytes=len(payload),
            content_type=self._content_type,
            etag=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        )

    async def exists(self, ref: ObjectRef) -> bool:
        """Есть ли объект в хранилище."""
        return (ref.bucket, ref.key) in self._objects

    async def download_to(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        max_bytes: int,
        timeout_s: float,
        expected_checksum: Checksum | None = None,
    ) -> Checksum:
        """Скачивает объект, повторяя проверки настоящего адаптера."""
        del timeout_s
        payload = self._payload_of(ref)
        self.downloads.append(ref)
        if len(payload) > max_bytes:
            raise DocumentTooLargeError(
                "объект в хранилище больше допустимого предела",
                context={"limit_bytes": max_bytes, "object_key": ref.key},
            )
        checksum = Checksum.sha256_of(payload)
        if expected_checksum is not None and not checksum.matches(expected_checksum):
            raise ChecksumMismatch(
                expected=expected_checksum.value, actual=checksum.value
            )
        destination.write_bytes(payload)
        return Checksum(ChecksumAlgorithm.SHA256, checksum.value)

    def _payload_of(self, ref: ObjectRef) -> bytes:
        try:
            return self._objects[(ref.bucket, ref.key)]
        except KeyError as error:
            raise SourceObjectNotFoundError(
                "объекта нет в хранилище",
                context={"bucket": ref.bucket, "object_key": ref.key},
            ) from error


class FlakyObjectStorage:
    """Хранилище, которое падает первые N раз, а дальше работает.

    Нужно тестам повторов: без него «упало и починилось» не воспроизвести.
    """

    def __init__(
        self,
        inner: ObjectStorage,
        *,
        fail_times: int,
        error: Exception,
    ) -> None:
        """Оборачивает настоящее хранилище счётчиком отказов."""
        self._inner = inner
        self._left = fail_times
        self._error = error
        self.attempts = 0

    async def stat(self, ref: ObjectRef) -> ObjectStatDTO:
        """Метаданные объекта."""
        self._maybe_fail()
        return await self._inner.stat(ref)

    async def exists(self, ref: ObjectRef) -> bool:
        """Есть ли объект в хранилище."""
        self._maybe_fail()
        return await self._inner.exists(ref)

    async def download_to(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        max_bytes: int,
        timeout_s: float,
        expected_checksum: Checksum | None = None,
    ) -> Checksum:
        """Скачивает объект после того, как отказы кончатся."""
        self._maybe_fail()
        return await self._inner.download_to(
            ref,
            destination,
            max_bytes=max_bytes,
            timeout_s=timeout_s,
            expected_checksum=expected_checksum,
        )

    def _maybe_fail(self) -> None:
        self.attempts += 1
        if self._left > 0:
            self._left -= 1
            raise self._error
