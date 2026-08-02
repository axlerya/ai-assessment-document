"""Хранилище корпуса: те же операции, но по локальным файлам.

Единственная подмена, которую делает стенд оценки. Всё остальное — тот же
`ProcessDocument`, что работает в проде: измерять отдельную сборку значило бы
получать числа про неё, а не про сервис.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.errors import (
    ChecksumMismatchError,
    DocumentTooLargeError,
    SourceObjectNotFoundError,
)
from document_worker.application.ports.object_storage import ObjectStatDTO
from document_worker.domain.value_objects.storage import Checksum, ChecksumAlgorithm

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.domain.value_objects.storage import ObjectRef

CONTENT_TYPE = "application/pdf"


@dataclass(frozen=True, slots=True)
class LocalCorpusStorage:
    """Читает объекты из каталога корпуса по ключу вида `<doc_id>/source.pdf`."""

    root: Path

    async def stat(self, ref: ObjectRef) -> ObjectStatDTO:
        """Метаданные объекта."""
        path = self._path_of(ref)
        return ObjectStatDTO(
            size_bytes=path.stat().st_size,
            content_type=CONTENT_TYPE,
            etag=None,
        )

    async def exists(self, ref: ObjectRef) -> bool:
        """Есть ли объект в хранилище."""
        return (self.root / ref.key).is_file()

    async def download_to(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        max_bytes: int,
        timeout_s: float,
        expected_checksum: Checksum | None = None,
    ) -> Checksum:
        """Копирует объект в рабочий каталог, повторяя проверки адаптера.

        Raises:
            DocumentTooLargeError: Объект больше допустимого предела.
            ChecksumMismatchError: Скопированное не совпало с заявленным.
        """
        del timeout_s
        path = self._path_of(ref)
        if path.stat().st_size > max_bytes:
            raise DocumentTooLargeError(
                "объект корпуса больше допустимого предела",
                context={"limit_bytes": max_bytes, "object_key": ref.key},
            )
        shutil.copyfile(path, destination)
        checksum = Checksum(ChecksumAlgorithm.SHA256, _sha256(destination))
        if expected_checksum is not None and not checksum.matches(expected_checksum):
            raise ChecksumMismatchError(
                "контрольная сумма файла корпуса не совпала",
                context={"object_key": ref.key},
            )
        return checksum

    def _path_of(self, ref: ObjectRef) -> Path:
        path = self.root / ref.key
        if not path.is_file():
            raise SourceObjectNotFoundError(
                "объекта нет в корпусе",
                context={"bucket": ref.bucket, "object_key": ref.key},
            )
        return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
