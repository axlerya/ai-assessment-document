"""Загрузка исходного файла с проверками до и после скачивания.

Размер проверяется дважды: по метаданным — чтобы не начинать качать заведомо
слишком большое, и потоком — потому что метаданным хранилища верить нельзя.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.errors import (
    DocumentTooLargeError,
    UnsupportedMediaTypeError,
    translate_domain_error,
)
from document_worker.domain.errors import DomainError
from document_worker.domain.value_objects.storage import FileSize

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.ports.object_storage import ObjectStorage
    from document_worker.application.ports.system import TempWorkspace
    from document_worker.domain.entities.document import Document
    from document_worker.domain.value_objects.storage import Checksum

SOURCE_FILE_NAME = "source.pdf"


@dataclass(frozen=True, slots=True)
class LoadedSource:
    """Скачанный исходник и то, что о нём выяснилось по факту."""

    path: Path
    size: FileSize
    checksum: Checksum


@dataclass(frozen=True, slots=True)
class SourceDocumentLoader:
    """Приносит исходный файл в рабочий каталог."""

    storage: ObjectStorage
    config: ProcessingConfig

    async def load(
        self,
        document: Document,
        *,
        workspace: TempWorkspace,
    ) -> LoadedSource:
        """Скачивает исходник, проверяя тип, размер и контрольную сумму.

        Raises:
            UnsupportedMediaTypeError: Тип файла вне списка поддерживаемых.
            DocumentTooLargeError: Файл больше допустимого предела.
            ChecksumMismatchError: Скачанное не совпало с заявленным.
        """
        source = document.source
        limits = self.config.source
        if source.mime_type.value not in limits.supported_mime_types:
            raise UnsupportedMediaTypeError(
                "тип документа не поддерживается",
                context={
                    "mime_type": source.mime_type.value,
                    "supported": sorted(limits.supported_mime_types),
                },
            )

        stat = await self.storage.stat(source.ref)
        if stat.size_bytes > limits.max_file_size_bytes:
            # Отказ по метаданным экономит трафик целого документа.
            raise DocumentTooLargeError(
                "объявленный размер выше допустимого предела",
                context={
                    "size_bytes": stat.size_bytes,
                    "limit_bytes": limits.max_file_size_bytes,
                },
            )

        destination = workspace.path_for(SOURCE_FILE_NAME)
        checksum = await self.storage.download_to(
            source.ref,
            destination,
            max_bytes=limits.max_file_size_bytes,
            timeout_s=limits.download_timeout_s,
            expected_checksum=source.checksum,
        )
        size = FileSize(destination.stat().st_size)
        self._ensure_acceptable(document, size)
        return LoadedSource(path=destination, size=size, checksum=checksum)

    def _ensure_acceptable(self, document: Document, size: FileSize) -> None:
        source = document.source
        checked = type(source)(
            ref=source.ref,
            mime_type=source.mime_type,
            size=size,
            checksum=source.checksum,
        )
        try:
            checked.ensure_acceptable()
        except DomainError as error:
            raise translate_domain_error(error) from error
