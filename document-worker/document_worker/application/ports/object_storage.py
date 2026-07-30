"""Хранилище объектов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.domain.value_objects.storage import Checksum, ObjectRef


@dataclass(frozen=True, slots=True)
class ObjectStatDTO:
    """Метаданные объекта без его загрузки."""

    size_bytes: int
    content_type: str | None
    etag: str | None


@runtime_checkable
class ObjectStorage(Protocol):
    """Доступ к S3-совместимому хранилищу."""

    async def stat(self, ref: ObjectRef) -> ObjectStatDTO:
        """Метаданные объекта."""
        ...

    async def exists(self, ref: ObjectRef) -> bool:
        """Есть ли объект в хранилище."""
        ...

    async def download_to(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        max_bytes: int,
        timeout_s: float,
        expected_checksum: Checksum | None = None,
    ) -> Checksum:
        """Скачивает объект, обрывая поток при превышении предела.

        Возвращает контрольную сумму фактически скачанных байт.
        """
        ...
