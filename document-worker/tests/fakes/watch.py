"""Наблюдатели портов: кто и когда вызывался относительно транзакции.

Держать соединение с базой на время чтения PDF или скачивания файла — самый
дорогой способ исчерпать пул: страница читается секундами, транзакция обязана
жить миллисекунды. Проверить это можно только со стороны, снаружи кода.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from document_worker.application.dto.pdf import (
        PdfInspectionDTO,
        PdfPageTextDTO,
        TextLayerProbeDTO,
    )
    from document_worker.application.ports.object_storage import ObjectStatDTO
    from document_worker.application.ports.pdf import PdfDocumentReader, PdfInspector
    from document_worker.application.ports.unit_of_work import (
        UnitOfWork,
        UnitOfWorkFactory,
    )
    from document_worker.domain.value_objects.storage import Checksum, ObjectRef


@dataclass(slots=True)
class TransactionWatch:
    """Считает открытые транзакции и запоминает вызовы поверх них."""

    depth: int = 0
    inside: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def note(self, name: str) -> None:
        """Отмечает вызов и то, шёл ли он внутри открытой транзакции."""
        self.calls.append(name)
        if self.depth:
            self.inside.append(name)

    def wrap(self, factory: UnitOfWorkFactory) -> UnitOfWorkFactory:
        """Оборачивает фабрику единиц работы счётчиком глубины."""

        @contextlib.asynccontextmanager
        async def watched(
            *,
            statement_timeout_ms: int,
            read_only: bool = False,
        ) -> AsyncIterator[UnitOfWork]:
            self.depth += 1
            try:
                async with factory(
                    statement_timeout_ms=statement_timeout_ms, read_only=read_only
                ) as uow:
                    yield uow
            finally:
                self.depth -= 1

        return watched


@dataclass(frozen=True, slots=True)
class WatchedStorage:
    """Хранилище объектов, докладывающее о своих вызовах."""

    inner: Any
    watch: TransactionWatch

    async def stat(self, ref: ObjectRef) -> ObjectStatDTO:
        """Метаданные объекта."""
        self.watch.note("storage.stat")
        return await self.inner.stat(ref)

    async def exists(self, ref: ObjectRef) -> bool:
        """Есть ли объект в хранилище."""
        self.watch.note("storage.exists")
        return await self.inner.exists(ref)

    async def download_to(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        max_bytes: int,
        timeout_s: float,
        expected_checksum: Checksum | None = None,
    ) -> Checksum:
        """Скачивает объект."""
        self.watch.note("storage.download")
        return await self.inner.download_to(
            ref,
            destination,
            max_bytes=max_bytes,
            timeout_s=timeout_s,
            expected_checksum=expected_checksum,
        )


@dataclass(frozen=True, slots=True)
class WatchedInspector:
    """Инспектор PDF, докладывающий о своих вызовах."""

    inner: PdfInspector
    watch: TransactionWatch

    async def inspect(self, path: Path) -> PdfInspectionDTO:
        """Читает число страниц и признаки защиты."""
        self.watch.note("pdf.inspect")
        return await self.inner.inspect(path)


@dataclass(frozen=True, slots=True)
class WatchedHandle:
    """Открытый документ, докладывающий о чтении страниц."""

    inner: Any
    watch: TransactionWatch

    async def read_page_text(self, number: int) -> PdfPageTextDTO:
        """Читает текстовый слой страницы."""
        self.watch.note("pdf.read_page")
        return await self.inner.read_page_text(number)

    async def probe(self) -> TextLayerProbeDTO:
        """Признаки пригодности текстового слоя."""
        self.watch.note("pdf.probe")
        return await self.inner.probe()


@dataclass(frozen=True, slots=True)
class WatchedReader:
    """Читатель PDF, докладывающий об открытии документа."""

    inner: PdfDocumentReader
    watch: TransactionWatch

    @contextlib.asynccontextmanager
    async def open(self, path: Path) -> AsyncIterator[WatchedHandle]:
        """Открывает документ на чтение текста."""
        self.watch.note("pdf.open")
        async with self.inner.open(path) as handle:
            yield WatchedHandle(inner=handle, watch=self.watch)
