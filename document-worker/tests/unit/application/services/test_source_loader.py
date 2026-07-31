"""Загрузка исходного файла: тип, размер, контрольная сумма."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from document_worker.application.config import ProcessingConfig, SourceConfig
from document_worker.application.errors import (
    ChecksumMismatchError,
    CorruptedDocumentError,
    DocumentTooLargeError,
    SourceObjectNotFoundError,
    UnsupportedMediaTypeError,
)
from document_worker.application.services.source_loader import (
    SOURCE_FILE_NAME,
    SourceDocumentLoader,
)
from document_worker.domain.value_objects.storage import Checksum, MimeType
from document_worker.domain.value_objects.versioning import PipelineVersion
from document_worker.infrastructure.storage.temp_workspace import TempDirWorkspace
from tests.factories import make_document
from tests.fakes.storage import InMemoryObjectStorage

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.domain.entities.document import Document

pytestmark = pytest.mark.unit

CONTENT = b"%PDF-1.7\n" + b"contract" * 64
MAX_BYTES = 4096


@pytest.fixture
def config() -> ProcessingConfig:
    return ProcessingConfig(
        pipeline_version=PipelineVersion(1, 0, 0),
        consumer_name="worker-1",
        document_timeout_s=60.0,
        source=SourceConfig(max_file_size_bytes=MAX_BYTES, download_timeout_s=5.0),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> TempDirWorkspace:
    return TempDirWorkspace(root=tmp_path)


def _storage(document: Document, payload: bytes = CONTENT) -> InMemoryObjectStorage:
    storage = InMemoryObjectStorage()
    storage.put(document.source.ref, payload)
    return storage


async def test_download_puts_the_file_into_the_workspace(
    config: ProcessingConfig,
    workspace: TempDirWorkspace,
) -> None:
    document = make_document()
    loader = SourceDocumentLoader(storage=_storage(document), config=config)

    loaded = await loader.load(document, workspace=workspace)

    assert loaded.path == workspace.path_for(SOURCE_FILE_NAME)
    assert loaded.path.read_bytes() == CONTENT
    assert int(loaded.size) == len(CONTENT)
    assert loaded.checksum == Checksum.sha256_of(CONTENT)


async def test_unsupported_mime_type_is_rejected_before_download(
    config: ProcessingConfig,
    workspace: TempDirWorkspace,
) -> None:
    document = make_document()
    document.source = replace(document.source, mime_type=MimeType("image/png"))
    storage = _storage(document)
    loader = SourceDocumentLoader(storage=storage, config=config)

    with pytest.raises(UnsupportedMediaTypeError):
        await loader.load(document, workspace=workspace)

    assert storage.downloads == []


async def test_oversized_object_is_rejected_by_metadata(
    config: ProcessingConfig,
    workspace: TempDirWorkspace,
) -> None:
    # Отказ по метаданным экономит трафик целого документа.
    document = make_document()
    storage = _storage(document, b"%PDF-1.7\n" + bytes(MAX_BYTES))
    loader = SourceDocumentLoader(storage=storage, config=config)

    with pytest.raises(DocumentTooLargeError):
        await loader.load(document, workspace=workspace)

    assert storage.downloads == []


async def test_checksum_mismatch_is_permanent(
    config: ProcessingConfig,
    workspace: TempDirWorkspace,
) -> None:
    document = make_document()
    document.source = replace(
        document.source, checksum=Checksum.sha256_of(b"another file")
    )
    loader = SourceDocumentLoader(storage=_storage(document), config=config)

    with pytest.raises(ChecksumMismatchError):
        await loader.load(document, workspace=workspace)


async def test_empty_file_is_rejected_after_download(
    config: ProcessingConfig,
    workspace: TempDirWorkspace,
) -> None:
    # Метаданным хранилища верить нельзя: размер сверяется по скачанному.
    document = make_document()
    loader = SourceDocumentLoader(storage=_storage(document, b""), config=config)

    with pytest.raises(CorruptedDocumentError):
        await loader.load(document, workspace=workspace)


async def test_missing_object_propagates_the_storage_error(
    config: ProcessingConfig,
    workspace: TempDirWorkspace,
) -> None:
    document = make_document()
    loader = SourceDocumentLoader(storage=InMemoryObjectStorage(), config=config)

    with pytest.raises(SourceObjectNotFoundError):
        await loader.load(document, workspace=workspace)
