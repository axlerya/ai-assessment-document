"""Документ ↔ строка documents."""

from __future__ import annotations

from document_worker.domain.entities.document import Document
from document_worker.domain.value_objects.enums import DocumentStatus, ProcessingStage
from document_worker.domain.value_objects.identifiers import CorrelationId, DocumentId
from document_worker.domain.value_objects.storage import (
    Checksum,
    ChecksumAlgorithm,
    FileSize,
    MimeType,
    ObjectRef,
    SourceFile,
)
from document_worker.domain.value_objects.versioning import PipelineVersion
from document_worker.infrastructure.persistence.models.document import DocumentRow


def _version_of(document: Document) -> str | None:
    version = document.pipeline_version
    return str(version) if version is not None else None


def document_to_row(document: Document) -> DocumentRow:
    """Собирает строку документа целиком."""
    return DocumentRow(**document_to_values(document))


def document_to_values(document: Document) -> dict[str, object]:
    """Значения колонок документа.

    Заявленные продюсером значения дублируются определёнными: у домена одна
    правда о файле, и хранить в declared_* что-то другое он не может.
    """
    source = document.source
    return {
        "id": document.id.value,
        "bucket": source.ref.bucket,
        "object_key": source.ref.key,
        "declared_mime_type": source.mime_type.value,
        "detected_mime_type": source.mime_type.value,
        "declared_size_bytes": int(source.size),
        "size_bytes": int(source.size),
        "checksum_algorithm": ChecksumAlgorithm.SHA256.value,
        "source_checksum": source.checksum.value if source.checksum else None,
        "checksum": source.checksum.value if source.checksum else None,
        "page_count": document.page_count,
        "status": document.status.value,
        "pipeline_version": _version_of(document),
        "correlation_id": str(document.correlation_id),
        "failure_code": document.failure_code,
        "failure_stage": document.failure_stage.value
        if document.failure_stage
        else None,
        "failure_message": document.failure_message,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "processing_started_at": document.processing_started_at,
        "processing_finished_at": document.processed_at,
    }


def apply_document_to_row(document: Document, row: DocumentRow) -> None:
    """Переносит в строку только те колонки, которыми владеет воркер.

    Строку создаёт сервис приёма файлов, поэтому заявленные значения и
    метаданные источника здесь не трогаются.
    """
    source = document.source
    row.detected_mime_type = source.mime_type.value
    row.size_bytes = int(source.size)
    row.checksum = source.checksum.value if source.checksum else None
    row.page_count = document.page_count
    row.status = document.status.value
    row.pipeline_version = _version_of(document)
    row.correlation_id = str(document.correlation_id)
    row.failure_code = document.failure_code
    row.failure_stage = document.failure_stage.value if document.failure_stage else None
    row.failure_message = document.failure_message
    row.updated_at = document.updated_at
    row.processing_started_at = document.processing_started_at
    row.processing_finished_at = document.processed_at


def document_to_domain(row: DocumentRow) -> Document:
    """Восстанавливает документ из строки.

    Снимок качества не восстанавливается: в схеме его нет, он пересчитывается
    по document_pages.
    """
    return Document(
        id=DocumentId(row.id),
        source=_source_of(row),
        status=DocumentStatus(row.status),
        pipeline_version=PipelineVersion.parse(row.pipeline_version)
        if row.pipeline_version
        else None,
        correlation_id=CorrelationId(row.correlation_id),
        created_at=row.created_at,
        updated_at=row.updated_at,
        page_count=row.page_count,
        processing_started_at=row.processing_started_at,
        processed_at=row.processing_finished_at,
        failure_code=row.failure_code,
        failure_message=row.failure_message,
        failure_stage=ProcessingStage(row.failure_stage) if row.failure_stage else None,
    )


def _source_of(row: DocumentRow) -> SourceFile:
    # Определённое значение точнее заявленного, поэтому оно и берётся первым.
    checksum = row.checksum or row.source_checksum
    return SourceFile(
        ref=ObjectRef(bucket=row.bucket, key=row.object_key),
        mime_type=MimeType(row.detected_mime_type or row.declared_mime_type),
        size=FileSize(
            row.size_bytes if row.size_bytes is not None else row.declared_size_bytes
        ),
        checksum=Checksum(ChecksumAlgorithm(row.checksum_algorithm), checksum)
        if checksum
        else None,
    )
