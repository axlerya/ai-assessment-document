"""Перевод состояния индексации в строку и обратно."""

from __future__ import annotations

from ai_worker.domain.entities.document_index import DocumentIndex
from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
from ai_worker.domain.value_objects.enums import IndexStatus, SourceStatus
from ai_worker.domain.value_objects.identifiers import (
    CorrelationId,
    DocumentId,
    EventId,
    IndexId,
)
from ai_worker.domain.value_objects.source_snapshot import SourceSnapshot
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PipelineVersion,
)
from ai_worker.infrastructure.persistence.models.index import DocumentIndexRow


def index_to_row(index: DocumentIndex) -> DocumentIndexRow:
    """Готовит прогон индексации к записи."""
    return DocumentIndexRow(
        id=index.id.value,
        document_id=index.document_id.value,
        embedding_version=str(index.embedding.version),
        chunking_version=str(index.source.chunking_version),
        pipeline_version=str(index.source.pipeline_version),
        model_name=index.embedding.model_name,
        status=index.status.value,
        source_status=index.source.status.value,
        chunks_total=index.chunks_total,
        chunks_embedded=index.chunks_embedded,
        chunks_failed=index.chunks_failed,
        source_event_id=index.source_event_id.value,
        correlation_id=None
        if index.correlation_id is None
        else index.correlation_id.value,
        failure_code=index.failure_code,
        failure_message=index.failure_message,
        started_at=index.started_at,
        finished_at=index.finished_at,
    )


def row_to_index(row: DocumentIndexRow) -> DocumentIndex:
    """Восстанавливает прогон индексации из строки."""
    return DocumentIndex(
        id=IndexId(row.id),
        document_id=DocumentId(row.document_id),
        embedding=EmbeddingIdentity(
            version=EmbeddingVersion.parse(row.embedding_version),
            model_name=row.model_name,
        ),
        source=SourceSnapshot(
            pipeline_version=PipelineVersion.parse(row.pipeline_version),
            chunking_version=ChunkingVersion.parse(row.chunking_version),
            status=SourceStatus(row.source_status),
        ),
        source_event_id=EventId(row.source_event_id),
        status=IndexStatus(row.status),
        correlation_id=None
        if row.correlation_id is None
        else CorrelationId(row.correlation_id),
        chunks_total=row.chunks_total,
        chunks_embedded=row.chunks_embedded,
        chunks_failed=row.chunks_failed,
        failure_code=row.failure_code,
        failure_message=row.failure_message,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )
