"""Чанк ↔ строка document_chunks.

Номеров два: доменный `ordinal` плотный внутри страницы, колонка `chunk_index`
плотная по документу. Мост между ними — порядок последовательности, которую
репозиторий нумерует перечислением, поэтому оба номера приходят снаружи.
"""

from __future__ import annotations

from decimal import Decimal

from document_worker.domain.entities.document_chunk import DocumentChunk
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.domain.value_objects.identifiers import (
    ChunkId,
    DocumentId,
    PageId,
)
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.storage import Checksum, ChecksumAlgorithm
from document_worker.domain.value_objects.text import TextSpan
from document_worker.domain.value_objects.versioning import ChunkingVersion
from document_worker.infrastructure.persistence.models.chunk import DocumentChunkRow


def chunk_to_row(chunk: DocumentChunk, *, chunk_index: int) -> DocumentChunkRow:
    """Собирает строку чанка со сквозным номером по документу."""
    return DocumentChunkRow(**chunk_to_values(chunk, chunk_index=chunk_index))


def chunk_to_values(chunk: DocumentChunk, *, chunk_index: int) -> dict[str, object]:
    """Значения колонок чанка."""
    return {
        "id": chunk.id.value,
        "document_id": chunk.document_id.value,
        "page_id": chunk.page_id.value,
        "page_number": int(chunk.page_number),
        "chunking_version": str(chunk.chunking_version),
        "chunk_index": chunk_index,
        "start_offset": chunk.span.start,
        "end_offset": chunk.span.end,
        "text": chunk.content,
        "token_count": chunk.token_count,
        "overlap_prefix_chars": chunk.overlap_prefix_chars,
        "extraction_method": chunk.method.value,
        "avg_ocr_confidence": None
        if chunk.avg_confidence is None
        else Decimal(str(chunk.avg_confidence.value)),
        "illegible_span_count": chunk.illegible_span_count,
        "heading_path": list(chunk.heading_path),
        "content_hash": chunk.checksum.value,
    }


def chunk_to_domain(row: DocumentChunkRow, *, ordinal: int) -> DocumentChunk:
    """Восстанавливает чанк из строки с номером внутри страницы."""
    return DocumentChunk(
        id=ChunkId(row.id),
        document_id=DocumentId(row.document_id),
        page_id=PageId(row.page_id),
        page_number=PageNumber(row.page_number),
        ordinal=ordinal,
        content=row.text,
        span=TextSpan(row.start_offset, row.end_offset),
        method=ExtractionMethod(row.extraction_method),
        avg_confidence=None
        if row.avg_ocr_confidence is None
        else OcrConfidence(float(row.avg_ocr_confidence)),
        illegible_span_count=row.illegible_span_count,
        chunking_version=ChunkingVersion.parse(row.chunking_version),
        checksum=Checksum(ChecksumAlgorithm.SHA256, row.content_hash),
        token_count=row.token_count,
        heading_path=tuple(row.heading_path),
        overlap_prefix_chars=row.overlap_prefix_chars,
    )
