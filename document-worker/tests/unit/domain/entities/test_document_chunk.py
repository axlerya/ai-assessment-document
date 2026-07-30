"""Тесты сущности чанка документа."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from document_worker.domain.entities.document_chunk import DocumentChunk
from document_worker.domain.entities.document_page import DocumentPage
from document_worker.domain.errors import ChunkSpanMismatch, InvariantViolation
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.domain.value_objects.identifiers import ChunkId, DocumentId, PageId
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.storage import Checksum
from document_worker.domain.value_objects.text import TextSpan
from document_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    PipelineVersion,
)

pytestmark = pytest.mark.unit

CHUNK_ID = ChunkId(uuid.UUID("44444444-4444-5444-9444-444444444444"))
PAGE_ID = PageId(uuid.UUID("22222222-2222-5222-9222-222222222222"))
DOCUMENT_ID = DocumentId(uuid.UUID("11111111-1111-5111-9111-111111111111"))
NUMBER = PageNumber(14)
CHUNKING_VERSION = ChunkingVersion(1, 0, 0)
CREATED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

CONTENT = "договор аренды нежилого помещения"


def _page() -> DocumentPage:
    return DocumentPage.from_text_layer(
        page_id=PAGE_ID,
        document_id=DOCUMENT_ID,
        number=NUMBER,
        pipeline_version=PipelineVersion(1, 0, 0),
        content=CONTENT,
        now=CREATED_AT,
    )


def _chunk(
    *,
    span: TextSpan | None = None,
    ordinal: int = 0,
    token_count: int = 8,
    overlap_prefix_chars: int = 0,
) -> DocumentChunk:
    return DocumentChunk.from_page_slice(
        chunk_id=CHUNK_ID,
        page=_page(),
        ordinal=ordinal,
        span=span or TextSpan(0, len(CONTENT)),
        avg_confidence=None,
        token_count=token_count,
        chunking_version=CHUNKING_VERSION,
        overlap_prefix_chars=overlap_prefix_chars,
    )


def test_content_is_taken_from_page_slice() -> None:
    chunk = _chunk(span=TextSpan(0, 7))

    assert chunk.content == "договор"
    assert chunk.char_count == 7


def test_chunk_offsets_are_page_relative() -> None:
    chunk = _chunk(span=TextSpan(0, 7))

    assert chunk.span.start == 0
    assert chunk.page_id == PAGE_ID
    assert chunk.page_number == NUMBER


def test_checksum_matches_content() -> None:
    chunk = _chunk(span=TextSpan(0, 7))

    assert chunk.checksum == Checksum.sha256_of("договор".encode())


def test_chunk_rejects_blank_content() -> None:
    with pytest.raises(InvariantViolation):
        _chunk(span=TextSpan(7, 8))


def test_chunk_rejects_span_length_mismatch() -> None:
    with pytest.raises(ChunkSpanMismatch):
        DocumentChunk(
            id=CHUNK_ID,
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=NUMBER,
            ordinal=0,
            content="договор",
            span=TextSpan(0, 10),
            method=ExtractionMethod.TEXT_LAYER,
            avg_confidence=None,
            illegible_span_count=0,
            chunking_version=CHUNKING_VERSION,
            checksum=Checksum.sha256_of("договор".encode()),
            token_count=2,
        )


def test_chunk_rejects_checksum_not_matching_content() -> None:
    with pytest.raises(InvariantViolation):
        DocumentChunk(
            id=CHUNK_ID,
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=NUMBER,
            ordinal=0,
            content="договор",
            span=TextSpan(0, 7),
            method=ExtractionMethod.TEXT_LAYER,
            avg_confidence=None,
            illegible_span_count=0,
            chunking_version=CHUNKING_VERSION,
            checksum=Checksum.sha256_of(b"other"),
            token_count=2,
        )


def test_chunk_rejects_negative_ordinal() -> None:
    with pytest.raises(InvariantViolation):
        _chunk(ordinal=-1)


@pytest.mark.parametrize("token_count", [0, 1025])
def test_chunk_rejects_token_count_outside_bounds(token_count: int) -> None:
    with pytest.raises(InvariantViolation):
        _chunk(token_count=token_count)


def test_chunk_rejects_overlap_longer_than_content() -> None:
    with pytest.raises(InvariantViolation):
        _chunk(span=TextSpan(0, 7), overlap_prefix_chars=7)


def test_chunk_accepts_overlap_shorter_than_content() -> None:
    chunk = _chunk(span=TextSpan(0, 7), overlap_prefix_chars=3)

    assert chunk.overlap_prefix_chars == 3


def test_chunk_rejects_none_extraction_method() -> None:
    with pytest.raises(InvariantViolation):
        DocumentChunk(
            id=CHUNK_ID,
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=NUMBER,
            ordinal=0,
            content="договор",
            span=TextSpan(0, 7),
            method=ExtractionMethod.NONE,
            avg_confidence=None,
            illegible_span_count=0,
            chunking_version=CHUNKING_VERSION,
            checksum=Checksum.sha256_of("договор".encode()),
            token_count=2,
        )


def test_text_layer_chunk_rejects_confidence() -> None:
    with pytest.raises(InvariantViolation):
        DocumentChunk(
            id=CHUNK_ID,
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=NUMBER,
            ordinal=0,
            content="договор",
            span=TextSpan(0, 7),
            method=ExtractionMethod.TEXT_LAYER,
            avg_confidence=OcrConfidence(1.0),
            illegible_span_count=0,
            chunking_version=CHUNKING_VERSION,
            checksum=Checksum.sha256_of("договор".encode()),
            token_count=2,
        )


def test_ocr_chunk_requires_confidence() -> None:
    with pytest.raises(InvariantViolation):
        DocumentChunk(
            id=CHUNK_ID,
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=NUMBER,
            ordinal=0,
            content="договор",
            span=TextSpan(0, 7),
            method=ExtractionMethod.OCR,
            avg_confidence=None,
            illegible_span_count=0,
            chunking_version=CHUNKING_VERSION,
            checksum=Checksum.sha256_of("договор".encode()),
            token_count=2,
        )


def test_citation_points_at_page_and_offsets() -> None:
    chunk = _chunk(span=TextSpan(0, 7))

    assert chunk.citation() == "p.14 [0:7]"


def test_chunk_follows_previous_chunk_of_same_page() -> None:
    first = _chunk(span=TextSpan(0, 7), ordinal=0)
    second = _chunk(span=TextSpan(8, 14), ordinal=1)

    assert second.follows(first)
    assert not first.follows(second)


def test_chunk_of_another_page_does_not_follow() -> None:
    other_page = PageId(uuid.UUID("55555555-5555-5555-9555-555555555555"))
    first = _chunk(span=TextSpan(0, 7), ordinal=0)
    second = DocumentChunk(
        id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        page_id=other_page,
        page_number=PageNumber(15),
        ordinal=1,
        content="договор",
        span=TextSpan(0, 7),
        method=ExtractionMethod.TEXT_LAYER,
        avg_confidence=None,
        illegible_span_count=0,
        chunking_version=CHUNKING_VERSION,
        checksum=Checksum.sha256_of("договор".encode()),
        token_count=2,
    )

    assert not second.follows(first)


def test_has_illegible_reflects_span_count() -> None:
    chunk = _chunk(span=TextSpan(0, 7))

    assert not chunk.has_illegible


def test_chunk_equality_is_by_identity() -> None:
    assert _chunk(span=TextSpan(0, 7)) == _chunk(span=TextSpan(8, 14))
