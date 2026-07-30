"""Round-trip доменных сущностей через строки таблиц.

Ручной маппер расходится с доменом молча: сущность обрастает полем, маппер о нём
не узнаёт, и данные теряются по дороге в БД. Round-trip — единственное, что это
ловит, поэтому он идёт по сгенерированным сущностям, а не по трём примерам.
`dataclasses.replace` вызывает конструктор, то есть повторно прогоняет
инварианты восстановленной сущности.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings

from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.infrastructure.persistence.mappers.chunk import (
    chunk_to_domain,
    chunk_to_row,
)
from document_worker.infrastructure.persistence.mappers.document import (
    document_to_domain,
    document_to_row,
)
from document_worker.infrastructure.persistence.mappers.job import (
    job_to_domain,
    job_to_row,
)
from document_worker.infrastructure.persistence.mappers.page import (
    page_spans_to_rows,
    page_to_domain,
    page_to_row,
)
from tests import strategies

if TYPE_CHECKING:
    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.document_chunk import DocumentChunk
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.entities.processing_job import ProcessingJob

pytestmark = pytest.mark.unit

EXAMPLES = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@EXAMPLES
@given(strategies.documents())
def test_document_roundtrip_preserves_all_fields(document: Document) -> None:
    restored = document_to_domain(document_to_row(document))

    # stats выводятся из document_pages и в строке документа не хранятся.
    assert restored == replace(document, stats=None)


@EXAMPLES
@given(strategies.documents())
def test_document_roundtrip_result_passes_domain_invariants_again(
    document: Document,
) -> None:
    restored = document_to_domain(document_to_row(document))

    assert replace(restored).source == document.source


@EXAMPLES
@given(strategies.pages())
def test_page_roundtrip_preserves_legibility_and_failure(page: DocumentPage) -> None:
    restored = page_to_domain(page_to_row(page), page_spans_to_rows(page))

    assert restored.status is page.status
    assert restored.failure == page.failure
    assert restored.text == page.text
    assert restored.warnings == page.warnings
    assert restored.image_ref == page.image_ref
    assert restored.render_dpi == page.render_dpi
    assert restored.created_at == page.created_at


@EXAMPLES
@given(strategies.pages())
def test_page_roundtrip_preserves_none_confidence_for_text_layer(
    page: DocumentPage,
) -> None:
    restored = page_to_domain(page_to_row(page), page_spans_to_rows(page))

    if page.method in (ExtractionMethod.TEXT_LAYER, ExtractionMethod.NONE):
        assert restored.confidence is None
    else:
        assert restored.confidence == page.confidence


@EXAMPLES
@given(strategies.pages())
def test_page_roundtrip_result_passes_domain_invariants_again(
    page: DocumentPage,
) -> None:
    restored = page_to_domain(page_to_row(page), page_spans_to_rows(page))

    assert replace(restored).text == page.text


@EXAMPLES
@given(strategies.chunks())
def test_chunk_roundtrip_preserves_offsets_and_checksum(chunk: DocumentChunk) -> None:
    row = chunk_to_row(chunk, chunk_index=chunk.ordinal)

    restored = chunk_to_domain(row, ordinal=chunk.ordinal)

    # Идентичность чанка и страницы — по id, поэтому поля сверяются поимённо.
    assert restored.id == chunk.id
    assert restored.span == chunk.span
    assert restored.checksum == chunk.checksum
    assert restored.content == chunk.content
    assert restored.avg_confidence == chunk.avg_confidence
    assert restored.method is chunk.method
    assert restored.token_count == chunk.token_count
    assert restored.heading_path == chunk.heading_path
    assert restored.overlap_prefix_chars == chunk.overlap_prefix_chars
    assert restored.illegible_span_count == chunk.illegible_span_count
    assert restored.chunking_version == chunk.chunking_version
    assert restored.page_number == chunk.page_number


@EXAMPLES
@given(strategies.chunks())
def test_chunk_roundtrip_result_passes_domain_invariants_again(
    chunk: DocumentChunk,
) -> None:
    row = chunk_to_row(chunk, chunk_index=chunk.ordinal)

    restored = chunk_to_domain(row, ordinal=chunk.ordinal)

    assert replace(restored).content == chunk.content


@EXAMPLES
@given(strategies.jobs())
def test_job_roundtrip_preserves_counters_and_failure(job: ProcessingJob) -> None:
    restored = job_to_domain(job_to_row(job))

    # result_status выводится из статуса документа и в строке прогона не хранится.
    assert restored == replace(job, result_status=None)


@EXAMPLES
@given(strategies.jobs())
def test_job_roundtrip_result_passes_domain_invariants_again(
    job: ProcessingJob,
) -> None:
    restored = job_to_domain(job_to_row(job))

    assert replace(restored).attempt == job.attempt
