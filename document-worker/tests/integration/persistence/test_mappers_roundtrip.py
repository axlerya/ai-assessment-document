"""Round-trip доменных сущностей через настоящий PostgreSQL.

Unit-версия round-trip ловит расхождение маппера с доменом, но не видит того,
что делает с данными сама СУБД: numeric округляет, varchar усекает, а CHECK
отвергает то, что домен считает допустимым. Здесь сущность едет в базу и
возвращается оттуда.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

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
from document_worker.infrastructure.persistence.models.chunk import DocumentChunkRow
from document_worker.infrastructure.persistence.models.document import DocumentRow
from document_worker.infrastructure.persistence.models.job import ProcessingJobRow
from document_worker.infrastructure.persistence.models.page import (
    DocumentPageRow,
    IllegibleSpanRow,
)
from tests.factories import (
    make_chunk,
    make_document,
    make_failed_page,
    make_illegible_page,
    make_job,
    make_ocr_page,
    make_text_layer_page,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.domain.entities.document_page import DocumentPage

pytestmark = pytest.mark.integration


async def _reload(session: AsyncSession, row: object, model: type) -> object:
    await session.flush()
    session.expire_all()
    return await session.get(model, row.id)  # type: ignore[attr-defined, arg-type]


async def test_document_roundtrip_through_postgres(session: AsyncSession) -> None:
    document = make_document()

    row = document_to_row(document)
    session.add(row)
    stored = await _reload(session, row, DocumentRow)

    assert document_to_domain(stored) == replace(document, stats=None)


@pytest.mark.parametrize(
    "page_factory",
    [make_text_layer_page, make_ocr_page, make_illegible_page, make_failed_page],
    ids=["text_layer", "ocr", "illegible", "failed"],
)
async def test_page_roundtrip_through_postgres(
    session: AsyncSession,
    page_factory: object,
) -> None:
    document = make_document()
    session.add(document_to_row(document))
    page: DocumentPage = page_factory(document)  # type: ignore[operator]

    row = page_to_row(page)
    session.add(row)
    session.add_all(page_spans_to_rows(page))
    await session.flush()
    session.expire_all()
    stored = await session.get(DocumentPageRow, row.id)
    spans = (
        (
            await session.execute(
                select(IllegibleSpanRow)
                .where(IllegibleSpanRow.page_id == row.id)
                .order_by(IllegibleSpanRow.span_index)
            )
        )
        .scalars()
        .all()
    )

    assert stored is not None
    restored = page_to_domain(stored, spans)
    assert restored.status is page.status
    assert restored.text == page.text
    assert restored.failure == page.failure
    assert restored.confidence == page.confidence
    assert restored.image_ref == page.image_ref
    assert restored.render_dpi == page.render_dpi
    assert restored.warnings == page.warnings


async def test_page_roundtrip_preserves_none_confidence_for_text_layer(
    session: AsyncSession,
) -> None:
    document = make_document()
    session.add(document_to_row(document))
    page = make_text_layer_page(document)

    row = page_to_row(page)
    session.add(row)
    stored = await _reload(session, row, DocumentPageRow)

    assert page_to_domain(stored, []).confidence is None  # type: ignore[arg-type]


async def test_chunk_roundtrip_through_postgres(session: AsyncSession) -> None:
    document = make_document()
    session.add(document_to_row(document))
    page = make_ocr_page(document)
    session.add(page_to_row(page))
    chunk = make_chunk(page)

    row = chunk_to_row(chunk, chunk_index=0)
    session.add(row)
    stored = await _reload(session, row, DocumentChunkRow)

    restored = chunk_to_domain(stored, ordinal=chunk.ordinal)  # type: ignore[arg-type]
    assert restored.content == chunk.content
    assert restored.span == chunk.span
    assert restored.checksum == chunk.checksum
    assert restored.avg_confidence == chunk.avg_confidence
    assert restored.heading_path == chunk.heading_path


async def test_job_roundtrip_through_postgres(session: AsyncSession) -> None:
    document = make_document()
    session.add(document_to_row(document))
    job = make_job(document)

    row = job_to_row(job)
    session.add(row)
    stored = await _reload(session, row, ProcessingJobRow)

    assert job_to_domain(stored) == replace(job, result_status=None)  # type: ignore[arg-type]


async def test_confidence_survives_postgres_without_losing_a_digit(
    session: AsyncSession,
) -> None:
    # numeric(4,3) отрезал бы четвёртый знак, а домен округляет до четвёртого:
    # средняя уверенность документа разошлась бы с пересчитанной по страницам.
    document = make_document()
    session.add(document_to_row(document))
    page = make_ocr_page(document, confidence=0.8123)

    row = page_to_row(page)
    session.add(row)
    stored = await _reload(session, row, DocumentPageRow)

    restored = page_to_domain(stored, [])  # type: ignore[arg-type]
    assert restored.confidence is not None
    assert restored.confidence.value == 0.8123
