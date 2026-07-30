"""Репозиторий чанков: пакетная вставка и сверка числа записанных строк."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from document_worker.application.errors import ChunkPersistenceMismatchError
from document_worker.domain.value_objects.storage import Checksum
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.mappers.page import page_to_row
from document_worker.infrastructure.persistence.repositories.chunks import (
    SqlAlchemyDocumentChunkRepository,
)
from tests.factories import (
    CHUNKING_VERSION,
    PAGE_TEXT,
    make_chunk,
    make_document,
    make_ocr_page,
    make_text_layer_page,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.domain.entities.document_chunk import DocumentChunk
    from document_worker.domain.entities.document_page import DocumentPage

pytestmark = pytest.mark.integration


async def _pages(session: AsyncSession) -> tuple[DocumentPage, DocumentPage]:
    document = make_document()
    session.add(document_to_row(document))
    first = make_text_layer_page(document, number=1)
    second = make_ocr_page(document, number=2)
    session.add(page_to_row(first))
    session.add(page_to_row(second))
    await session.flush()
    return first, second


def _shifted(page: DocumentPage, ordinal: int) -> DocumentChunk:
    """Чанк той же страницы с другим смещением: ключ уникальности — смещение."""
    return make_chunk(page, ordinal=ordinal, start=ordinal * len(PAGE_TEXT))


async def test_add_all_writes_all_chunks_of_document_in_one_transaction(
    session: AsyncSession,
) -> None:
    first, second = await _pages(session)
    repository = SqlAlchemyDocumentChunkRepository(session)
    chunks = [_shifted(first, 0), _shifted(first, 1), _shifted(second, 0)]

    written = await repository.add_all(chunks)

    assert written == 3
    assert await repository.count(first.document_id, CHUNKING_VERSION) == 3


async def test_add_all_numbers_chunks_densely_across_the_document(
    session: AsyncSession,
) -> None:
    # Доменный ordinal плотный внутри страницы, chunk_index — по документу;
    # мост между ними это порядок переданной последовательности.
    first, second = await _pages(session)
    repository = SqlAlchemyDocumentChunkRepository(session)

    await repository.add_all(
        [_shifted(first, 0), _shifted(first, 1), _shifted(second, 0)]
    )

    assert await _chunk_indexes(session) == [0, 1, 2]


async def test_repeated_add_all_creates_no_duplicates(session: AsyncSession) -> None:
    first, _ = await _pages(session)
    repository = SqlAlchemyDocumentChunkRepository(session)
    chunks = [_shifted(first, 0), _shifted(first, 1)]
    await repository.add_all(chunks)

    written = await repository.add_all(chunks)

    assert written == 0
    assert await repository.count(first.document_id, CHUNKING_VERSION) == 2


async def test_add_all_of_empty_sequence_writes_nothing(
    session: AsyncSession,
) -> None:
    await _pages(session)
    repository = SqlAlchemyDocumentChunkRepository(session)

    assert await repository.add_all([]) == 0


async def test_conflicting_chunk_with_other_content_is_an_error(
    session: AsyncSession,
) -> None:
    # Молчаливая потеря половины чанков начиналась ровно здесь: строка не
    # вставилась, а её содержимое отличается от уже лежащего.
    first, _ = await _pages(session)
    repository = SqlAlchemyDocumentChunkRepository(session)
    original = _shifted(first, 0)
    await repository.add_all([original])
    other_text = PAGE_TEXT[:-1] + "!"
    conflicting = replace(
        original,
        content=other_text,
        checksum=Checksum.sha256_of(other_text.encode("utf-8")),
    )

    with pytest.raises(ChunkPersistenceMismatchError):
        await repository.add_all([conflicting])


async def _chunk_indexes(session: AsyncSession) -> list[int]:
    result = await session.execute(
        text("SELECT chunk_index FROM document_chunks ORDER BY chunk_index")
    )
    return [int(row[0]) for row in result]
