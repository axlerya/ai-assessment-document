"""Репозиторий страниц: идемпотентная вставка, возобновление и чтение."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.domain.value_objects.enums import ExtractionMethod, PageStatus
from document_worker.domain.value_objects.versioning import PipelineVersion
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.repositories.pages import (
    SqlAlchemyDocumentPageRepository,
)
from tests.factories import (
    PIPELINE_VERSION,
    make_document,
    make_failed_page,
    make_illegible_page,
    make_ocr_page,
    make_text_layer_page,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.domain.entities.document import Document

pytestmark = pytest.mark.integration

NEWER_VERSION = PipelineVersion(2, 0, 0)


async def _document(session: AsyncSession) -> Document:
    document = make_document()
    session.add(document_to_row(document))
    await session.flush()
    return document


async def test_add_writes_page_with_its_illegible_spans(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    page = make_illegible_page(document)

    added = await repository.add(page)

    assert added
    pages = await repository.load_pages(
        document.id,
        PIPELINE_VERSION,
        statuses=frozenset({PageStatus.PARTIALLY_ILLEGIBLE}),
    )
    assert len(pages) == 1
    assert pages[0].illegible_spans == page.illegible_spans


async def test_add_is_idempotent_on_conflict(session: AsyncSession) -> None:
    # Повторная доставка гасится уникальным ограничением, а не проверкой в коде.
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    page = make_ocr_page(document)
    await repository.add(page)

    added_again = await repository.add(page)

    assert not added_again
    assert await repository.count(document.id, PIPELINE_VERSION) == 1


async def test_add_of_second_page_with_same_number_is_rejected_quietly(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    await repository.add(make_ocr_page(document, number=1))

    added = await repository.add(make_text_layer_page(document, number=1))

    assert not added


async def test_list_persisted_page_numbers_returns_only_current_version(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    await repository.add(make_ocr_page(document, number=2))
    await repository.add(make_text_layer_page(document, number=1))

    numbers = await repository.list_persisted_page_numbers(
        document.id, PIPELINE_VERSION
    )

    assert numbers == frozenset({1, 2})


async def test_list_persisted_page_numbers_of_unknown_version_is_empty(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    await repository.add(make_ocr_page(document))

    numbers = await repository.list_persisted_page_numbers(document.id, NEWER_VERSION)

    assert numbers == frozenset()


async def test_list_summaries_carries_metrics_without_text(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    page = make_illegible_page(document)
    await repository.add(page)

    summaries = await repository.list_summaries(document.id, PIPELINE_VERSION)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.page_id == page.id
    assert summary.status is PageStatus.PARTIALLY_ILLEGIBLE
    assert summary.method is ExtractionMethod.HYBRID
    assert summary.char_count == page.char_count
    assert summary.illegible_char_count == page.text.illegible_char_count


async def test_list_summaries_returns_pages_in_page_number_order(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    await repository.add(make_illegible_page(document, number=3))
    await repository.add(make_text_layer_page(document, number=1))
    await repository.add(make_ocr_page(document, number=2))

    summaries = await repository.list_summaries(document.id, PIPELINE_VERSION)

    assert [int(summary.page_number) for summary in summaries] == [1, 2, 3]


async def test_load_pages_returns_only_requested_statuses(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    await repository.add(make_text_layer_page(document, number=1))
    await repository.add(make_failed_page(document, number=4))

    pages = await repository.load_pages(
        document.id, PIPELINE_VERSION, statuses=frozenset({PageStatus.FAILED})
    )

    assert [int(page.number) for page in pages] == [4]
    assert pages[0].failure is not None


async def test_load_pages_returns_pages_in_page_number_order(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    await repository.add(make_ocr_page(document, number=2))
    await repository.add(make_text_layer_page(document, number=1))

    pages = await repository.load_pages(
        document.id,
        PIPELINE_VERSION,
        statuses=frozenset({PageStatus.EXTRACTED}),
    )

    assert [int(page.number) for page in pages] == [1, 2]


async def test_load_pages_of_empty_document_returns_nothing(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)

    pages = await repository.load_pages(
        document.id,
        PIPELINE_VERSION,
        statuses=frozenset({PageStatus.EXTRACTED}),
    )

    assert pages == ()


async def test_count_counts_only_pages_of_this_version(
    session: AsyncSession,
) -> None:
    document = await _document(session)
    repository = SqlAlchemyDocumentPageRepository(session)
    await repository.add(make_ocr_page(document))

    assert await repository.count(document.id, PIPELINE_VERSION) == 1
    assert await repository.count(document.id, NEWER_VERSION) == 0
