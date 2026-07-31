"""Чанкование документа: чтение вне транзакции, вставка одной пачкой."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from document_worker.application.dto.commands import CreateDocumentChunksCommand
from document_worker.application.dto.results import CreateDocumentChunksResult
from document_worker.application.errors import DomainInvariantViolationError
from document_worker.application.use_cases.create_document_chunks import (
    CreateDocumentChunks,
)
from document_worker.domain.chunking.blocks import BlockKind
from document_worker.domain.chunking.chunk_assembler import ChunkDraft
from document_worker.domain.chunking.quality import ChunkQualityEvaluator
from document_worker.domain.constants import MAX_CHUNK_TOKENS
from document_worker.domain.value_objects.enums import PageStatus
from document_worker.domain.value_objects.text import TextSpan
from document_worker.infrastructure.chunking.runner import CpuPoolChunkingRunner
from document_worker.infrastructure.persistence.mappers.chunk import chunk_to_domain
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.mappers.job import job_to_row
from document_worker.infrastructure.persistence.mappers.page import (
    page_spans_to_values,
    page_to_row,
)
from document_worker.infrastructure.persistence.models.chunk import DocumentChunkRow
from document_worker.infrastructure.persistence.models.page import IllegibleSpanRow
from tests.factories import (
    make_document,
    make_failed_page,
    make_illegible_page,
    make_job,
    make_text_layer_page,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.chunking.policy import ChunkingPolicy
    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.document_chunk import DocumentChunk
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.value_objects.identifiers import JobId
    from document_worker.infrastructure.cpu.executor import CpuPool
    from tests.fakes.system import FixedClock, SequentialIdGenerator

pytestmark = pytest.mark.integration

CLAUSE = (
    "Статья 1. Предмет договора\n"
    "1.1 Исполнитель обязуется поставить товар в согласованный срок.\n"
    "1.2 Заказчик обязуется принять товар и оплатить его стоимость."
)


@pytest.fixture
def use_case(
    uow_factory: UnitOfWorkFactory,
    cpu_pool: CpuPool,
    clock: FixedClock,
    ids: SequentialIdGenerator,
    config: ProcessingConfig,
) -> CreateDocumentChunks:
    """Use case на настоящих репозиториях и настоящем токенизаторе."""
    return CreateDocumentChunks(
        uow_factory=uow_factory,
        chunker=CpuPoolChunkingRunner(pool=cpu_pool),
        ids=ids,
        clock=clock,
        config=config,
    )


class Prepared:
    """Документ вместе со своими страницами и открытым прогоном."""

    def __init__(
        self,
        document: Document,
        pages: list[DocumentPage],
        job_id: JobId,
    ) -> None:
        """Запоминает подготовленное состояние."""
        self.document = document
        self.pages = pages
        self.job_id = job_id

    @property
    def command(self) -> CreateDocumentChunksCommand:
        """Команда чанкования этого документа."""
        return CreateDocumentChunksCommand(
            document_id=self.document.id,
            correlation_id=self.document.correlation_id,
            job_id=self.job_id,
        )


async def prepare(
    session: AsyncSession,
    *contents: str,
    with_illegible: bool = False,
    with_failed: bool = False,
) -> Prepared:
    """Кладёт в базу документ, его страницы и прогон обработки."""
    document = make_document()
    session.add(document_to_row(document))
    pages: list[DocumentPage] = [
        make_text_layer_page(document, number=number, content=content)
        for number, content in enumerate(contents, start=1)
    ]
    if with_illegible:
        pages.append(make_illegible_page(document, number=len(pages) + 1))
    if with_failed:
        pages.append(make_failed_page(document, number=len(pages) + 1))
    for page in pages:
        session.add(page_to_row(page))
        # Диапазоны неразборчивости живут своей таблицей: без них страница
        # читается обратно со статусом, противоречащим пустому списку.
        for values in page_spans_to_values(page):
            session.add(IllegibleSpanRow(**values))
    job = make_job(document)
    session.add(job_to_row(job))
    await session.commit()
    return Prepared(document, pages, job.id)


async def stored_chunks(
    session: AsyncSession,
    document: Document,
) -> list[DocumentChunk]:
    """Чанки, прочитанные обратно из PostgreSQL."""
    rows = await session.execute(
        select(DocumentChunkRow)
        .where(DocumentChunkRow.document_id == document.id.value)
        .order_by(DocumentChunkRow.chunk_index)
    )
    return [chunk_to_domain(row, ordinal=row.chunk_index) for row in rows.scalars()]


async def test_creates_chunks_for_all_readable_pages(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
) -> None:
    prepared = await prepare(session, CLAUSE, CLAUSE)

    result = await use_case.execute(prepared.command)

    assert result.chunks_created >= len(prepared.pages)
    assert result.chunks_total == result.chunks_created


async def test_repeated_run_creates_no_duplicates(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
) -> None:
    # Повторная доставка чанкует документ заново; дубли гасит уникальность пары
    # «страница, смещение», а итог берётся из базы, а не из числа вставленных.
    prepared = await prepare(session, CLAUSE)

    first = await use_case.execute(prepared.command)
    second = await use_case.execute(prepared.command)

    assert second.chunks_created == 0
    assert second.chunks_total == first.chunks_total


async def test_chunks_illegible_page_instead_of_dropping_it(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
) -> None:
    # Устав требует сохранить исходный результат распознавания и пометить
    # диапазон, а не выбросить страницу.
    prepared = await prepare(session, CLAUSE, with_illegible=True)
    illegible = next(
        page for page in prepared.pages if page.status is PageStatus.PARTIALLY_ILLEGIBLE
    )

    await use_case.execute(prepared.command)

    chunks = await stored_chunks(session, prepared.document)
    assert any(chunk.page_id == illegible.id for chunk in chunks)


async def test_skips_pages_without_text(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
) -> None:
    prepared = await prepare(session, CLAUSE, with_failed=True)
    failed = next(page for page in prepared.pages if page.status is PageStatus.FAILED)

    await use_case.execute(prepared.command)

    chunks = await stored_chunks(session, prepared.document)
    assert all(chunk.page_id != failed.id for chunk in chunks)


async def test_document_without_pages_yields_no_chunks(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
) -> None:
    prepared = await prepare(session)

    result = await use_case.execute(prepared.command)

    assert result == CreateDocumentChunksResult(chunks_created=0, chunks_total=0)


async def test_chunk_text_matches_page_slice_read_back_from_database(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
) -> None:
    # Инвариант проверяется на строке из PostgreSQL, а не на объекте в памяти:
    # расхождение здесь означает процитированный не тот фрагмент документа.
    prepared = await prepare(session, CLAUSE)

    await use_case.execute(prepared.command)

    by_id = {page.id: page for page in prepared.pages}
    for chunk in await stored_chunks(session, prepared.document):
        source = by_id[chunk.page_id].text.content
        assert source[chunk.span.start : chunk.span.end] == chunk.content


async def test_propagates_page_metadata_into_chunks(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
    config: ProcessingConfig,
) -> None:
    prepared = await prepare(session, CLAUSE)

    await use_case.execute(prepared.command)

    by_id = {page.id: page for page in prepared.pages}
    for chunk in await stored_chunks(session, prepared.document):
        page = by_id[chunk.page_id]
        assert chunk.method is page.method
        assert chunk.page_number == page.number
        assert chunk.document_id == prepared.document.id
        assert chunk.chunking_version == config.chunking.version


async def test_chunk_index_is_dense_enumeration_of_document_ordered_chunks(
    session: AsyncSession,
    use_case: CreateDocumentChunks,
) -> None:
    # Сквозной номер строится по всему документу, а не по остатку страниц:
    # нумерация по остатку заняла бы чужие индексы и потеряла бы половину.
    prepared = await prepare(session, CLAUSE, CLAUSE, CLAUSE)

    await use_case.execute(prepared.command)

    rows = await session.execute(
        select(DocumentChunkRow.chunk_index, DocumentChunkRow.page_number)
        .where(DocumentChunkRow.document_id == prepared.document.id.value)
        .order_by(DocumentChunkRow.chunk_index)
    )
    listed = list(rows)
    assert [row.chunk_index for row in listed] == list(range(len(listed)))
    assert [row.page_number for row in listed] == sorted(
        row.page_number for row in listed
    )


class BrokenChunker:
    """Возвращает черновик, нарушающий инвариант сущности."""

    async def run(
        self,
        pages: Sequence[DocumentPage],
        policy: ChunkingPolicy,  # noqa: ARG002 — порт требует этот параметр
    ) -> tuple[ChunkDraft, ...]:
        """Отдаёт черновик с числом токенов выше потолка чанка."""
        page = pages[0]
        span = TextSpan(0, len(page.text.content))
        return (
            ChunkDraft(
                page_id=page.id,
                page_number=page.number,
                ordinal=0,
                span=span,
                text=span.slice_of(page.text.content),
                token_count=MAX_CHUNK_TOKENS + 1,
                heading_path=(),
                overlap_prefix_chars=0,
                kind=BlockKind.PARAGRAPH,
                quality=ChunkQualityEvaluator().evaluate(
                    page=page, span=span, own_tokens=1
                ),
            ),
        )


async def test_broken_draft_becomes_a_permanent_error(
    session: AsyncSession,
    uow_factory: UnitOfWorkFactory,
    clock: FixedClock,
    ids: SequentialIdGenerator,
    config: ProcessingConfig,
) -> None:
    # Нарушенный инвариант домена обязан прийти в presentation классифицированным:
    # иначе сообщение уйдёт на бесконечный повтор вместо очереди разбора.
    prepared = await prepare(session, CLAUSE)
    use_case = CreateDocumentChunks(
        uow_factory=uow_factory,
        chunker=BrokenChunker(),
        ids=ids,
        clock=clock,
        config=config,
    )

    with pytest.raises(DomainInvariantViolationError):
        await use_case.execute(prepared.command)
