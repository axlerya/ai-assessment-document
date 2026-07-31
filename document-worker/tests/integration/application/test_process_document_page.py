"""Обработка одной страницы: чтение вне транзакции, запись внутри."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from document_worker.application.dto.commands import (
    ExtractDocumentTextCommand,
    ProcessDocumentCommand,
    ProcessDocumentPageCommand,
)
from document_worker.application.dto.extraction import (
    DocumentExtraction,
    PagePlanEntryDTO,
)
from document_worker.application.errors import PageRenderError
from document_worker.application.services.message_claim import MessageClaimService
from document_worker.application.use_cases.extract_document_text import (
    ExtractDocumentText,
)
from document_worker.application.use_cases.process_document_page import (
    ProcessDocumentPage,
)
from document_worker.domain.normalization.normalizer import TextNormalizer
from document_worker.domain.policies.text_layer_quality import TextLayerQualityPolicy
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    PageFailureReason,
    PageStatus,
)
from document_worker.domain.value_objects.identifiers import EventId
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.infrastructure.pdf.pdfplumber_text_reader import (
    PdfPlumberDocumentReader,
)
from document_worker.infrastructure.pdf.pikepdf_inspector import PikePdfInspector
from document_worker.infrastructure.pdf.pypdfium2_page_renderer import (
    PdfiumPageRenderer,
)
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from tests.factories import make_document
from tests.fakes import pdf_builder
from tests.integration.application.conftest import NOW, PIPELINE_VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.pdf import PdfPageTextDTO, TextLayerProbeDTO
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.processing_job import ProcessingJob
    from document_worker.infrastructure.cpu.executor import CpuPool
    from tests.fakes.system import FixedClock, SequentialIdGenerator

pytestmark = pytest.mark.integration

OpenCase = Callable[["Path"], AbstractAsyncContextManager["Case"]]

MAX_PAGES = 8
MAX_PIXELS = 4_000_000
HYPHENATED_LINES = (
    "The Supplier hereby under-",
    "takes to deliver the goods within thirty days.",
)


@pytest.fixture
def extract(cpu_pool: CpuPool) -> ExtractDocumentText:
    return ExtractDocumentText(
        inspector=PikePdfInspector(pool=cpu_pool, max_pages=MAX_PAGES),
        reader=PdfPlumberDocumentReader(pool=cpu_pool),
        renderer=PdfiumPageRenderer(pool=cpu_pool, max_pixels=MAX_PIXELS),
        policy=TextLayerQualityPolicy(),
    )


@pytest.fixture
def claim_service(
    uow_factory: UnitOfWorkFactory,
    clock: FixedClock,
    ids: SequentialIdGenerator,
    config: ProcessingConfig,
) -> MessageClaimService:
    return MessageClaimService(
        uow_factory=uow_factory, clock=clock, ids=ids, config=config
    )


@pytest.fixture
def process_page(
    uow_factory: UnitOfWorkFactory,
    clock: FixedClock,
    ids: SequentialIdGenerator,
    config: ProcessingConfig,
) -> ProcessDocumentPage:
    return ProcessDocumentPage(
        uow_factory=uow_factory,
        normalizer=TextNormalizer(),
        ids=ids,
        clock=clock,
        config=config,
    )


@dataclass(frozen=True, slots=True)
class Case:
    """Захваченный документ вместе с открытым планом извлечения."""

    document: Document
    job: ProcessingJob
    extraction: DocumentExtraction

    def command(self, number: int = 1) -> ProcessDocumentPageCommand:
        """Команда обработки указанной страницы этого документа."""
        return ProcessDocumentPageCommand(
            document_id=self.document.id,
            correlation_id=self.document.correlation_id,
            job_id=self.job.id,
            entry=self.extraction.plan.pages[number - 1],
            extraction=self.extraction,
        )


@pytest.fixture
def open_case(
    session: AsyncSession,
    claim_service: MessageClaimService,
    extract: ExtractDocumentText,
) -> OpenCase:
    """Захватывает документ и открывает план извлечения указанного файла."""

    def factory(path: Path) -> AbstractAsyncContextManager[Case]:
        return _case(session, claim_service, extract, path)

    return factory


@contextlib.asynccontextmanager
async def _case(
    session: AsyncSession,
    claim_service: MessageClaimService,
    extract: ExtractDocumentText,
    path: Path,
) -> AsyncIterator[Case]:
    document = make_document()
    session.add(document_to_row(document))
    await session.commit()
    claim = await claim_service.claim(
        ProcessDocumentCommand(
            event_id=EventId.generate(),
            document_id=document.id,
            correlation_id=document.correlation_id,
            object_ref=document.source.ref,
            mime_type=document.source.mime_type,
            occurred_at=NOW,
        )
    )
    assert claim.job is not None
    async with extract.execute(
        ExtractDocumentTextCommand(
            document_id=document.id,
            correlation_id=document.correlation_id,
            source_path=path,
        )
    ) as extraction:
        yield Case(document=document, job=claim.job, extraction=extraction)


async def _stored_pages(
    uow_factory: UnitOfWorkFactory,
    document: Document,
) -> int:
    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        return await uow.pages.count(document.id, PIPELINE_VERSION)


async def _stored_job(
    uow_factory: UnitOfWorkFactory,
    document: Document,
) -> ProcessingJob:
    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        job = await uow.jobs.get(document.id, PIPELINE_VERSION)
    assert job is not None
    return job


async def test_text_layer_page_is_saved_without_confidence(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with open_case(path) as case:
        result = await process_page.execute(case.command())

    assert result.persisted
    assert result.status is PageStatus.EXTRACTED
    assert result.method is ExtractionMethod.TEXT_LAYER
    assert result.confidence is None


async def test_saved_text_is_normalized(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    # Перенос со строки на строку рвёт слово, и без склейки оно не найдётся
    # ни поиском, ни чанкованием.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", lines=HYPHENATED_LINES)

    async with open_case(path) as case:
        await process_page.execute(case.command())
        document = case.document

    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        pages = await uow.pages.load_pages(
            document.id,
            PIPELINE_VERSION,
            statuses=frozenset({PageStatus.EXTRACTED}),
        )
    assert "undertakes" in pages[0].text.content


async def test_page_moves_the_counters_of_its_job(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with open_case(path) as case:
        await process_page.execute(case.command())
        document = case.document

    job = await _stored_job(uow_factory, document)
    assert job.pages_text_layer == 1


async def test_repeated_page_is_not_written_twice(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with open_case(path) as case:
        first = await process_page.execute(case.command())
        second = await process_page.execute(case.command())
        document = case.document

    assert first.persisted
    assert not second.persisted
    assert await _stored_pages(uow_factory, document) == 1


async def test_repeated_page_is_not_counted_twice(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    # Счётчик прогона прибавляется в той же транзакции, что и страница, и
    # прибавляться на несохранённой странице ему не с чего.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with open_case(path) as case:
        await process_page.execute(case.command())
        await process_page.execute(case.command())
        document = case.document

    job = await _stored_job(uow_factory, document)
    assert job.pages_text_layer == 1


async def test_page_needing_recognition_fails_until_ocr_arrives(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    tmp_path: Path,
) -> None:
    # Распознавания ещё нет, и страница сохраняется отказом, а не пропадает.
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    async with open_case(path) as case:
        result = await process_page.execute(case.command())

    assert result.persisted
    assert result.status is PageStatus.FAILED
    assert result.method is ExtractionMethod.NONE
    assert result.failure_reason is PageFailureReason.OCR_FAILED


async def test_page_level_error_is_stored_as_a_failed_page(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    tmp_path: Path,
) -> None:
    # Одна нечитаемая страница не отменяет документ: она фиксируется отказом,
    # а обработка идёт дальше.
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    async with open_case(path) as case:
        broken = ProcessDocumentPageCommand(
            document_id=case.document.id,
            correlation_id=case.document.correlation_id,
            job_id=case.job.id,
            entry=PagePlanEntryDTO(
                number=PageNumber(1),
                method=ExtractionMethod.TEXT_LAYER,
                reasons=(),
            ),
            extraction=DocumentExtraction(
                plan=case.extraction.plan,
                pdf=_UnreadablePdf(),
                renderer=None,
            ),
        )
        result = await process_page.execute(broken)

    assert result.persisted
    assert result.status is PageStatus.FAILED
    assert result.failure_reason is PageFailureReason.RENDER_FAILED


async def test_failed_page_counts_as_failed_in_the_job(
    open_case: OpenCase,
    process_page: ProcessDocumentPage,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    async with open_case(path) as case:
        await process_page.execute(case.command())
        document = case.document

    job = await _stored_job(uow_factory, document)
    assert job.pages_failed == 1
    assert job.pages_text_layer == 0


@dataclass(frozen=True, slots=True)
class _UnreadablePdf:
    """Документ, страницы которого не читаются."""

    async def read_page_text(self, number: int) -> PdfPageTextDTO:
        raise PageRenderError("страницу не удалось прочитать", page_number=number)

    async def probe(self) -> TextLayerProbeDTO:
        raise NotImplementedError
