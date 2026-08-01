"""Ветка распознавания в обработке страницы.

Отказ распознавания — результат обработки, а не ошибка сообщения: повторная
доставка даст ровно то же самое, поэтому страница уходит в отказ, а документ
продолжает обрабатываться.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from document_worker.application.dto.commands import (
    ExtractDocumentTextCommand,
    ProcessDocumentPageCommand,
)
from document_worker.application.dto.ocr import (
    ConfidenceSource,
    OcrResult,
    RecognizedWordDTO,
)
from document_worker.application.errors import PageOcrTimeoutError
from document_worker.application.use_cases.extract_document_text import (
    ExtractDocumentText,
)
from document_worker.application.use_cases.process_document_page import (
    ProcessDocumentPage,
)
from document_worker.domain.normalization.normalizer import TextNormalizer
from document_worker.domain.policies.page_legibility import PageLegibilityPolicy
from document_worker.domain.policies.text_layer_quality import TextLayerQualityPolicy
from document_worker.domain.value_objects.enums import ExtractionMethod, PageStatus
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.infrastructure.ocr.preprocessor import OpenCvImagePreprocessor
from document_worker.infrastructure.ocr.rapidocr_engine import RapidOcrEngine
from document_worker.infrastructure.pdf.pdfplumber_text_reader import (
    PdfPlumberDocumentReader,
)
from document_worker.infrastructure.pdf.pikepdf_inspector import PikePdfInspector
from document_worker.infrastructure.pdf.pypdfium2_page_renderer import (
    PdfiumPageRenderer,
)
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.mappers.job import job_to_row
from tests.factories import make_document, make_job
from tests.fakes import pdf_builder
from tests.integration.application.conftest import PIPELINE_VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.extraction import DocumentExtraction
    from document_worker.application.dto.ocr import PreparedPage
    from document_worker.application.dto.results import ProcessDocumentPageResult
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.entities.document import Document
    from document_worker.infrastructure.cpu.executor import CpuPool
    from tests.fakes.system import FixedClock, SequentialIdGenerator

pytestmark = pytest.mark.integration

MAX_PAGES = 8
MAX_PIXELS = 8_000_000
BOX = BoundingBox(0.1, 0.1, 0.4, 0.2)


@dataclass
class ScriptedEngine:
    """Движок, отвечающий по заранее написанному сценарию."""

    outcomes: list[OcrResult | Exception]
    seen_dpi: list[int] = field(default_factory=list)

    async def recognize(
        self,
        page: PreparedPage,
        *,
        languages: Sequence[str],  # noqa: ARG002 — сценарий от языка не зависит
        timeout_s: float,  # noqa: ARG002 — таймаут разыгрывает сам сценарий
        options: Any = None,  # noqa: ARG002 — порт допускает опции
    ) -> OcrResult:
        self.seen_dpi.append(page.image.dpi)
        outcome = self.outcomes.pop(0) if self.outcomes else self.outcomes[-1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def recognized(*words: tuple[str, float], line_height: float = 40.0) -> OcrResult:
    """Результат распознавания с заданными словами и уверенностью."""
    return OcrResult(
        words=tuple(
            RecognizedWordDTO(
                text=text,
                confidence=confidence,
                bbox=BOX,
                line_index=0,
                word_index=index,
                confidence_source=ConfidenceSource.WORD,
            )
            for index, (text, confidence) in enumerate(words)
        ),
        line_count=1 if words else 0,
        median_line_height_px=line_height,
        engine_version="scripted",
        elapsed_ms=1,
    )


@dataclass(frozen=True)
class Wiring:
    """Всё, из чего собирается обработка страницы, кроме самого движка."""

    uow_factory: UnitOfWorkFactory
    pool: CpuPool
    clock: FixedClock
    ids: SequentialIdGenerator
    config: ProcessingConfig

    def with_engine(self, engine: Any) -> ProcessDocumentPage:
        """Обработка страницы на настоящей предобработке и заданном движке."""
        return ProcessDocumentPage(
            uow_factory=self.uow_factory,
            normalizer=TextNormalizer(),
            preprocessor=OpenCvImagePreprocessor(pool=self.pool),
            engine=engine,
            legibility=PageLegibilityPolicy(),
            ids=self.ids,
            clock=self.clock,
            config=self.config,
        )


@pytest.fixture
def wiring(
    uow_factory: UnitOfWorkFactory,
    cpu_pool: CpuPool,
    clock: FixedClock,
    ids: SequentialIdGenerator,
    config: ProcessingConfig,
) -> Wiring:
    """Собранная обвязка обработки страницы."""
    return Wiring(
        uow_factory=uow_factory, pool=cpu_pool, clock=clock, ids=ids, config=config
    )


@contextlib.asynccontextmanager
async def opened(
    path: Path,
    cpu_pool: CpuPool,
    document: Document,
) -> AsyncIterator[DocumentExtraction]:
    """План извлечения вместе с открытыми сессиями чтения и рендера."""
    extract = ExtractDocumentText(
        inspector=PikePdfInspector(pool=cpu_pool, max_pages=MAX_PAGES),
        reader=PdfPlumberDocumentReader(pool=cpu_pool),
        renderer=PdfiumPageRenderer(pool=cpu_pool, max_pixels=MAX_PIXELS),
        policy=TextLayerQualityPolicy(),
    )
    async with extract.execute(
        ExtractDocumentTextCommand(
            document_id=document.id,
            correlation_id=document.correlation_id,
            source_path=path,
        )
    ) as extraction:
        yield extraction


async def run_page(
    use_case: ProcessDocumentPage,
    extraction: DocumentExtraction,
    document: Document,
    job_id: Any,
) -> ProcessDocumentPageResult:
    """Обрабатывает первую страницу плана."""
    return await use_case.execute(
        ProcessDocumentPageCommand(
            document_id=document.id,
            correlation_id=document.correlation_id,
            job_id=job_id,
            entry=extraction.plan.pages[0],
            extraction=extraction,
        )
    )


async def stored_document(session: AsyncSession) -> tuple[Document, Any]:
    """Документ и открытый прогон в базе."""
    document = make_document()
    session.add(document_to_row(document))
    job = make_job(document)
    session.add(job_to_row(job))
    await session.commit()
    return document, job.id


async def test_scan_is_read_by_recognition_with_confidence(
    session: AsyncSession,
    wiring: Wiring,
    model_dir: Path,
    tmp_path: Path,
) -> None:
    document, job_id = await stored_document(session)
    use_case = wiring.with_engine(RapidOcrEngine(pool=wiring.pool, model_dir=model_dir))
    source = pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf")

    async with opened(source, wiring.pool, document) as extraction:
        assert extraction.plan.pages[0].method is ExtractionMethod.OCR
        result = await run_page(use_case, extraction, document, job_id)

    assert result.method is ExtractionMethod.OCR
    assert result.confidence is not None
    assert result.char_count > 0
    assert result.status is not PageStatus.FAILED


async def test_page_without_recognized_text_is_marked_illegible(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    # Основной сценарий раздела устава о неразборчивых фрагментах: движок не
    # выдал ничего, и это обязано быть представимо, а не падать.
    document, job_id = await stored_document(session)
    use_case = wiring.with_engine(ScriptedEngine([recognized()]))
    source = pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf")

    async with opened(source, wiring.pool, document) as extraction:
        result = await run_page(use_case, extraction, document, job_id)

    assert result.status is PageStatus.ILLEGIBLE
    assert result.method is ExtractionMethod.OCR


async def test_low_confidence_word_becomes_illegible_span_with_raw_text(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    # Устав запрещает подменять неразборчивое предположением: сохраняется ровно
    # то, что выдал движок.
    document, job_id = await stored_document(session)
    engine = ScriptedEngine(
        [recognized(("Договор", 0.95), ("поставки", 0.95), ("тваров", 0.20))]
    )
    use_case = wiring.with_engine(engine)
    source = pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf")

    async with opened(source, wiring.pool, document) as extraction:
        result = await run_page(use_case, extraction, document, job_id)

    async with wiring.uow_factory(statement_timeout_ms=2000, read_only=True) as uow:
        pages = await uow.pages.load_pages(
            document.id,
            PIPELINE_VERSION,
            statuses=frozenset({PageStatus.PARTIALLY_ILLEGIBLE, PageStatus.ILLEGIBLE}),
        )
    assert result.status is not PageStatus.EXTRACTED
    span = pages[0].illegible_spans[0]
    assert span.raw_text == span.span.slice_of(pages[0].text.content)
    assert "тваров" in pages[0].text.content


async def test_retries_at_higher_dpi_and_keeps_the_best_result(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    # Рост разрешения помогает только детектору: если строка и так высокая,
    # повышать нечего, а если низкая — вторая попытка обязана состояться.
    document, job_id = await stored_document(session)
    engine = ScriptedEngine(
        [
            recognized(("мутно", 0.30), line_height=12.0),
            recognized(("Договор", 0.95), ("поставки", 0.95), line_height=12.0),
        ]
    )
    use_case = wiring.with_engine(engine)
    source = pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf")

    async with opened(source, wiring.pool, document) as extraction:
        result = await run_page(use_case, extraction, document, job_id)

    assert engine.seen_dpi == [
        wiring.config.ocr.dpi_primary,
        wiring.config.ocr.dpi_retry,
    ]
    async with wiring.uow_factory(statement_timeout_ms=2000, read_only=True) as uow:
        pages = await uow.pages.load_pages(
            document.id, PIPELINE_VERSION, statuses=frozenset({result.status})
        )
    assert pages[0].text.content == "Договор поставки"


async def test_high_line_does_not_trigger_dpi_escalation(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    # Дело не в разрешении: шум, рукопись или печать выше не станут читаемее.
    document, job_id = await stored_document(session)
    engine = ScriptedEngine([recognized(("мутно", 0.30), line_height=90.0)])
    use_case = wiring.with_engine(engine)
    source = pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf")

    async with opened(source, wiring.pool, document) as extraction:
        await run_page(use_case, extraction, document, job_id)

    assert engine.seen_dpi == [wiring.config.ocr.dpi_primary]


async def test_timeout_falls_back_to_degraded_attempt(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    document, job_id = await stored_document(session)
    engine = ScriptedEngine(
        [
            PageOcrTimeoutError("не уложилась", page_number=1),
            recognized(("Договор", 0.95), ("поставки", 0.95)),
        ]
    )
    use_case = wiring.with_engine(engine)
    source = pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf")

    async with opened(source, wiring.pool, document) as extraction:
        result = await run_page(use_case, extraction, document, job_id)

    assert engine.seen_dpi == [
        wiring.config.ocr.dpi_primary,
        wiring.config.ocr.dpi_degraded,
    ]
    assert result.status is not PageStatus.FAILED


async def test_page_is_failed_after_timeouts_without_failing_the_document(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    # Повторная доставка даст ровно тот же результат: retry бессмыслен, а DLQ
    # вредна — документ обработан на девяносто пять процентов и полезен.
    document, job_id = await stored_document(session)
    engine = ScriptedEngine(
        [
            PageOcrTimeoutError("не уложилась", page_number=1),
            PageOcrTimeoutError("не уложилась", page_number=1),
        ]
    )
    use_case = wiring.with_engine(engine)
    source = pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf")

    async with opened(source, wiring.pool, document) as extraction:
        result = await run_page(use_case, extraction, document, job_id)

    assert result.status is PageStatus.FAILED
    assert result.method is ExtractionMethod.NONE
