"""Оркестратор обработки документа: сквозной сценарий текстового PDF.

Транзакций здесь нет и быть не может — их держат исполнители, и оркестратору
физически нечем открыть транзакцию поверх цикла по страницам.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import pytest

from document_worker.application.dto.commands import ProcessDocumentCommand
from document_worker.application.errors import (
    ConcurrentProcessingError,
    ProcessingDeadlineExceededError,
    StorageUnavailableError,
    UnsupportedMediaTypeError,
)
from document_worker.application.services.message_claim import MessageClaimService
from document_worker.application.services.page_runner import PageSequenceRunner
from document_worker.application.services.source_loader import SourceDocumentLoader
from document_worker.application.use_cases.complete_document_processing import (
    CompleteDocumentProcessing,
)
from document_worker.application.use_cases.create_document_chunks import (
    CreateDocumentChunks,
)
from document_worker.application.use_cases.extract_document_text import (
    ExtractDocumentText,
)
from document_worker.application.use_cases.fail_document_processing import (
    FailDocumentProcessing,
)
from document_worker.application.use_cases.process_document import (
    RETRIES_EXHAUSTED,
    ProcessDocument,
)
from document_worker.application.use_cases.process_document_page import (
    ProcessDocumentPage,
)
from document_worker.domain.normalization.normalizer import TextNormalizer
from document_worker.domain.policies.document_status import DocumentStatusPolicy
from document_worker.domain.policies.page_legibility import PageLegibilityPolicy
from document_worker.domain.policies.text_layer_quality import TextLayerQualityPolicy
from document_worker.domain.value_objects.enums import (
    DocumentStatus,
    ExtractionMethod,
    PageStatus,
)
from document_worker.domain.value_objects.identifiers import EventId
from document_worker.infrastructure.chunking.runner import CpuPoolChunkingRunner
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
from document_worker.infrastructure.storage.temp_workspace import (
    TempDirWorkspaceFactory,
)
from tests.factories import make_document, make_text_layer_page
from tests.fakes import pdf_builder
from tests.fakes.storage import InMemoryObjectStorage
from tests.fakes.watch import (
    TransactionWatch,
    WatchedInspector,
    WatchedReader,
    WatchedStorage,
)
from tests.integration.application.conftest import NOW, PIPELINE_VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.entities.document import Document
    from document_worker.infrastructure.cpu.executor import CpuPool
    from tests.fakes.system import FixedClock, SequentialIdGenerator

pytestmark = pytest.mark.integration

MAX_PAGES = 8
MAX_PIXELS = 4_000_000
# Политика считает документ короче двухсот символов «без извлекаемого текста»,
# а на страницу построителя приходится сто с небольшим.
PAGES = 3
LONG_TEXT = "договор поставки товаров и услуг между сторонами настоящего дела " * 4


@dataclass(frozen=True, slots=True)
class Wiring:
    """Собранный оркестратор вместе с тем, что нужно тесту для проверок."""

    process: ProcessDocument
    storage: InMemoryObjectStorage
    watch: TransactionWatch


@pytest.fixture
def wiring(  # noqa: PLR0913, PLR0917 — оркестратор собирается из всех этих частей
    uow_factory: UnitOfWorkFactory,
    cpu_pool: CpuPool,
    clock: FixedClock,
    ids: SequentialIdGenerator,
    config: ProcessingConfig,
    model_dir: Path,
    tmp_path: Path,
) -> Wiring:
    """Оркестратор на настоящих репозиториях, PDF-адаптерах и фейке хранилища."""
    watch = TransactionWatch()
    watched_factory = watch.wrap(uow_factory)
    storage = InMemoryObjectStorage()
    process = ProcessDocument(
        claim_service=MessageClaimService(
            uow_factory=watched_factory, clock=clock, ids=ids, config=config
        ),
        source_loader=SourceDocumentLoader(
            storage=WatchedStorage(inner=storage, watch=watch), config=config
        ),
        extract_text=ExtractDocumentText(
            inspector=WatchedInspector(
                inner=PikePdfInspector(pool=cpu_pool, max_pages=MAX_PAGES),
                watch=watch,
            ),
            reader=WatchedReader(
                inner=PdfPlumberDocumentReader(pool=cpu_pool), watch=watch
            ),
            renderer=PdfiumPageRenderer(pool=cpu_pool, max_pixels=MAX_PIXELS),
            policy=TextLayerQualityPolicy(),
        ),
        page_runner=PageSequenceRunner(
            process_page=ProcessDocumentPage(
                uow_factory=watched_factory,
                normalizer=TextNormalizer(),
                preprocessor=OpenCvImagePreprocessor(pool=cpu_pool),
                engine=RapidOcrEngine(pool=cpu_pool, model_dir=model_dir),
                legibility=PageLegibilityPolicy(),
                ids=ids,
                clock=clock,
                config=config,
            )
        ),
        create_chunks=CreateDocumentChunks(
            uow_factory=watched_factory,
            chunker=CpuPoolChunkingRunner(pool=cpu_pool),
            ids=ids,
            clock=clock,
            config=config,
        ),
        complete=CompleteDocumentProcessing(
            uow_factory=watched_factory,
            status_policy=DocumentStatusPolicy(),
            clock=clock,
            config=config,
        ),
        fail=FailDocumentProcessing(
            uow_factory=watched_factory, clock=clock, config=config
        ),
        workspaces=TempDirWorkspaceFactory(base_dir=tmp_path),
        config=config,
    )
    return Wiring(process=process, storage=storage, watch=watch)


async def _document_with_source(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
    *,
    pages: int = PAGES,
    payload: bytes | None = None,
) -> Document:
    document = make_document()
    session.add(document_to_row(document))
    await session.commit()
    source = (
        payload
        or pdf_builder.make_text_pdf(tmp_path / "source.pdf", pages=pages).read_bytes()
    )
    wiring.storage.put(document.source.ref, source)
    return document


def _command(
    document: Document, *, event_id: EventId | None = None
) -> ProcessDocumentCommand:
    return ProcessDocumentCommand(
        event_id=event_id or EventId.generate(),
        document_id=document.id,
        correlation_id=document.correlation_id,
        object_ref=document.source.ref,
        mime_type=document.source.mime_type,
        occurred_at=NOW,
    )


async def _pages_of(uow_factory: UnitOfWorkFactory, document: Document) -> int:
    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        return await uow.pages.count(document.id, PIPELINE_VERSION)


async def _outbox_size(uow_factory: UnitOfWorkFactory) -> int:
    async with uow_factory(statement_timeout_ms=1000) as uow:
        records = await uow.outbox.fetch_pending(
            limit=10, now=NOW, lease_owner="test", lease_seconds=30
        )
    return len(records)


async def _stored(uow_factory: UnitOfWorkFactory, document: Document) -> Document:
    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        stored = await uow.documents.get(document.id)
    assert stored is not None
    return stored


def test_orchestrator_has_no_unit_of_work_dependency() -> None:
    # Транзакция поверх цикла по страницам держала бы соединение все минуты
    # обработки; отсутствие фабрики в конструкторе делает это невозможным.
    declared = {str(field.type) for field in fields(ProcessDocument)}

    assert not [name for name in declared if "UnitOfWork" in name]


async def test_text_document_is_processed_end_to_end(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)

    result = await wiring.process.execute(_command(document))

    assert result.status is DocumentStatus.PROCESSED
    assert result.pages_total == PAGES
    assert await _pages_of(uow_factory, document) == PAGES
    assert await _outbox_size(uow_factory) == 1


async def test_no_pdf_or_storage_call_happens_inside_a_transaction(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)

    await wiring.process.execute(_command(document))

    assert wiring.watch.calls
    assert wiring.watch.inside == []


async def test_second_delivery_adds_neither_page_nor_event(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)
    command = _command(document)
    await wiring.process.execute(command)

    again = await wiring.process.execute(command)

    assert again.duplicate
    assert await _pages_of(uow_factory, document) == PAGES
    assert await _outbox_size(uow_factory) == 1


async def test_duplicate_message_never_touches_the_storage(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)
    command = _command(document)
    await wiring.process.execute(command)
    wiring.watch.calls.clear()

    await wiring.process.execute(command)

    assert wiring.watch.calls == []


async def test_resumed_delivery_processes_only_missing_pages(  # noqa: PLR0913, PLR0917 — возобновление проверяется на всей сборке
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    clock: FixedClock,
    config: ProcessingConfig,
    tmp_path: Path,
) -> None:
    # Первый воркер успел сохранить страницу и пропал, не завершив документ.
    # Второй обязан дочитать остаток, а не начать сначала.
    document = await _document_with_source(session, wiring, tmp_path)
    command = _command(document)
    await wiring.process.claim_service.claim(command)
    async with uow_factory(statement_timeout_ms=1000) as uow:
        await uow.pages.add(make_text_layer_page(document, number=1, content=LONG_TEXT))
        await uow.commit()
    clock.advance(seconds=config.claim_lease_s + 1)
    wiring.watch.calls.clear()

    result = await wiring.process.execute(command)

    assert result.status is DocumentStatus.PROCESSED
    assert await _pages_of(uow_factory, document) == PAGES
    assert result.pages_processed == PAGES - 1


async def test_live_lease_of_another_worker_is_a_transient_error(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    # Брокер передоставил сообщение, пока первый воркер ещё работает: его лиз
    # жив, и попытка расходуется намеренно — иначе зависший воркер гонял бы
    # сообщение по первой ступени retry без предела.
    document = await _document_with_source(session, wiring, tmp_path)
    command = _command(document)
    await wiring.process.claim_service.claim(command)

    with pytest.raises(ConcurrentProcessingError):
        await wiring.process.execute(command)


async def test_permanent_error_fails_the_document_without_reraising(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    # Результат зафиксирован, и повторять нечего: presentation подтверждает
    # сообщение и кладёт копию в DLQ для разбора.
    document = await _document_with_source(
        session,
        wiring,
        tmp_path,
        payload=pdf_builder.make_non_pdf_file(tmp_path / "not.pdf").read_bytes(),
    )

    result = await wiring.process.execute(_command(document))

    assert result.status is DocumentStatus.FAILED
    stored = await _stored(uow_factory, document)
    assert stored.status is DocumentStatus.FAILED
    assert stored.failure_code == UnsupportedMediaTypeError.code


async def test_permanent_error_enqueues_a_failure_event(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(
        session,
        wiring,
        tmp_path,
        payload=pdf_builder.make_non_pdf_file(tmp_path / "not.pdf").read_bytes(),
    )

    await wiring.process.execute(_command(document))

    assert await _outbox_size(uow_factory) == 1


async def test_transient_error_is_reraised_and_leaves_the_document_processing(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    # Временная недоступность хранилища не повод хоронить документ навсегда.
    document = await _document_with_source(session, wiring, tmp_path)
    wiring.storage.fail_next(StorageUnavailableError("хранилище недоступно"))

    with pytest.raises(StorageUnavailableError):
        await wiring.process.execute(_command(document))

    stored = await _stored(uow_factory, document)
    assert stored.status is DocumentStatus.PROCESSING


async def test_transient_error_releases_the_claim_for_the_next_delivery(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)
    wiring.storage.fail_next(StorageUnavailableError("хранилище недоступно"))
    command = _command(document)
    with pytest.raises(StorageUnavailableError):
        await wiring.process.execute(command)

    result = await wiring.process.execute(command)

    assert result.status is DocumentStatus.PROCESSED


async def test_temp_workspace_is_removed_after_processing(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)

    await wiring.process.execute(_command(document))

    assert [entry.name for entry in tmp_path.iterdir() if entry.is_dir()] == []


async def test_temp_workspace_is_removed_after_a_failure(
    session: AsyncSession,
    wiring: Wiring,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)
    wiring.storage.fail_next(StorageUnavailableError("хранилище недоступно"))

    with pytest.raises(StorageUnavailableError):
        await wiring.process.execute(_command(document))

    assert [entry.name for entry in tmp_path.iterdir() if entry.is_dir()] == []


async def test_scanned_document_goes_through_recognition(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    # Скан читается распознаванием и доходит до чанков: отказ здесь означал бы,
    # что сканы для сервиса по-прежнему не существуют.
    document = await _document_with_source(
        session,
        wiring,
        tmp_path,
        payload=pdf_builder.make_ocr_scan_pdf(
            tmp_path / "scan.pdf", pages=PAGES
        ).read_bytes(),
    )

    result = await wiring.process.execute(_command(document))

    assert result.status is not DocumentStatus.FAILED
    assert result.chunks_total > 0
    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        pages = await uow.pages.load_pages(
            document.id,
            PIPELINE_VERSION,
            statuses=frozenset({PageStatus.EXTRACTED, PageStatus.PARTIALLY_ILLEGIBLE}),
        )
    assert {page.method for page in pages} == {ExtractionMethod.OCR}
    assert pages[0].render_dpi is not None


async def test_processing_beyond_the_deadline_is_a_transient_error(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    config: ProcessingConfig,
    tmp_path: Path,
) -> None:
    # Общий бюджет документа кончился: сохранённые страницы остаются, следующая
    # доставка продолжит с них, а в failed документ не уходит.
    document = await _document_with_source(session, wiring, tmp_path)
    impatient = replace(wiring.process, config=replace(config, document_timeout_s=1e-6))

    with pytest.raises(ProcessingDeadlineExceededError):
        await impatient.execute(_command(document))

    stored = await _stored(uow_factory, document)
    assert stored.status is DocumentStatus.PROCESSING


async def test_transient_error_on_the_last_attempt_fails_the_document(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    # Лестница повторов исчерпана: держать документ в обработке дальше нечем,
    # и отказ обязан быть зафиксирован до того, как сообщение уйдёт в DLQ.
    document = await _document_with_source(session, wiring, tmp_path)
    wiring.storage.fail_next(StorageUnavailableError("хранилище недоступно"))

    result = await wiring.process.execute(_last_attempt(_command(document)))

    assert result.status is DocumentStatus.FAILED
    stored = await _stored(uow_factory, document)
    assert stored.status is DocumentStatus.FAILED
    assert stored.failure_code == RETRIES_EXHAUSTED


async def test_transient_error_before_the_last_attempt_is_still_retried(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    document = await _document_with_source(session, wiring, tmp_path)
    wiring.storage.fail_next(StorageUnavailableError("хранилище недоступно"))
    command = replace(_command(document), attempt=4, max_attempts=5)

    with pytest.raises(StorageUnavailableError):
        await wiring.process.execute(command)

    stored = await _stored(uow_factory, document)
    assert stored.status is DocumentStatus.PROCESSING


async def test_live_lease_on_the_last_attempt_does_not_bury_the_document(
    session: AsyncSession,
    wiring: Wiring,
    uow_factory: UnitOfWorkFactory,
    tmp_path: Path,
) -> None:
    # Лиз держит другой воркер, и он, возможно, работает. Пометить документ
    # отказом значит выбросить его результат: сообщение уходит в DLQ, а
    # документ остаётся в обработке за живым владельцем.
    document = await _document_with_source(session, wiring, tmp_path)
    command = _command(document)
    await wiring.process.claim_service.claim(command)

    with pytest.raises(ConcurrentProcessingError):
        await wiring.process.execute(_last_attempt(command))

    stored = await _stored(uow_factory, document)
    assert stored.status is DocumentStatus.PROCESSING


def _last_attempt(command: ProcessDocumentCommand) -> ProcessDocumentCommand:
    return replace(command, attempt=5, max_attempts=5)
