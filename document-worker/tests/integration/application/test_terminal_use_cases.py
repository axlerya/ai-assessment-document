"""Терминальные транзакции: T4 и T4f.

Статус документа, строка прогона, исходящее событие и отметка сообщения
пишутся вместе. Ноль строк от guard-UPDATE — это дубль, а не ошибка: именно
эта ветка не даёт пометить корректно обработанный документ отказом.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from document_worker.application.dto.commands import (
    CompleteDocumentProcessingCommand,
    FailDocumentProcessingCommand,
    ProcessDocumentCommand,
)
from document_worker.application.dto.results import TerminalOutcome
from document_worker.application.errors import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    DomainInvariantViolationError,
)
from document_worker.application.services.message_claim import MessageClaimService
from document_worker.application.use_cases.complete_document_processing import (
    CompleteDocumentProcessing,
)
from document_worker.application.use_cases.fail_document_processing import (
    FailDocumentProcessing,
)
from document_worker.domain.events import (
    DocumentPartiallyProcessed,
    DocumentProcessed,
    DocumentProcessingFailed,
)
from document_worker.domain.policies.document_status import DocumentStatusPolicy
from document_worker.domain.value_objects.enums import (
    DocumentStatus,
    JobStatus,
    ProcessingStage,
)
from document_worker.domain.value_objects.identifiers import DocumentId, EventId, JobId
from document_worker.domain.value_objects.storage import Checksum, FileSize
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from tests.factories import (
    make_document,
    make_failed_page,
    make_text_layer_page,
)
from tests.integration.application.conftest import NOW, PIPELINE_VERSION

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.results import OutboxRecordDTO
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.entities.document import Document
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.entities.processing_job import ProcessingJob
    from document_worker.domain.value_objects.versioning import PipelineVersion
    from tests.fakes.system import FixedClock, SequentialIdGenerator

pytestmark = pytest.mark.integration

# Политика требует двухсот символов на документ, иначе он «без извлекаемого
# текста»; страницы по умолчанию короче.
LONG_TEXT = "договор поставки товаров и услуг между сторонами настоящего дела " * 4
LEASE_OWNER = "test-relay"
SOURCE_SIZE = FileSize(2048)
SOURCE_CHECKSUM = Checksum.sha256_of(b"source")


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
def complete(
    uow_factory: UnitOfWorkFactory,
    clock: FixedClock,
    config: ProcessingConfig,
) -> CompleteDocumentProcessing:
    return CompleteDocumentProcessing(
        uow_factory=uow_factory,
        status_policy=DocumentStatusPolicy(),
        clock=clock,
        config=config,
    )


@pytest.fixture
def fail(
    uow_factory: UnitOfWorkFactory,
    clock: FixedClock,
    config: ProcessingConfig,
) -> FailDocumentProcessing:
    return FailDocumentProcessing(uow_factory=uow_factory, clock=clock, config=config)


@dataclass(frozen=True, slots=True)
class Claimed:
    """Захваченный документ вместе с его прогоном и событием."""

    document: Document
    job: ProcessingJob
    event_id: EventId

    def completion(self, *, page_count: int) -> CompleteDocumentProcessingCommand:
        """Команда завершения обработки этого документа."""
        return CompleteDocumentProcessingCommand(
            document_id=self.document.id,
            correlation_id=self.document.correlation_id,
            event_id=self.event_id,
            job_id=self.job.id,
            page_count=page_count,
            chunks_total=0,
            source_size=SOURCE_SIZE,
            source_checksum=SOURCE_CHECKSUM,
        )

    def failure(
        self,
        *,
        code: str = "corrupted_document",
        pages_persisted: int = 0,
    ) -> FailDocumentProcessingCommand:
        """Команда фиксации отказа по этому документу."""
        return FailDocumentProcessingCommand(
            document_id=self.document.id,
            correlation_id=self.document.correlation_id,
            event_id=self.event_id,
            job_id=self.job.id,
            error_code=code,
            error_message="файл не читается",
            stage=ProcessingStage.TEXT_EXTRACTION,
            pages_persisted=pages_persisted,
        )


async def _claim(
    session: AsyncSession,
    claim_service: MessageClaimService,
) -> Claimed:
    document = make_document()
    session.add(document_to_row(document))
    await session.commit()
    event_id = EventId.generate()
    claim = await claim_service.claim(
        ProcessDocumentCommand(
            event_id=event_id,
            document_id=document.id,
            correlation_id=document.correlation_id,
            object_ref=document.source.ref,
            mime_type=document.source.mime_type,
            occurred_at=NOW,
        )
    )
    assert claim.job is not None
    return Claimed(document=document, job=claim.job, event_id=event_id)


async def _add_pages(
    uow_factory: UnitOfWorkFactory,
    *pages: DocumentPage,
) -> None:
    async with uow_factory(statement_timeout_ms=1000) as uow:
        for page in pages:
            await uow.pages.add(page)
        await uow.commit()


async def _add_two_read_pages(
    uow_factory: UnitOfWorkFactory,
    document: Document,
) -> None:
    await _add_pages(
        uow_factory,
        make_text_layer_page(document, number=1, content=LONG_TEXT),
        make_text_layer_page(document, number=2, content=LONG_TEXT),
    )


async def _reload(uow_factory: UnitOfWorkFactory, document: Document) -> Document:
    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        stored = await uow.documents.get(document.id)
    assert stored is not None
    return stored


async def _job_of(uow_factory: UnitOfWorkFactory, document: Document) -> ProcessingJob:
    async with uow_factory(statement_timeout_ms=1000, read_only=True) as uow:
        job = await uow.jobs.get(document.id, PIPELINE_VERSION)
    assert job is not None
    return job


def _deterministic_id(
    *,
    document_id: DocumentId,
    pipeline_version: PipelineVersion,
    event_type: str,
) -> uuid.UUID:
    return EventId.deterministic(
        document_id=document_id,
        pipeline_version=pipeline_version,
        event_type=event_type,
    ).value


async def _outbox(uow_factory: UnitOfWorkFactory) -> tuple[OutboxRecordDTO, ...]:
    async with uow_factory(statement_timeout_ms=1000) as uow:
        return await uow.outbox.fetch_pending(
            limit=10, now=NOW, lease_owner=LEASE_OWNER, lease_seconds=30
        )


async def test_all_pages_read_gives_a_processed_document(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_two_read_pages(uow_factory, document)

    result = await complete.execute(claimed.completion(page_count=2))

    assert result.terminal is TerminalOutcome.APPLIED
    assert result.status is DocumentStatus.PROCESSED
    assert result.event_type == DocumentProcessed.event_type


async def test_unread_page_gives_a_partially_processed_document(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_pages(
        uow_factory,
        make_text_layer_page(document, number=1, content=LONG_TEXT),
        make_text_layer_page(document, number=2, content=LONG_TEXT),
        make_failed_page(document, number=3),
    )

    result = await complete.execute(claimed.completion(page_count=3))

    assert result.status is DocumentStatus.PARTIALLY_PROCESSED
    assert result.event_type == DocumentPartiallyProcessed.event_type


async def test_document_without_usable_pages_ends_failed(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Скан без распознавания даёт ровно этот случай, и завершать его успехом
    # нельзя: пригодного текста в документе нет.
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_pages(
        uow_factory,
        make_failed_page(document, number=1),
        make_failed_page(document, number=2),
    )

    result = await complete.execute(claimed.completion(page_count=2))

    assert result.status is DocumentStatus.FAILED
    assert result.event_type == DocumentProcessingFailed.event_type


async def test_terminal_transaction_writes_all_four_records(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_two_read_pages(uow_factory, document)

    await complete.execute(claimed.completion(page_count=2))

    stored = await _reload(uow_factory, document)
    job = await _job_of(uow_factory, document)
    assert stored.status is DocumentStatus.PROCESSED
    assert stored.page_count == 2
    assert job.status is JobStatus.SUCCEEDED
    assert job.pages_total == 2
    assert len(await _outbox(uow_factory)) == 1


async def test_outbox_event_id_is_deterministic(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_two_read_pages(uow_factory, document)

    await complete.execute(claimed.completion(page_count=2))

    records = await _outbox(uow_factory)
    assert records[0].event_id == _deterministic_id(
        document_id=document.id,
        pipeline_version=PIPELINE_VERSION,
        event_type=DocumentProcessed.event_type,
    )


async def test_second_completion_is_a_duplicate_and_changes_nothing(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Ветка, из-за которой корректно обработанный документ помечался отказом.
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_two_read_pages(uow_factory, document)
    await complete.execute(claimed.completion(page_count=2))

    again = await complete.execute(claimed.completion(page_count=2))

    assert again.terminal is TerminalOutcome.DUPLICATE
    assert again.status is DocumentStatus.PROCESSED
    assert len(await _outbox(uow_factory)) == 1


async def test_completion_closes_the_message(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Незакрытое сообщение держит лиз и не даёт повторной доставке ничего
    # понять про документ.
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_two_read_pages(uow_factory, document)

    await complete.execute(claimed.completion(page_count=2))

    second = await claim_service.claim(
        ProcessDocumentCommand(
            event_id=claimed.event_id,
            document_id=document.id,
            correlation_id=document.correlation_id,
            object_ref=document.source.ref,
            mime_type=document.source.mime_type,
            occurred_at=NOW,
        )
    )
    assert not second.should_process


async def test_completion_of_a_missing_document_raises_not_found(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
) -> None:
    # Строка существовала на входе и исчезла в процессе: её удалили извне.
    claimed = await _claim(session, claim_service)
    vanished = CompleteDocumentProcessingCommand(
        document_id=DocumentId(uuid.uuid4()),
        correlation_id=claimed.document.correlation_id,
        event_id=claimed.event_id,
        job_id=claimed.job.id,
        page_count=1,
        chunks_total=0,
        source_size=SOURCE_SIZE,
        source_checksum=SOURCE_CHECKSUM,
    )

    with pytest.raises(DocumentNotFoundError):
        await complete.execute(vanished)


async def test_failure_records_its_code_and_stage(
    session: AsyncSession,
    claim_service: MessageClaimService,
    fail: FailDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)

    result = await fail.execute(claimed.failure())

    assert result.terminal is TerminalOutcome.APPLIED
    stored = await _reload(uow_factory, claimed.document)
    assert stored.status is DocumentStatus.FAILED
    assert stored.failure_code == "corrupted_document"
    assert stored.failure_stage is ProcessingStage.TEXT_EXTRACTION


async def test_failure_enqueues_one_event(
    session: AsyncSession,
    claim_service: MessageClaimService,
    fail: FailDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)

    await fail.execute(claimed.failure())

    records = await _outbox(uow_factory)
    assert len(records) == 1
    assert records[0].event_id == _deterministic_id(
        document_id=claimed.document.id,
        pipeline_version=PIPELINE_VERSION,
        event_type=DocumentProcessingFailed.event_type,
    )


async def test_failure_fails_the_job_too(
    session: AsyncSession,
    claim_service: MessageClaimService,
    fail: FailDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)

    await fail.execute(claimed.failure())

    job = await _job_of(uow_factory, claimed.document)
    assert job.status is JobStatus.FAILED
    assert job.error_code == "corrupted_document"


async def test_repeated_failure_is_a_duplicate(
    session: AsyncSession,
    claim_service: MessageClaimService,
    fail: FailDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)
    await fail.execute(claimed.failure())

    again = await fail.execute(claimed.failure())

    assert again.terminal is TerminalOutcome.DUPLICATE
    assert len(await _outbox(uow_factory)) == 1


async def test_failure_does_not_touch_an_already_completed_document(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    fail: FailDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    claimed = await _claim(session, claim_service)
    document = claimed.document
    await _add_two_read_pages(uow_factory, document)
    await complete.execute(claimed.completion(page_count=2))

    result = await fail.execute(claimed.failure())

    assert result.terminal is TerminalOutcome.DUPLICATE
    stored = await _reload(uow_factory, document)
    assert stored.status is DocumentStatus.PROCESSED


async def test_failure_closes_the_message(
    session: AsyncSession,
    claim_service: MessageClaimService,
    fail: FailDocumentProcessing,
) -> None:
    claimed = await _claim(session, claim_service)
    document = claimed.document

    await fail.execute(claimed.failure())

    second = await claim_service.claim(
        ProcessDocumentCommand(
            event_id=claimed.event_id,
            document_id=document.id,
            correlation_id=document.correlation_id,
            object_ref=document.source.ref,
            mime_type=document.source.mime_type,
            occurred_at=NOW,
        )
    )
    assert not second.should_process


async def test_completion_of_an_empty_document_is_a_permanent_error(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
) -> None:
    # Инспектор такой документ не пропускает, но нарушенный инвариант домена
    # обязан долетать прикладной неисправимой ошибкой, а не доменной.
    claimed = await _claim(session, claim_service)

    with pytest.raises(CorruptedDocumentError):
        await complete.execute(claimed.completion(page_count=0))


async def test_completion_with_more_pages_than_declared_is_a_permanent_error(
    session: AsyncSession,
    claim_service: MessageClaimService,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Сохранённых страниц больше, чем насчитал инспектор: числа разошлись,
    # и записывать такой прогон нельзя.
    claimed = await _claim(session, claim_service)
    await _add_two_read_pages(uow_factory, claimed.document)

    with pytest.raises(DomainInvariantViolationError):
        await complete.execute(claimed.completion(page_count=1))


async def test_completion_without_a_job_is_an_invariant_violation(
    session: AsyncSession,
    complete: CompleteDocumentProcessing,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Прогон открывается захватом, и его отсутствие означает сломанное
    # состояние, а не штатную ветку.
    document = make_document(status=DocumentStatus.PROCESSING)
    session.add(document_to_row(document))
    await session.commit()
    command = CompleteDocumentProcessingCommand(
        document_id=document.id,
        correlation_id=document.correlation_id,
        event_id=EventId.generate(),
        job_id=JobId(uuid.uuid4()),
        page_count=1,
        chunks_total=0,
        source_size=SOURCE_SIZE,
        source_checksum=SOURCE_CHECKSUM,
    )
    del uow_factory

    with pytest.raises(DomainInvariantViolationError):
        await complete.execute(command)


async def test_failure_of_a_missing_document_raises_not_found(
    session: AsyncSession,
    claim_service: MessageClaimService,
    fail: FailDocumentProcessing,
) -> None:
    claimed = await _claim(session, claim_service)
    vanished = FailDocumentProcessingCommand(
        document_id=DocumentId(uuid.uuid4()),
        correlation_id=claimed.document.correlation_id,
        event_id=claimed.event_id,
        job_id=claimed.job.id,
        error_code="corrupted_document",
        error_message="файл не читается",
        stage=ProcessingStage.TEXT_EXTRACTION,
    )

    with pytest.raises(DocumentNotFoundError):
        await fail.execute(vanished)


async def test_failure_without_a_job_is_an_invariant_violation(
    session: AsyncSession,
    fail: FailDocumentProcessing,
) -> None:
    document = make_document(status=DocumentStatus.PROCESSING)
    session.add(document_to_row(document))
    await session.commit()
    command = FailDocumentProcessingCommand(
        document_id=document.id,
        correlation_id=document.correlation_id,
        event_id=EventId.generate(),
        job_id=JobId(uuid.uuid4()),
        error_code="corrupted_document",
        error_message="файл не читается",
        stage=ProcessingStage.TEXT_EXTRACTION,
    )

    with pytest.raises(DomainInvariantViolationError):
        await fail.execute(command)
