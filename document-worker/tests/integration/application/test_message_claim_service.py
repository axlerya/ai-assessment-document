"""Захват сообщения: четыре исхода и то, что происходит с документом."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from document_worker.application.dto.commands import ProcessDocumentCommand
from document_worker.application.dto.results import ClaimOutcome, MessageOutcome
from document_worker.application.errors import (
    ConcurrentProcessingError,
    DocumentNotFoundError,
    InvalidCommandError,
)
from document_worker.application.services.message_claim import MessageClaimService
from document_worker.domain.value_objects.enums import DocumentStatus, JobStatus
from document_worker.domain.value_objects.identifiers import EventId
from document_worker.domain.value_objects.storage import MimeType, ObjectRef
from document_worker.infrastructure.persistence.mappers.document import document_to_row
from document_worker.infrastructure.persistence.repositories.documents import (
    SqlAlchemyDocumentRepository,
)
from tests.factories import make_document, make_text_layer_page
from tests.integration.application.conftest import NOW, PIPELINE_VERSION

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.entities.document import Document
    from tests.fakes.system import FixedClock, SequentialIdGenerator

pytestmark = pytest.mark.integration


@pytest.fixture
def service(
    uow_factory: UnitOfWorkFactory,
    clock: FixedClock,
    ids: SequentialIdGenerator,
    config: ProcessingConfig,
) -> MessageClaimService:
    return MessageClaimService(
        uow_factory=uow_factory, clock=clock, ids=ids, config=config
    )


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


async def _persist(session: AsyncSession, document: Document) -> Document:
    session.add(document_to_row(document))
    await session.commit()
    return document


async def test_claim_returns_proceed_for_new_event(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    document = await _persist(session, make_document())

    result = await service.claim(_command(document))

    assert result.outcome is ClaimOutcome.PROCEED
    assert result.should_process
    assert result.persisted_page_numbers == frozenset()


async def test_claim_moves_document_into_processing(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    document = await _persist(session, make_document())

    await service.claim(_command(document))

    stored = await _reload(session, document)
    assert stored.status is DocumentStatus.PROCESSING
    assert stored.pipeline_version == PIPELINE_VERSION
    assert stored.processing_started_at == NOW


async def test_claim_opens_a_running_job(
    service: MessageClaimService,
    session: AsyncSession,
    uow_factory: UnitOfWorkFactory,
) -> None:
    document = await _persist(session, make_document())

    await service.claim(_command(document))

    async with uow_factory(statement_timeout_ms=1000) as uow:
        job = await uow.jobs.get(document.id, PIPELINE_VERSION)
    assert job is not None
    assert job.status is JobStatus.RUNNING


async def test_claim_hands_the_open_job_to_the_caller(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    # Страницы и терминальная транзакция пишутся в этот прогон, и лишний
    # запрос за ним стоил бы обращения к базе на каждой странице.
    document = await _persist(session, make_document())

    result = await service.claim(_command(document))

    assert result.job is not None
    assert result.job.status is JobStatus.RUNNING


async def test_resumed_delivery_continues_the_same_job(
    service: MessageClaimService,
    session: AsyncSession,
    clock: FixedClock,
    config: ProcessingConfig,
) -> None:
    document = await _persist(session, make_document())
    command = _command(document)
    first = await service.claim(command)
    clock.advance(seconds=config.claim_lease_s + 1)

    second = await service.claim(command)

    assert first.job is not None
    assert second.job is not None
    assert second.job.id == first.job.id


async def test_skipped_message_opens_no_job(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    document = await _persist(
        session, make_document(status=DocumentStatus.PROCESSED, page_count=1)
    )

    result = await service.claim(_command(document))

    assert result.outcome is ClaimOutcome.SKIP
    assert result.job is None


async def test_claim_raises_transient_when_document_row_is_absent(
    service: MessageClaimService,
) -> None:
    # Строку создаёт сервис приёма файлов: сообщение обогнало его коммит.
    document = make_document()

    with pytest.raises(DocumentNotFoundError):
        await service.claim(_command(document))


async def test_claim_rejects_command_with_object_key_mismatch(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    # Обработать чужой объект по чужой команде значит записать в документ
    # содержимое другого файла.
    document = await _persist(session, make_document())
    command = _command(document)
    foreign = ProcessDocumentCommand(
        event_id=command.event_id,
        document_id=command.document_id,
        correlation_id=command.correlation_id,
        object_ref=ObjectRef(bucket="documents", key="someone/else.pdf"),
        mime_type=command.mime_type,
        occurred_at=command.occurred_at,
    )

    with pytest.raises(InvalidCommandError):
        await service.claim(foreign)


async def test_claim_rejects_command_with_mime_type_mismatch(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    # Команда описывает не тот файл, что записан в строке документа: обработать
    # по ней значит записать в документ содержимое другого.
    document = await _persist(session, make_document())
    command = replace(_command(document), mime_type=MimeType("image/png"))

    with pytest.raises(InvalidCommandError):
        await service.claim(command)


async def test_claim_returns_reject_concurrent_for_live_lease(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    document = await _persist(session, make_document())
    command = _command(document)
    await service.claim(command)

    with pytest.raises(ConcurrentProcessingError):
        await service.claim(command)


async def test_claim_returns_resume_for_expired_lease(
    service: MessageClaimService,
    session: AsyncSession,
    clock: FixedClock,
    config: ProcessingConfig,
) -> None:
    document = await _persist(session, make_document())
    command = _command(document)
    await service.claim(command)
    clock.advance(seconds=config.claim_lease_s + 1)

    result = await service.claim(command)

    assert result.outcome is ClaimOutcome.RESUME
    assert result.should_process
    assert result.attempts == 2


async def test_claim_loads_persisted_page_numbers_for_resume(
    service: MessageClaimService,
    session: AsyncSession,
    clock: FixedClock,
    config: ProcessingConfig,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Возобновление обрабатывает только недостающие страницы.
    document = await _persist(session, make_document())
    command = _command(document)
    await service.claim(command)
    await _persist_page(uow_factory, document, number=1)
    clock.advance(seconds=config.claim_lease_s + 1)

    result = await service.claim(command)

    assert result.persisted_page_numbers == frozenset({1})


async def test_claim_returns_skip_for_completed_message(
    service: MessageClaimService,
    session: AsyncSession,
    uow_factory: UnitOfWorkFactory,
) -> None:
    document = await _persist(session, make_document())
    command = _command(document)
    await service.claim(command)
    async with uow_factory(statement_timeout_ms=1000) as uow:
        await uow.messages.mark_completed(
            command.event_id,
            outcome=MessageOutcome.PROCESSED,
            completed_at=NOW + timedelta(minutes=1),
        )
        await uow.commit()

    result = await service.claim(command)

    assert result.outcome is ClaimOutcome.SKIP
    assert not result.should_process


async def test_claim_returns_skip_for_already_processed_document(
    service: MessageClaimService,
    session: AsyncSession,
) -> None:
    # Другой event_id, документ уже терминален той же версией пайплайна:
    # работа выполнена, повторять нечего.
    document = await _persist(
        session,
        make_document(status=DocumentStatus.PROCESSED),
    )

    result = await service.claim(_command(document))

    assert result.outcome is ClaimOutcome.SKIP


async def test_skip_for_already_processed_document_leaves_no_claim(
    service: MessageClaimService,
    session: AsyncSession,
    uow_factory: UnitOfWorkFactory,
) -> None:
    # Иначе осталась бы запись in_progress, которую некому завершить.
    document = await _persist(
        session,
        make_document(status=DocumentStatus.PROCESSED),
    )

    await service.claim(_command(document))

    async with uow_factory(statement_timeout_ms=1000) as uow:
        outcome = await uow.messages.try_claim(
            service._claim_dto(_command(document), NOW)
        )
    assert outcome.outcome is ClaimOutcome.PROCEED


async def test_release_lets_the_next_delivery_resume_immediately(
    service: MessageClaimService,
    session: AsyncSession,
    clock: FixedClock,
) -> None:
    document = await _persist(session, make_document())
    command = _command(document)
    await service.claim(command)

    clock.advance(seconds=1)
    await service.release(command)
    result = await service.claim(command)

    assert result.outcome is ClaimOutcome.RESUME


async def _reload(session: AsyncSession, document: Document) -> Document:
    session.expire_all()
    stored = await SqlAlchemyDocumentRepository(session).get(document.id)
    assert stored is not None
    return stored


async def _persist_page(
    uow_factory: UnitOfWorkFactory,
    document: Document,
    *,
    number: int,
) -> None:
    async with uow_factory(statement_timeout_ms=1000) as uow:
        await uow.pages.add(make_text_layer_page(document, number=number))
        await uow.commit()
