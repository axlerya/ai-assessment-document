"""Барьер идемпотентности доставки: четыре исхода claim и снятие лиза."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from document_worker.application.dto.results import (
    ClaimOutcome,
    MessageClaimDTO,
    MessageOutcome,
)
from document_worker.domain.value_objects.identifiers import EventId
from document_worker.infrastructure.persistence.repositories.processed_messages import (
    SqlAlchemyProcessedMessageRepository,
)
from tests.factories import (
    NOW,
    PIPELINE_VERSION,
    make_document,
    new_correlation_id,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.integration

LEASE = timedelta(minutes=15)
MESSAGE_TYPE = "document.process.requested"


def _claim(
    *,
    event_id: EventId | None = None,
    owner: str = "worker-1",
    at: object = NOW,
    lease_until: object = None,
) -> MessageClaimDTO:
    document = make_document()
    return MessageClaimDTO(
        event_id=event_id or EventId.generate(),
        document_id=document.id,
        correlation_id=new_correlation_id(),
        pipeline_version=PIPELINE_VERSION,
        message_type=MESSAGE_TYPE,
        lease_owner=owner,
        lease_expires_at=lease_until or (NOW + LEASE),  # type: ignore[operator]
        claimed_at=at,  # type: ignore[arg-type]
    )


async def test_try_claim_returns_proceed_for_new_event(session: AsyncSession) -> None:
    repository = SqlAlchemyProcessedMessageRepository(session)

    outcome = await repository.try_claim(_claim())

    assert outcome.outcome is ClaimOutcome.PROCEED
    assert outcome.attempts == 1


async def test_try_claim_returns_reject_concurrent_for_live_lease(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim()
    await repository.try_claim(claim)

    outcome = await repository.try_claim(claim)

    assert outcome.outcome is ClaimOutcome.REJECT_CONCURRENT


async def test_try_claim_returns_resume_for_expired_lease(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim(lease_until=NOW - timedelta(seconds=1))
    await repository.try_claim(claim)

    outcome = await repository.try_claim(
        MessageClaimDTO(
            event_id=claim.event_id,
            document_id=claim.document_id,
            correlation_id=claim.correlation_id,
            pipeline_version=claim.pipeline_version,
            message_type=claim.message_type,
            lease_owner="worker-2",
            lease_expires_at=NOW + LEASE,
            claimed_at=NOW,
        )
    )

    assert outcome.outcome is ClaimOutcome.RESUME
    assert outcome.attempts == 2


async def test_try_claim_returns_skip_for_completed_message(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim()
    await repository.try_claim(claim)
    await repository.mark_completed(
        claim.event_id, outcome=MessageOutcome.PROCESSED, completed_at=NOW + LEASE
    )

    outcome = await repository.try_claim(claim)

    assert outcome.outcome is ClaimOutcome.SKIP


async def test_resume_carries_persisted_page_numbers(session: AsyncSession) -> None:
    # Возобновление обрабатывает только недостающие страницы, поэтому номера
    # уже сохранённых приходят вместе с исходом claim.
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim(lease_until=NOW - timedelta(seconds=1))
    await repository.try_claim(claim)
    await _persist_page(session, claim, page_number=1)

    outcome = await repository.try_claim(claim)

    assert outcome.outcome is ClaimOutcome.RESUME
    assert outcome.persisted_page_numbers == frozenset({1})


async def test_mark_completed_stores_the_outcome(session: AsyncSession) -> None:
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim()
    await repository.try_claim(claim)

    await repository.mark_completed(
        claim.event_id,
        outcome=MessageOutcome.PARTIALLY_PROCESSED,
        completed_at=NOW + LEASE,
    )

    stored = await _row(session, claim)
    assert stored["status"] == "completed"
    assert stored["outcome"] == MessageOutcome.PARTIALLY_PROCESSED.value
    assert stored["lease_owner"] is None


async def test_release_keeps_message_in_progress(session: AsyncSession) -> None:
    # Работа не завершена, терять её нельзя: следующая доставка получит RESUME
    # немедленно, а не через таймаут лиза.
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim()
    await repository.try_claim(claim)

    await repository.release(claim.event_id, at=NOW + timedelta(minutes=1))

    stored = await _row(session, claim)
    assert stored["status"] == "in_progress"
    assert stored["lease_owner"] is None


async def test_released_message_is_claimable_again(session: AsyncSession) -> None:
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim()
    await repository.try_claim(claim)
    await repository.release(claim.event_id, at=NOW + timedelta(minutes=1))

    outcome = await repository.try_claim(claim)

    assert outcome.outcome is ClaimOutcome.RESUME


async def test_duplicate_insert_inside_savepoint_keeps_transaction_usable(
    session: AsyncSession,
) -> None:
    # В PostgreSQL любая ошибка переводит транзакцию в aborted, поэтому там,
    # где конфликт ожидаем, он обязан быть накрыт точкой отката.
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim()
    await repository.try_claim(claim)

    nested = await session.begin_nested()
    try:
        await _raw_insert(session, claim)
    except DBAPIError:
        await nested.rollback()

    assert await repository.try_claim(_claim()) is not None


async def test_duplicate_insert_without_savepoint_aborts_transaction(
    session: AsyncSession,
) -> None:
    repository = SqlAlchemyProcessedMessageRepository(session)
    claim = _claim()
    await repository.try_claim(claim)
    with pytest.raises(DBAPIError):
        await _raw_insert(session, claim)

    with pytest.raises(DBAPIError):
        await repository.try_claim(_claim())


async def _raw_insert(session: AsyncSession, claim: MessageClaimDTO) -> None:
    await session.execute(
        text(
            "INSERT INTO processed_messages (event_id, document_id, pipeline_version,"
            " message_type, status, lease_owner, lease_expires_at, correlation_id)"
            " VALUES (:event_id, :document_id, :version, :type, 'in_progress',"
            " 'worker-9', :until, :correlation_id)"
        ),
        {
            "event_id": claim.event_id.value,
            "document_id": claim.document_id.value,
            "version": str(claim.pipeline_version),
            "type": claim.message_type,
            "until": claim.lease_expires_at,
            "correlation_id": str(claim.correlation_id),
        },
    )


async def _row(session: AsyncSession, claim: MessageClaimDTO) -> dict[str, object]:
    result = await session.execute(
        text(
            "SELECT status, outcome, lease_owner, attempts FROM processed_messages"
            " WHERE event_id = :event_id"
        ),
        {"event_id": claim.event_id.value},
    )
    return dict(result.mappings().one())


async def _persist_page(
    session: AsyncSession,
    claim: MessageClaimDTO,
    *,
    page_number: int,
) -> None:
    await session.execute(
        text(
            "INSERT INTO documents (id, bucket, object_key, declared_mime_type,"
            " declared_size_bytes, correlation_id) VALUES (:id, 'documents',"
            " :key, 'application/pdf', 1024, :correlation_id)"
        ),
        {
            "id": claim.document_id.value,
            "key": f"{claim.document_id}.pdf",
            "correlation_id": str(claim.correlation_id),
        },
    )
    await session.execute(
        text(
            "INSERT INTO document_pages (id, document_id, pipeline_version,"
            " page_number, status, extraction_method, text, text_length)"
            " VALUES (gen_random_uuid(), :document_id, :version, :number,"
            " 'extracted', 'text_layer', 'договор', 7)"
        ),
        {
            "document_id": claim.document_id.value,
            "version": str(claim.pipeline_version),
            "number": page_number,
        },
    )
