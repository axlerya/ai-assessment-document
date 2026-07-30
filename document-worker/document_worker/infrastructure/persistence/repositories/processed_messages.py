"""Барьер идемпотентности доставки.

Порядок исходов задаётся одним INSERT ... ON CONFLICT DO UPDATE: проверять
наличие строки отдельным запросом нельзя, между проверкой и вставкой
вклинивается второй воркер.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import literal_column, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from document_worker.application.dto.results import ClaimOutcome, ClaimOutcomeDTO
from document_worker.infrastructure.persistence.models.message import (
    ProcessedMessageRow,
)
from document_worker.infrastructure.persistence.models.page import DocumentPageRow

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy import Executable
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.application.dto.results import (
        MessageClaimDTO,
        MessageOutcome,
    )
    from document_worker.domain.value_objects.identifiers import EventId

IN_PROGRESS = "in_progress"
COMPLETED = "completed"


class SqlAlchemyProcessedMessageRepository:
    """Claim сообщения с лизом и его завершение."""

    def __init__(self, session: AsyncSession) -> None:
        """Работает в транзакции переданной сессии."""
        self._session = session

    async def try_claim(self, claim: MessageClaimDTO) -> ClaimOutcomeDTO:
        """Пытается занять сообщение и сообщает, что делать дальше."""
        insert = pg_insert(ProcessedMessageRow).values(
            event_id=claim.event_id.value,
            document_id=claim.document_id.value,
            pipeline_version=str(claim.pipeline_version),
            message_type=claim.message_type,
            status=IN_PROGRESS,
            lease_owner=claim.lease_owner,
            lease_expires_at=claim.lease_expires_at,
            correlation_id=str(claim.correlation_id),
            attempts=1,
            first_seen_at=claim.claimed_at,
            updated_at=claim.claimed_at,
        )
        # Лиз перехватывается только у протухшей записи: живой означает, что
        # документ прямо сейчас обрабатывает другой воркер.
        statement: Executable = insert.on_conflict_do_update(
            index_elements=[ProcessedMessageRow.event_id],
            set_={
                "lease_owner": insert.excluded.lease_owner,
                "lease_expires_at": insert.excluded.lease_expires_at,
                "attempts": ProcessedMessageRow.attempts + 1,
                "updated_at": insert.excluded.updated_at,
            },
            where=(
                (ProcessedMessageRow.status == IN_PROGRESS)
                & (ProcessedMessageRow.lease_expires_at <= claim.claimed_at)
            ),
        ).returning(
            literal_column("xmax = 0").label("inserted"),
            ProcessedMessageRow.attempts,
        )
        claimed = (await self._session.execute(statement)).one_or_none()
        if claimed is None:
            return await self._rejected(claim)
        if claimed.inserted:
            return ClaimOutcomeDTO(outcome=ClaimOutcome.PROCEED, attempts=1)
        return ClaimOutcomeDTO(
            outcome=ClaimOutcome.RESUME,
            persisted_page_numbers=await self._persisted_page_numbers(claim),
            attempts=claimed.attempts,
        )

    async def mark_completed(
        self,
        event_id: EventId,
        *,
        outcome: MessageOutcome,
        completed_at: datetime,
    ) -> None:
        """Отмечает сообщение обработанным — только вместе с результатом."""
        statement = (
            update(ProcessedMessageRow)
            .where(ProcessedMessageRow.event_id == event_id.value)
            .values(
                status=COMPLETED,
                outcome=outcome.value,
                lease_owner=None,
                lease_expires_at=None,
                completed_at=completed_at,
                updated_at=completed_at,
            )
        )
        await self._session.execute(statement)

    async def release(self, event_id: EventId, *, at: datetime) -> None:
        """Просрочивает лиз, оставляя запись незавершённой.

        Статус остаётся in_progress: работа не закончена. Владелец не снимается
        — запись без владельца схема не принимает, да и обнулять его незачем:
        истёкший лиз уже даёт следующей доставке RESUME, причём немедленно.
        """
        statement = (
            update(ProcessedMessageRow)
            .where(
                ProcessedMessageRow.event_id == event_id.value,
                ProcessedMessageRow.status == IN_PROGRESS,
            )
            .values(lease_expires_at=at, updated_at=at)
        )
        await self._session.execute(statement)

    async def _rejected(self, claim: MessageClaimDTO) -> ClaimOutcomeDTO:
        statement = select(
            ProcessedMessageRow.status, ProcessedMessageRow.attempts
        ).where(ProcessedMessageRow.event_id == claim.event_id.value)
        row = (await self._session.execute(statement)).one()
        outcome = (
            ClaimOutcome.SKIP
            if row.status == COMPLETED
            else ClaimOutcome.REJECT_CONCURRENT
        )
        return ClaimOutcomeDTO(outcome=outcome, attempts=row.attempts)

    async def _persisted_page_numbers(self, claim: MessageClaimDTO) -> frozenset[int]:
        statement = select(DocumentPageRow.page_number).where(
            DocumentPageRow.document_id == claim.document_id.value,
            DocumentPageRow.pipeline_version == str(claim.pipeline_version),
        )
        return frozenset((await self._session.execute(statement)).scalars().all())
