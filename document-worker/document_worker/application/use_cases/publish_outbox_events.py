"""Публикация накопленных событий: транзакции T5a и T5b.

Транзакция не удерживается на время сетевого вызова — тот же довод, что и для
страниц: соединение с базой стоит дороже, чем повторная публикация. Отсюда
три шага: взять пачку под лизом и закоммитить, опубликовать вне транзакции,
отметить опубликованные отдельной транзакцией.

Гарантия — «хотя бы один раз»: сбой между публикацией и отметкой даёт дубль,
и потребители снимают его по детерминированному идентификатору события.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.dto.results import PublishOutboxEventsResult

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from document_worker.application.config import OutboxConfig
    from document_worker.application.dto.results import OutboxRecordDTO
    from document_worker.application.ports.event_publisher import EventPublisher
    from document_worker.application.ports.system import Clock
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory

# Таймаут транзакций реле: обе короткие, ни одна не ждёт сети.
RELAY_TIMEOUT_MS = 5_000


@dataclass(frozen=True, slots=True)
class PublishOutboxEvents:
    """Забирает пачку событий, публикует её и отмечает опубликованные."""

    uow_factory: UnitOfWorkFactory
    publisher: EventPublisher
    clock: Clock
    config: OutboxConfig
    lease_owner: str

    async def execute(self) -> PublishOutboxEventsResult:
        """Публикует одну пачку накопленных событий."""
        now = self.clock.now()
        records = await self._take(now)
        if not records:
            return PublishOutboxEventsResult(fetched=0, published=0, failed=0)

        published: list[UUID] = []
        failures: list[tuple[OutboxRecordDTO, str]] = []
        for record in records:
            try:
                await self.publisher.publish(record)
            except Exception as error:  # noqa: BLE001 — причина уходит в строку события
                # Дальше пачку не гоним: брокер недоступен целиком, и каждая
                # следующая публикация — это ещё один таймаут.
                failures.append((record, str(error)))
                break
            published.append(record.event_id)

        await self._settle(published, failures, now=self.clock.now())
        return PublishOutboxEventsResult(
            fetched=len(records),
            published=len(published),
            failed=len(failures),
        )

    async def _take(self, now: datetime) -> tuple[OutboxRecordDTO, ...]:
        async with self.uow_factory(statement_timeout_ms=RELAY_TIMEOUT_MS) as uow:
            records = await uow.outbox.fetch_pending(
                limit=self.config.batch_size,
                now=now,
                lease_owner=self.lease_owner,
                lease_seconds=self.config.lease_seconds,
            )
            await uow.commit()
        return records

    async def _settle(
        self,
        published: list[UUID],
        failures: list[tuple[OutboxRecordDTO, str]],
        *,
        now: datetime,
    ) -> None:
        # Пачка непуста, а каждое её событие либо опубликовано, либо упало:
        # пустой список сюда не доходит.
        async with self.uow_factory(statement_timeout_ms=RELAY_TIMEOUT_MS) as uow:
            if published:
                await uow.outbox.mark_published(published, published_at=now)
            for record, error in failures:
                await uow.outbox.reschedule(
                    record.event_id,
                    error=error,
                    available_at=now + self.config.backoff_for(record.attempts),
                )
            await uow.commit()
