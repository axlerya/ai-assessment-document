"""Захват сообщения: транзакция I0 и снятие лиза.

Порядок внутри транзакции обязателен: сначала проверяется терминальность
прогона, потом занимается сообщение. Обратный порядок оставлял бы в ветке «уже
проиндексирован» запись `in_progress`, которую некому перевести в `completed`.

Состояние источника читается до транзакции: это чужие таблицы, только на
чтение, и держать под них открытую транзакцию незачем.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_worker.application.dto.messaging import ClaimOutcome
from ai_worker.application.errors import ConcurrentIndexing, SourceDocumentNotFound
from ai_worker.domain.entities.document_index import DocumentIndex
from ai_worker.domain.value_objects.enums import IndexStatus
from ai_worker.domain.value_objects.source_snapshot import SourceSnapshot
from ai_worker.domain.value_objects.versioning import ChunkingVersion

if TYPE_CHECKING:
    from datetime import datetime

    from ai_worker.application.config import ProcessingConfig
    from ai_worker.application.dto.commands import IndexDocumentCommand
    from ai_worker.application.ports.reading import ProcessedChunkReader
    from ai_worker.application.ports.system import Clock
    from ai_worker.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
    from ai_worker.domain.value_objects.identifiers import DocumentId, EventId

# Тип сообщения для namespace возобновления `(subject_id, message_type)`.
# Ключ маршрутизации сюда не годится: `document.processed` и
# `document.partially_processed` начинают одну и ту же работу, и разные типы
# разделили бы её на два namespace.
MESSAGE_TYPE = "document.indexing"


@dataclass(frozen=True, slots=True)
class IndexClaim:
    """Чем закончилась попытка занять сообщение."""

    outcome: ClaimOutcome
    index: DocumentIndex | None = None
    # Версия чанкования выбрана вместе с захватом: спрашивать её заново на
    # каждой пачке значило бы ходить в базу за уже известным.
    chunking_version: ChunkingVersion | None = None
    attempts: int = 1

    @property
    def should_process(self) -> bool:
        """Нужно ли браться за работу."""
        return self.outcome in (ClaimOutcome.PROCEED, ClaimOutcome.RESUME)


@dataclass(frozen=True, slots=True)
class MessageClaimService:
    """Занимает сообщение и открывает прогон индексации."""

    uow_factory: UnitOfWorkFactory
    reader: ProcessedChunkReader
    clock: Clock
    config: ProcessingConfig

    async def claim(self, command: IndexDocumentCommand) -> IndexClaim:
        """Занимает сообщение и переводит прогон в работу.

        Raises:
            SourceDocumentNotFound: Строки документа ещё нет.
            ConcurrentIndexing: Документ занят живым лизом другого воркера.
        """
        now = self.clock.now()
        source = await self._source_snapshot(command.document_id)
        version = self.config.embedding.policy.version
        async with self.uow_factory() as uow:
            existing = await uow.index.get(
                document_id=command.document_id, embedding_version=version
            )
            if existing is not None and existing.is_terminal:
                return IndexClaim(
                    outcome=ClaimOutcome.SKIP,
                    index=existing,
                    chunking_version=source.chunking_version,
                )
            claimed = await uow.messages.claim(
                event_id=command.event_id,
                subject_id=command.document_id.value,
                message_type=MESSAGE_TYPE,
                lease_owner=self.config.consumer_name,
                lease_seconds=self.config.claim_lease_s,
                at=now,
            )
            if claimed.outcome is ClaimOutcome.REJECT_CONCURRENT:
                # Попытка расходуется намеренно: иначе живой лиз зависшего
                # воркера гонял бы сообщение по первой ступени без предела.
                raise ConcurrentIndexing(
                    "документ уже обрабатывается другим воркером",
                    context={"document_id": str(command.document_id)},
                )
            if claimed.outcome is ClaimOutcome.SKIP:
                return IndexClaim(
                    outcome=ClaimOutcome.SKIP,
                    index=existing,
                    chunking_version=source.chunking_version,
                    attempts=claimed.attempts,
                )
            index = await self._open_run(uow, command, source, existing, at=now)
            await uow.commit()
            return IndexClaim(
                outcome=claimed.outcome,
                index=index,
                chunking_version=source.chunking_version,
                attempts=claimed.attempts,
            )

    async def release(self, event_id: EventId) -> None:
        """Просрочивает лиз, оставляя работу незавершённой.

        Следующая доставка получит возобновление немедленно, а не через
        таймаут лиза.
        """
        async with self.uow_factory() as uow:
            await uow.messages.release(event_id, at=self.clock.now())
            await uow.commit()

    async def _source_snapshot(self, document_id: DocumentId) -> SourceSnapshot:
        summary = await self.reader.document_summary(document_id)
        if summary is None:
            raise SourceDocumentNotFound(
                "строки документа ещё нет",
                context={"document_id": str(document_id)},
            )
        versions = await self.reader.chunking_versions(document_id)
        return SourceSnapshot(
            pipeline_version=summary.pipeline_version,
            # Версий может быть несколько; брать первую попавшуюся значило бы
            # индексировать устаревший корпус (ADR-0008).
            chunking_version=ChunkingVersion.highest_of(versions),
            status=summary.status,
        )

    async def _open_run(
        self,
        uow: UnitOfWork,
        command: IndexDocumentCommand,
        source: SourceSnapshot,
        existing: DocumentIndex | None,
        *,
        at: datetime,
    ) -> DocumentIndex:
        """Заводит прогон или подхватывает уже открытый."""
        identity = self.config.embedding.policy.identity
        index = existing
        if index is None:
            index = DocumentIndex.pending(
                document_id=command.document_id,
                embedding=identity,
                source=source,
                source_event_id=command.event_id,
                correlation_id=command.correlation_id,
            )
            await uow.index.add(index)
        if index.status is IndexStatus.PENDING:
            return await uow.index.start(index, at=at)
        # Прогон уже в работе: это возобновление после оборванной попытки, и
        # второй переход в `indexing` домен запрещает.
        return index
