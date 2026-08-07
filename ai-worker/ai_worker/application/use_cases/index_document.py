"""Индексация документа: I0 → I1ₙ → I2.

Чтение чанков и прогон модели идут вне открытой транзакции (ADR-0006): инференс
занимает минуты, и держать на это время транзакцию — то же, от чего отказался
document-worker.

Каждая пачка фиксируется своей транзакцией. Падение между пачками оставляет
документ возобновляемым: уже построенные эмбеддинги на месте, и следующая
доставка не считает их заново.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_worker.application.events import document_indexed_event
from ai_worker.domain.entities.chunk_embedding import ChunkEmbedding
from ai_worker.domain.errors import InvariantViolation
from ai_worker.domain.value_objects.enums import IndexStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from ai_worker.application.config import ProcessingConfig
    from ai_worker.application.dto.commands import IndexDocumentCommand
    from ai_worker.application.dto.messaging import ClaimOutcome
    from ai_worker.application.ports.embedding import EmbeddingProvider
    from ai_worker.application.ports.reading import ProcessedChunkReader
    from ai_worker.application.ports.system import Clock
    from ai_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from ai_worker.application.services.message_claim import MessageClaimService
    from ai_worker.domain.entities.document_index import DocumentIndex
    from ai_worker.domain.entities.source_chunk import SourceChunk
    from ai_worker.domain.value_objects.hashing import ContentHash
    from ai_worker.domain.value_objects.identifiers import ChunkId, DocumentId
    from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector
    from ai_worker.domain.value_objects.versioning import ChunkingVersion

    Vectors = tuple[DenseVector, SparseVector]

FAILURE_CODE = "no_indexable_chunks"


@dataclass(frozen=True, slots=True)
class IndexDocumentResult:
    """Чем закончилась обработка сообщения."""

    outcome: ClaimOutcome
    status: IndexStatus | None = None
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_failed: int = 0
    events_enqueued: int = 0


@dataclass(frozen=True, slots=True)
class _Tally:
    """Счётчики прогона."""

    total: int = 0
    embedded: int = 0
    failed: int = 0

    def plus(self, *, embedded: int, failed: int) -> _Tally:
        """Прибавляет итог одной пачки."""
        return _Tally(
            total=self.total,
            embedded=self.embedded + embedded,
            failed=self.failed + failed,
        )


@dataclass(frozen=True, slots=True)
class IndexDocument:
    """Строит эмбеддинги документа и объявляет его проиндексированным."""

    claims: MessageClaimService
    uow_factory: UnitOfWorkFactory
    reader: ProcessedChunkReader
    embeddings: EmbeddingProvider
    clock: Clock
    config: ProcessingConfig

    async def __call__(self, command: IndexDocumentCommand) -> IndexDocumentResult:
        """Обрабатывает одно сообщение об обработанном документе."""
        started_at = self.clock.now()
        claim = await self.claims.claim(command)
        if not claim.should_process or claim.index is None:
            return IndexDocumentResult(
                outcome=claim.outcome,
                status=claim.index.status if claim.index else None,
            )
        # Версия чанкования выбрана вместе с захватом — там же, где прочитано
        # состояние источника.
        chunking_version = claim.chunking_version
        if chunking_version is None:  # pragma: no cover — захват их отдаёт вместе
            msg = "захват вернул прогон без версии чанкования"
            raise RuntimeError(msg)
        try:
            tally = await self._embed_document(
                command.document_id, chunking_version=chunking_version
            )
            return await self._finish(
                command,
                index=claim.index,
                chunking_version=chunking_version,
                tally=tally,
                started_at=started_at,
                outcome=claim.outcome,
            )
        except Exception:
            # Лиз отпускается, чтобы следующая доставка получила возобновление
            # немедленно, а не через таймаут захвата.
            await self.claims.release(command.event_id)
            raise

    async def _embed_document(
        self,
        document_id: DocumentId,
        *,
        chunking_version: ChunkingVersion,
    ) -> _Tally:
        stored = await self._stored_hashes(document_id)
        tally = _Tally()
        total = 0
        pending: list[SourceChunk] = []
        async for chunk in self.reader.chunks(
            document_id, chunking_version=chunking_version
        ):
            total += 1
            if stored.get(chunk.ref.chunk_id) == chunk.content_hash:
                # Текст не менялся и версия та же: пересчитывать нечего.
                tally = tally.plus(embedded=1, failed=0)
                continue
            pending.append(chunk)
            if len(pending) >= self.config.embedding.batch_size:
                tally = await self._flush(pending, tally)
                pending = []
        if pending:
            tally = await self._flush(pending, tally)
        return _Tally(total=total, embedded=tally.embedded, failed=tally.failed)

    async def _stored_hashes(
        self,
        document_id: DocumentId,
    ) -> Mapping[ChunkId, ContentHash]:
        async with self.uow_factory() as uow:
            return await uow.embeddings.stored_hashes(
                document_id=document_id,
                embedding_version=self.config.embedding.policy.version,
            )

    async def _flush(self, chunks: Sequence[SourceChunk], tally: _Tally) -> _Tally:
        """Считает пачку и фиксирует её своей транзакцией (I1ₙ)."""
        vectors = await self._embed(chunks)
        identity = self.config.embedding.policy.identity
        built = [
            ChunkEmbedding.of(
                chunk=chunk, embedding=identity, dense=pair[0], sparse=pair[1]
            )
            for chunk, pair in zip(chunks, vectors, strict=True)
            if pair is not None
        ]
        if built:
            async with self.uow_factory() as uow:
                await uow.embeddings.add_many(built)
                await uow.commit()
        return tally.plus(embedded=len(built), failed=len(chunks) - len(built))

    async def _embed(self, chunks: Sequence[SourceChunk]) -> list[Vectors | None]:
        """Считает пачку, отдавая `None` за чанк, который модель не осилила."""
        try:
            pairs = await self.embeddings.embed_passages(
                [chunk.text for chunk in chunks],
                timeout_s=self.config.embedding.timeout_s,
            )
        except InvariantViolation:
            if len(chunks) == 1:
                # Один нечитаемый фрагмент не имеет права стоить документа.
                return [None]
            # Пачка не сообщает, какой чанк её испортил, и повторный счёт по
            # одному — единственный способ отдать отказ виновному, а не всем.
            isolated: list[Vectors | None] = []
            for chunk in chunks:
                isolated.extend(await self._embed([chunk]))
            return isolated
        return list(pairs)

    async def _finish(  # noqa: PLR0913 — терминальная транзакция описывается всем этим
        self,
        command: IndexDocumentCommand,
        *,
        index: DocumentIndex,
        chunking_version: ChunkingVersion,
        tally: _Tally,
        started_at: datetime,
        outcome: ClaimOutcome,
    ) -> IndexDocumentResult:
        """Пишет терминальное состояние, событие и отметку сообщения (I2)."""
        now = self.clock.now()
        finished = self._terminal(index, tally, at=now)
        duration_ms = max(0, int((now - started_at).total_seconds() * 1000))
        enqueued = 0
        async with self.uow_factory() as uow:
            applied = await uow.index.finish(finished, expected=IndexStatus.INDEXING)
            if applied and finished.status is IndexStatus.INDEXED:
                enqueued = await uow.outbox.enqueue(
                    [
                        document_indexed_event(
                            finished,
                            chunking_version=chunking_version,
                            duration_ms=duration_ms,
                            occurred_at=now,
                        )
                    ]
                )
            await uow.messages.mark_completed(command.event_id, at=now)
            await uow.commit()
        return IndexDocumentResult(
            outcome=outcome,
            status=finished.status,
            chunks_total=tally.total,
            chunks_embedded=tally.embedded,
            chunks_failed=tally.failed,
            events_enqueued=enqueued,
        )

    def _terminal(
        self,
        index: DocumentIndex,
        tally: _Tally,
        *,
        at: datetime,
    ) -> DocumentIndex:
        if tally.embedded == 0:
            # Документ без единого эмбеддинга не проиндексирован: поиск нашёл
            # бы по нему ровно ничего, а документ считался бы готовым.
            return index.fail(
                code=FAILURE_CODE,
                message="ни один чанк не дал пригодного представления",
                at=at,
                chunks_total=tally.total,
                chunks_failed=tally.failed,
            )
        return index.complete(
            chunks_total=tally.total,
            chunks_embedded=tally.embedded,
            chunks_failed=tally.failed,
            at=at,
        )
