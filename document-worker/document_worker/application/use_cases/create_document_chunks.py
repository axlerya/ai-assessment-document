"""Чанкование документа: чтение вне транзакции, вставка одной пачкой.

Все чанки документа пишутся одним batch-INSERT в одной транзакции. Частичное
чанкование не поддерживается намеренно: набор, собранный по остатку страниц,
получил бы сквозные номера, уже занятые чанками первых страниц, и вставка
молча отбросила бы половину документа — без дублей, без ошибок и без единого
наблюдаемого симптома, кроме того, что retrieval никогда не найдёт вторую
половину договора.

Порядок последовательности значим: сквозной номер присваивается перечислением,
и переставленный список дал бы другую нумерацию при повторе.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from document_worker.application.dto.results import (
    CreateDocumentChunksResult,
    JobProgressDTO,
)
from document_worker.application.errors import translate_domain_error
from document_worker.domain.entities.document_chunk import DocumentChunk
from document_worker.domain.errors import DomainError
from document_worker.domain.value_objects.enums import PageStatus
from document_worker.domain.value_objects.identifiers import ChunkId

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from document_worker.application.config import ProcessingConfig
    from document_worker.application.dto.commands import CreateDocumentChunksCommand
    from document_worker.application.ports.chunking import ChunkingRunner
    from document_worker.application.ports.system import Clock, IdGenerator
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.chunking.chunk_assembler import ChunkDraft
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.value_objects.identifiers import PageId

# Страница с непустым текстом чанкуется, даже если распознана она плохо: устав
# требует сохранить исходный результат и пометить диапазон, а не выбросить его.
# Страницы failed текста не имеют вовсе и чанков не порождают по построению.
CHUNKABLE_PAGE_STATUSES: Final[frozenset[PageStatus]] = frozenset(
    {PageStatus.EXTRACTED, PageStatus.PARTIALLY_ILLEGIBLE, PageStatus.ILLEGIBLE}
)


@dataclass(frozen=True, slots=True)
class CreateDocumentChunks:
    """Разбивает прочитанные страницы документа на чанки."""

    uow_factory: UnitOfWorkFactory
    chunker: ChunkingRunner
    ids: IdGenerator
    clock: Clock
    config: ProcessingConfig

    async def execute(
        self,
        command: CreateDocumentChunksCommand,
    ) -> CreateDocumentChunksResult:
        """Чанкует документ целиком и фиксирует результат транзакцией T3.

        Raises:
            ChunkPersistenceMismatchError: Вставлено не столько чанков, сколько
                подготовлено, и недостача не объясняется повтором.
        """
        pages = await self._load_pages(command)
        if not pages:
            return await self._persist(command, ())
        drafts = await self.chunker.run(pages, self.config.chunking)
        return await self._persist(command, self._build(pages, drafts))

    async def _load_pages(
        self,
        command: CreateDocumentChunksCommand,
    ) -> tuple[DocumentPage, ...]:
        """Транзакция T3r: только чтение, чанкование идёт уже вне неё."""
        async with self.uow_factory(
            statement_timeout_ms=self.config.tx.chunks_ms, read_only=True
        ) as uow:
            return await uow.pages.load_pages(
                command.document_id,
                self.config.pipeline_version,
                statuses=CHUNKABLE_PAGE_STATUSES,
            )

    def _build(
        self,
        pages: Sequence[DocumentPage],
        drafts: Sequence[ChunkDraft],
    ) -> tuple[DocumentChunk, ...]:
        by_id: Mapping[PageId, DocumentPage] = {page.id: page for page in pages}
        ordered = sorted(
            drafts, key=lambda draft: (int(draft.page_number), draft.ordinal)
        )
        try:
            return tuple(self._chunk(by_id[draft.page_id], draft) for draft in ordered)
        except DomainError as error:
            raise translate_domain_error(error) from error

    def _chunk(self, page: DocumentPage, draft: ChunkDraft) -> DocumentChunk:
        return DocumentChunk.from_page_slice(
            chunk_id=ChunkId(self.ids.new_uuid()),
            page=page,
            ordinal=draft.ordinal,
            span=draft.span,
            avg_confidence=draft.quality.avg_confidence,
            token_count=draft.token_count,
            chunking_version=self.config.chunking.version,
            heading_path=draft.heading_path,
            overlap_prefix_chars=draft.overlap_prefix_chars,
        )

    async def _persist(
        self,
        command: CreateDocumentChunksCommand,
        chunks: Sequence[DocumentChunk],
    ) -> CreateDocumentChunksResult:
        now = self.clock.now()
        async with self.uow_factory(
            statement_timeout_ms=self.config.tx.chunks_ms
        ) as uow:
            created = await uow.chunks.add_all(chunks)
            # Итог берётся из базы, а не из числа вставленных: на повторе
            # вставлено ноль, а чанки у документа есть.
            total = await uow.chunks.count(
                command.document_id, self.config.chunking.version
            )
            await uow.jobs.record_progress(
                command.job_id,
                JobProgressDTO(heartbeat_at=now, chunks_created=created),
            )
            await uow.commit()
        return CreateDocumentChunksResult(chunks_created=created, chunks_total=total)
