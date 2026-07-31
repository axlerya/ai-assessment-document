"""Запуск конвейера чанкования в пуле процессов.

Точка входа модульная, а не метод: в рабочий процесс уезжает только ссылка на
функцию и её аргументы, а сам конвейер и счётчик токенов строятся уже там —
объект кодировки tiktoken не пиклится.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.chunking.pipeline import build_pipeline
from document_worker.infrastructure.tokenization.tiktoken_counter import (
    TiktokenTokenCounter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.chunking.chunk_assembler import ChunkDraft
    from document_worker.domain.chunking.policy import ChunkingPolicy
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.infrastructure.cpu.executor import CpuPool


def chunk_document(
    pages: tuple[DocumentPage, ...],
    policy: ChunkingPolicy,
) -> tuple[ChunkDraft, ...]:
    """Разбивает страницы на чанки внутри рабочего процесса."""
    return build_pipeline(policy, TiktokenTokenCounter(policy.encoding)).run(pages)


@dataclass(frozen=True, slots=True)
class CpuPoolChunkingRunner:
    """Уводит чанкование с цикла событий в отдельный процесс."""

    pool: CpuPool

    async def run(
        self,
        pages: Sequence[DocumentPage],
        policy: ChunkingPolicy,
    ) -> tuple[ChunkDraft, ...]:
        """Разбивает страницы на черновики чанков."""
        return await self.pool.run(chunk_document, tuple(pages), policy)
