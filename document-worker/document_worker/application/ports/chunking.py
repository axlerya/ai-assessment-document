"""Порт исполнения доменного конвейера чанкования вне цикла событий.

Это не снятый ADR-0007 порт `DocumentChunker`: алгоритм здесь не абстрагирован,
в сигнатуре стоят собственные типы доменного конвейера, и никакой другой
реализации, кроме «прогнать этот же конвейер», у порта быть не может.
Абстрагирован пул процессов: конвейер синхронен и держит GIL секундами на
документ, а прямой вызов из корутины срывает heartbeat брокера ровно так же,
как это делали бы PDF-библиотеки.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.chunking.chunk_assembler import ChunkDraft
    from document_worker.domain.chunking.policy import ChunkingPolicy
    from document_worker.domain.entities.document_page import DocumentPage


@runtime_checkable
class ChunkingRunner(Protocol):
    """Выполняет конвейер чанкования, не блокируя цикл событий."""

    async def run(
        self,
        pages: Sequence[DocumentPage],
        policy: ChunkingPolicy,
    ) -> tuple[ChunkDraft, ...]:
        """Разбивает страницы на черновики чанков."""
        ...
