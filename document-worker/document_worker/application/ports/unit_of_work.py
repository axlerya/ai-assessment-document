"""Транзакционная граница. Единственное место, где вызывается commit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from document_worker.application.ports.repositories import (
        DocumentChunkRepository,
        DocumentPageRepository,
        DocumentRepository,
        OutboxRepository,
        ProcessedMessageRepository,
        ProcessingJobRepository,
    )


@runtime_checkable
class UnitOfWork(Protocol):
    """Набор репозиториев в одной транзакции."""

    documents: DocumentRepository
    pages: DocumentPageRepository
    chunks: DocumentChunkRepository
    jobs: ProcessingJobRepository
    messages: ProcessedMessageRepository
    outbox: OutboxRepository

    async def commit(self) -> None:
        """Фиксирует транзакцию."""
        ...

    async def rollback(self) -> None:
        """Откатывает транзакцию."""
        ...

    async def flush(self) -> None:
        """Отправляет накопленные изменения, не завершая транзакцию."""
        ...


@runtime_checkable
class UnitOfWorkFactory(Protocol):
    """Открывает транзакцию с явным таймаутом.

    Выход из контекста без ошибки не коммитит: нужен явный commit.
    """

    def __call__(
        self,
        *,
        statement_timeout_ms: int,
        read_only: bool = False,
    ) -> AbstractAsyncContextManager[UnitOfWork]:
        """Открывает единицу работы."""
        ...
