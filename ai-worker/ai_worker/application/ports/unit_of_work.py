"""Границы транзакции."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from types import TracebackType

    from ai_worker.application.ports.repositories import (
        DocumentIndexRepository,
        DraftRepository,
        EmbeddingRepository,
        OutboxRepository,
        ProcessedMessageRepository,
        RetrievalHistoryRepository,
    )


@runtime_checkable
class UnitOfWork(Protocol):
    """Транзакция вместе с репозиториями, которые в ней работают."""

    index: DocumentIndexRepository
    embeddings: EmbeddingRepository
    drafts: DraftRepository
    retrieval: RetrievalHistoryRepository
    messages: ProcessedMessageRepository
    outbox: OutboxRepository

    async def __aenter__(self) -> Self:
        """Открывает транзакцию."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Закрывает транзакцию: без явного `commit()` — откатом."""
        ...

    async def commit(self) -> None:
        """Помечает работу завершённой."""
        ...


@runtime_checkable
class UnitOfWorkFactory(Protocol):
    """Открывает новую единицу работы."""

    def __call__(self) -> UnitOfWork:
        """Возвращает единицу работы, готовую к входу в контекст."""
        ...
