"""Единица работы: границы транзакции знает сценарий, а не репозиторий.

Выход из контекста без явного `commit()` откатывает всё сделанное. Это не
перестраховка: сценарий, оборвавшийся посередине, обязан не оставить после
себя половины результата — иначе повторная доставка увидит состояние, которого
не бывает, и решит, что работа сделана.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from ai_worker.infrastructure.persistence.repositories.drafts import (
    SqlAlchemyDraftRepository,
    SqlAlchemyRetrievalHistoryRepository,
)
from ai_worker.infrastructure.persistence.repositories.index import (
    SqlAlchemyDocumentIndexRepository,
    SqlAlchemyEmbeddingRepository,
)
from ai_worker.infrastructure.persistence.repositories.messaging import (
    SqlAlchemyOutboxRepository,
    SqlAlchemyProcessedMessageRepository,
)

if TYPE_CHECKING:
    from types import TracebackType

    from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(slots=True)
class SqlAlchemyUnitOfWork:
    """Транзакция вместе с репозиториями, которые в ней работают."""

    connection: AsyncConnection
    index: SqlAlchemyDocumentIndexRepository
    embeddings: SqlAlchemyEmbeddingRepository
    drafts: SqlAlchemyDraftRepository
    retrieval: SqlAlchemyRetrievalHistoryRepository
    messages: SqlAlchemyProcessedMessageRepository
    outbox: SqlAlchemyOutboxRepository
    committed: bool = False

    @classmethod
    def on(cls, connection: AsyncConnection) -> Self:
        """Собирает единицу работы поверх готового соединения."""
        return cls(
            connection=connection,
            index=SqlAlchemyDocumentIndexRepository(connection),
            embeddings=SqlAlchemyEmbeddingRepository(connection),
            drafts=SqlAlchemyDraftRepository(connection),
            retrieval=SqlAlchemyRetrievalHistoryRepository(connection),
            messages=SqlAlchemyProcessedMessageRepository(connection),
            outbox=SqlAlchemyOutboxRepository(connection),
        )

    async def __aenter__(self) -> Self:
        """Открывает вложенную транзакцию сценария."""
        # Возврат отбрасывается намеренно: транзакция достаётся из
        # соединения при выходе, и второй ссылки на неё не заводится —
        # иначе их состояния могли бы разойтись.
        _ = await self.connection.begin_nested()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Фиксирует работу только при явном `commit()`."""
        del exc, traceback
        nested = self.connection.get_nested_transaction()
        if nested is None:  # pragma: no cover — контекст всегда открывает свою
            return
        if exc_type is None and self.committed:
            await nested.commit()
        else:
            await nested.rollback()

    async def commit(self) -> None:
        """Помечает работу завершённой: выход из контекста её зафиксирует."""
        self.committed = True
