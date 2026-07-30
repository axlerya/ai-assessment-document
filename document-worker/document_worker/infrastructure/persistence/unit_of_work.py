"""Единица работы: единственное место, где вызывается commit.

Коммит внутри репозитория опубликовал бы событие раньше, чем записан статус
документа, а отметку об обработке оставил бы без её результата. Кроме того,
подтверждать сообщение брокеру после N коммитов не в чем: при падении между
ними неясно, что подтверждать.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Self

from sqlalchemy.exc import DBAPIError

from document_worker.application.errors import PermanentError
from document_worker.infrastructure.persistence.errors import translate_db_error
from document_worker.infrastructure.persistence.repositories.chunks import (
    SqlAlchemyDocumentChunkRepository,
)
from document_worker.infrastructure.persistence.repositories.documents import (
    SqlAlchemyDocumentRepository,
)
from document_worker.infrastructure.persistence.repositories.jobs import (
    SqlAlchemyProcessingJobRepository,
)
from document_worker.infrastructure.persistence.repositories.outbox import (
    SqlAlchemyOutboxRepository,
)
from document_worker.infrastructure.persistence.repositories.pages import (
    SqlAlchemyDocumentPageRepository,
)
from document_worker.infrastructure.persistence.repositories.processed_messages import (
    SqlAlchemyProcessedMessageRepository,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class NestedUnitOfWorkError(PermanentError):
    """Единица работы открыта повторно или используется вне своего блока.

    Повтором это не лечится: так написан вызывающий код.
    """

    code = "nested_unit_of_work"


class SqlAlchemyUnitOfWork:
    """Транзакция и набор репозиториев поверх одной сессии."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Запоминает фабрику; сессия открывается при входе в блок."""
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._committed = False

    async def __aenter__(self) -> Self:
        """Открывает транзакцию и собирает репозитории поверх её сессии.

        Вложенная единица работы взяла бы второе соединение из пула, невидимое
        для первого: дедлок сам с собой.
        """
        if self._session is not None:
            raise NestedUnitOfWorkError(
                "единица работы уже открыта: передавайте её аргументом"
            )
        session = self._session_factory()
        # Транзакцией владеет сама сессия, объект перехода не нужен.
        _ = await session.begin()
        self._session = session
        self._committed = False
        self.documents = SqlAlchemyDocumentRepository(session)
        self.pages = SqlAlchemyDocumentPageRepository(session)
        self.chunks = SqlAlchemyDocumentChunkRepository(session)
        self.jobs = SqlAlchemyProcessingJobRepository(session)
        self.messages = SqlAlchemyProcessedMessageRepository(session)
        self.outbox = SqlAlchemyOutboxRepository(session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Откатывает всё, что не было закоммичено явно, и закрывает сессию."""
        session = self._require_session()
        try:
            if exc_type is not None or not self._committed:
                await session.rollback()
        finally:
            await session.close()
            self._session = None
            self._committed = False

    async def commit(self) -> None:
        """Фиксирует транзакцию."""
        session = self._require_session()
        try:
            await session.commit()
        except DBAPIError as error:
            await session.rollback()
            raise translate_db_error(error) from error
        self._committed = True

    async def rollback(self) -> None:
        """Откатывает транзакцию."""
        await self._require_session().rollback()

    async def flush(self) -> None:
        """Отправляет накопленные изменения, не завершая транзакцию."""
        session = self._require_session()
        try:
            await session.flush()
        except DBAPIError as error:
            raise translate_db_error(error) from error

    @asynccontextmanager
    async def savepoint(self) -> AsyncIterator[None]:
        """Точка отката внутри транзакции.

        В PostgreSQL любая ошибка переводит транзакцию в aborted, поэтому там,
        где конфликт ожидаем, продолжить её без точки отката нельзя.
        """
        nested = await self._require_session().begin_nested()
        try:
            yield
        except BaseException:
            if nested.is_active:
                await nested.rollback()
            raise
        else:
            await nested.commit()

    def _require_session(self) -> AsyncSession:
        if self._session is None:
            raise NestedUnitOfWorkError("единица работы используется вне своего блока")
        return self._session
