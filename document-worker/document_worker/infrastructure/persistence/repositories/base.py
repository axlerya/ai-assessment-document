"""Общая механика репозиториев.

Единственное место, где выполняются запросы: иначе трансляция ошибок драйвера
разошлась бы по двум десяткам методов и где-нибудь отстала.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import DBAPIError

from document_worker.infrastructure.persistence.errors import translate_db_error

if TYPE_CHECKING:
    from sqlalchemy import Executable
    from sqlalchemy.engine import Result
    from sqlalchemy.ext.asyncio import AsyncSession


class SqlAlchemyRepository:
    """Репозиторий поверх сессии Unit of Work. Метода commit здесь нет."""

    def __init__(self, session: AsyncSession) -> None:
        """Работает в транзакции переданной сессии."""
        self._session = session

    async def _execute(self, statement: Executable) -> Result[Any]:
        try:
            return await self._session.execute(statement)
        except DBAPIError as error:
            raise translate_db_error(error) from error
