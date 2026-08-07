"""Общая часть репозиториев.

`commit()` здесь нет и не будет: границы транзакции знает только тот, кто
знает границы работы. Репозиторий, коммитящий сам, оставил бы половину
результата зафиксированной, а вторую — потерянной, причём без единой ошибки.

Каждый запрос проходит через трансляцию отказов: наружу выходит прикладная
ошибка с классом, а не `DBAPIError`, по которому решить ничего нельзя.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from ai_worker.infrastructure.persistence.errors import translate

if TYPE_CHECKING:
    from sqlalchemy.engine import Result
    from sqlalchemy.ext.asyncio import AsyncConnection
    from sqlalchemy.sql import Executable


@dataclass(frozen=True, slots=True)
class SqlAlchemyRepository:
    """Репозиторий поверх соединения, которым владеет единица работы."""

    connection: AsyncConnection

    async def _execute(self, statement: Executable) -> Result[Any]:
        """Выполняет запрос, переводя отказ драйвера в прикладную ошибку.

        Raises:
            ApplicationError: Запрос не прошёл; класс ошибки определяет, что
                делать с сообщением дальше.
        """
        try:
            return await self.connection.execute(statement)
        except SQLAlchemyError as error:
            raise translate(error) from error


def values_of(row: object) -> dict[str, object]:
    """Значения полей, которые строке действительно задали.

    Берутся именно заданные поля, а не «непустые». Разница видна на пакетной
    вставке: SQLAlchemy строит запрос по ключам первой строки, и если у одной
    строки поле опущено как `None`, а у другой заполнено, то у второй оно
    молча пропадёт. Незаданное поле при этом остаётся за схемой — у неё есть
    DEFAULT, и подставлять `None` поверх него нельзя.
    """
    columns = {column.name for column in type(row).__table__.columns}  # type: ignore[attr-defined]
    return {name: value for name, value in vars(row).items() if name in columns}
