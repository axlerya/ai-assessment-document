"""Перевод отказов драйвера в прикладную классификацию.

Без перевода подписчик видит `DBAPIError` и решить ничего не может. Разница
между «база моргнула» и «строка противоречит схеме» стоит здесь либо
потерянного документа, либо вечного цикла повторов.

Неизвестный код состояния считается временным. Это осознанный перекос: лишний
повтор ограничен лестницей и стоит минут, а ошибочно подтверждённое сообщение
теряет документ навсегда.
"""

from __future__ import annotations

from typing import Final

from sqlalchemy.exc import IntegrityError

from ai_worker.application.errors import (
    ApplicationError,
    DuplicateRecord,
    InvariantRejectedByStorage,
    StorageConflict,
    StorageUnavailable,
)

UNIQUE_VIOLATION: Final[str] = "23505"

# Строка противоречит схеме: повтор даст ровно то же самое.
BROKEN_INVARIANT: Final[frozenset[str]] = frozenset(
    {
        "23514",  # нарушение CHECK
        "23503",  # нарушение внешнего ключа
        "23502",  # NOT NULL
        "23001",  # нарушение ограничения ссылочной целостности
        "22001",  # значение длиннее колонки
        "22003",  # число вне диапазона типа
        "22P02",  # неверное текстовое представление
    }
)

# Транзакция не прошла из-за конкуренции: её повтор уместен сразу.
CONCURRENCY: Final[frozenset[str]] = frozenset(
    {
        "40001",  # сбой сериализации
        "40P01",  # взаимоблокировка
    }
)


def translate(error: Exception) -> ApplicationError:
    """Переводит отказ драйвера в прикладную ошибку.

    Уже классифицированная ошибка возвращается как есть: повторный перевод мог
    бы превратить `PermanentError` во временную и отправить документ в вечный
    цикл повторов.
    """
    if isinstance(error, ApplicationError):
        return error
    sqlstate = _sqlstate(error)
    context: dict[str, object] = {"sqlstate": sqlstate or "unknown"}
    translated = _classify(error, sqlstate, context)
    translated.__cause__ = error
    return translated


def _classify(
    error: Exception,
    sqlstate: str | None,
    context: dict[str, object],
) -> ApplicationError:
    if sqlstate == UNIQUE_VIOLATION:
        return DuplicateRecord("строка с таким ключом уже есть", context=context)
    if sqlstate in CONCURRENCY:
        return StorageConflict(
            "транзакция не прошла из-за конкуренции", context=context
        )
    if sqlstate in BROKEN_INVARIANT or (
        isinstance(error, IntegrityError) and sqlstate is not None
    ):
        return InvariantRejectedByStorage("строка противоречит схеме", context=context)
    return StorageUnavailable("хранилище недоступно", context=context)


def _sqlstate(error: Exception) -> str | None:
    """Достаёт код состояния PostgreSQL из отказа драйвера.

    Код лежит то на самом исключении SQLAlchemy, то на обёрнутом исключении
    драйвера, и называется он в разных версиях `sqlstate` или `pgcode`.
    """
    for candidate in (error, getattr(error, "orig", None)):
        state = getattr(candidate, "sqlstate", None) or getattr(
            candidate, "pgcode", None
        )
        if isinstance(state, str) and state:
            return state
    return None
