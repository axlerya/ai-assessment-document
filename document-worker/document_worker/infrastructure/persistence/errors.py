"""SQLSTATE → прикладная ошибка.

Ожидаемые дубликаты гасит ON CONFLICT и сюда они не доходят, поэтому долетевший
23505 означает нарушенный инвариант. Неизвестный код считается неисправимым:
повторять непонятное без предела запрещено.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from document_worker.application.errors import (
    DatabaseTimeoutError,
    DatabaseUnavailableError,
    DomainInvariantViolationError,
    DuplicateRecordError,
    PermanentError,
    SchemaMisconfiguredError,
    SerializationConflictError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.exc import DBAPIError

    from document_worker.application.errors import ApplicationError

_UNIQUE_VIOLATION: Final[str] = "23505"

_BY_SQLSTATE: Final[Mapping[str, type[ApplicationError]]] = {
    "40001": SerializationConflictError,  # serialization_failure
    "40P01": SerializationConflictError,  # deadlock_detected
    "55P03": DatabaseTimeoutError,  # lock_not_available (lock_timeout)
    "57014": DatabaseTimeoutError,  # query_canceled (statement_timeout)
    "25P03": DatabaseTimeoutError,  # idle_in_transaction_session_timeout
    "53300": DatabaseUnavailableError,  # too_many_connections
    "53200": DatabaseUnavailableError,  # out_of_memory
    "57P01": DatabaseUnavailableError,  # admin_shutdown
    "57P02": DatabaseUnavailableError,  # crash_shutdown
    "57P03": DatabaseUnavailableError,  # cannot_connect_now
    "08000": DatabaseUnavailableError,
    "08001": DatabaseUnavailableError,
    "08003": DatabaseUnavailableError,
    "08004": DatabaseUnavailableError,
    "08006": DatabaseUnavailableError,
    "23502": DomainInvariantViolationError,  # not_null_violation
    "23503": DomainInvariantViolationError,  # foreign_key_violation
    "23514": DomainInvariantViolationError,  # check_violation
    "22001": DomainInvariantViolationError,  # string_data_right_truncation
    "22003": DomainInvariantViolationError,  # numeric_value_out_of_range
    "42601": SchemaMisconfiguredError,  # syntax_error
    "42703": SchemaMisconfiguredError,  # undefined_column
    "42P01": SchemaMisconfiguredError,  # undefined_table
    "42P10": SchemaMisconfiguredError,  # ON CONFLICT без подходящего арбитра
}


def translate_db_error(error: DBAPIError) -> ApplicationError:
    """Переводит ошибку драйвера в прикладную, сохраняя исходную как причину."""
    sqlstate = _sqlstate_of(error)
    context: dict[str, object] = {"sqlstate": sqlstate}

    if sqlstate == _UNIQUE_VIOLATION:
        translated: ApplicationError = DuplicateRecordError(
            str(error),
            constraint=_constraint_of(error),
            context=context,
        )
    elif error.connection_invalidated:
        # Разорванное соединение неисправимо для этой транзакции, но повтор
        # возьмёт новое — код при этом может быть любым.
        translated = DatabaseUnavailableError(str(error), context=context)
    else:
        failure = _BY_SQLSTATE.get(sqlstate or "", PermanentError)
        translated = failure(str(error), context=context)

    translated.__cause__ = error
    return translated


def _sqlstate_of(error: DBAPIError) -> str | None:
    for candidate in (error.orig, getattr(error.orig, "__cause__", None)):
        code = getattr(candidate, "sqlstate", None) or getattr(
            candidate, "pgcode", None
        )
        if code:
            return str(code)
    return None


def _constraint_of(error: DBAPIError) -> str | None:
    # asyncpg отдаёт имя ограничения полем, поэтому разбирать текст не нужно.
    for candidate in (error.orig, getattr(error.orig, "__cause__", None)):
        name = getattr(candidate, "constraint_name", None)
        if name:
            return str(name)
    return None
