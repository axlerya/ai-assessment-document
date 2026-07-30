"""Трансляция ошибок драйвера в прикладные.

Решение о повторе принимает application, а SQLSTATE знает только инфраструктура,
поэтому перевод живёт здесь и обязан быть полным: неизвестный код не повторяют.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import DBAPIError, IntegrityError

from document_worker.application.errors import (
    DatabaseTimeoutError,
    DatabaseUnavailableError,
    DomainInvariantViolationError,
    DuplicateRecordError,
    PermanentError,
    SchemaMisconfiguredError,
    SerializationConflictError,
    TransientError,
)
from document_worker.infrastructure.persistence.errors import translate_db_error

pytestmark = pytest.mark.unit


class _AsyncpgError(Exception):
    """Заглушка ошибки драйвера: SQLAlchemy читает у неё sqlstate."""

    def __init__(self, sqlstate: str, constraint_name: str | None = None) -> None:
        super().__init__(f"asyncpg error {sqlstate}")
        self.sqlstate = sqlstate
        self.pgcode = sqlstate
        self.constraint_name = constraint_name


def _error(
    sqlstate: str,
    *,
    constraint: str | None = None,
    integrity: bool = False,
    invalidated: bool = False,
) -> DBAPIError:
    original = _AsyncpgError(sqlstate, constraint)
    failure = IntegrityError if integrity else DBAPIError
    error = failure("SELECT 1", None, original)
    error.connection_invalidated = invalidated
    return error


def test_unique_violation_becomes_duplicate_record_error() -> None:
    translated = translate_db_error(
        _error(
            "23505",
            constraint="uq__document_pages__document__version__number",
            integrity=True,
        )
    )

    assert isinstance(translated, DuplicateRecordError)
    assert translated.constraint == "uq__document_pages__document__version__number"


@pytest.mark.parametrize("sqlstate", ["40001", "40P01"])
def test_serialization_failure_is_transient(sqlstate: str) -> None:
    translated = translate_db_error(_error(sqlstate))

    assert isinstance(translated, SerializationConflictError)
    assert isinstance(translated, TransientError)


@pytest.mark.parametrize("sqlstate", ["57014", "55P03", "25P03"])
def test_timeouts_become_database_timeout(sqlstate: str) -> None:
    assert isinstance(translate_db_error(_error(sqlstate)), DatabaseTimeoutError)


@pytest.mark.parametrize("sqlstate", ["53300", "53200", "57P01", "08006", "08003"])
def test_unavailability_becomes_database_unavailable(sqlstate: str) -> None:
    assert isinstance(translate_db_error(_error(sqlstate)), DatabaseUnavailableError)


def test_invalidated_connection_is_transient_whatever_the_code() -> None:
    translated = translate_db_error(_error("XX000", invalidated=True))

    assert isinstance(translated, DatabaseUnavailableError)


@pytest.mark.parametrize("sqlstate", ["23502", "23503", "23514", "22001", "22003"])
def test_constraint_violations_are_domain_invariant_violations(sqlstate: str) -> None:
    translated = translate_db_error(_error(sqlstate, integrity=True))

    assert isinstance(translated, DomainInvariantViolationError)


@pytest.mark.parametrize("sqlstate", ["42601", "42703", "42P01", "42P10"])
def test_schema_errors_become_schema_misconfigured(sqlstate: str) -> None:
    assert isinstance(translate_db_error(_error(sqlstate)), SchemaMisconfiguredError)


def test_unknown_sqlstate_is_permanent() -> None:
    # Бесконечно повторять непонятное запрещено: неизвестный код неисправим.
    translated = translate_db_error(_error("XX000"))

    assert isinstance(translated, PermanentError)
    assert not isinstance(translated, TransientError)


def test_missing_sqlstate_is_permanent() -> None:
    error = DBAPIError("SELECT 1", None, Exception("нет кода"))
    error.connection_invalidated = False

    assert isinstance(translate_db_error(error), PermanentError)


def test_translated_error_keeps_the_driver_exception_as_cause() -> None:
    original = _error("40001")

    translated = translate_db_error(original)

    assert translated.__cause__ is original


def test_translated_error_carries_sqlstate_in_context() -> None:
    translated = translate_db_error(_error("40001"))

    assert translated.context["sqlstate"] == "40001"


def test_constraint_name_is_absent_when_the_driver_does_not_report_it() -> None:
    error = DBAPIError("INSERT", None, Exception("нет имени"))
    error.connection_invalidated = False
    error.orig.sqlstate = "23505"  # type: ignore[union-attr]

    translated = translate_db_error(error)

    assert isinstance(translated, DuplicateRecordError)
    assert translated.constraint is None
