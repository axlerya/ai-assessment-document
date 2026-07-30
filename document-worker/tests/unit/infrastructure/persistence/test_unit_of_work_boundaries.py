"""Границы единицы работы на подставной сессии.

Настоящий PostgreSQL уронить на commit по заказу нельзя, а транслировать ошибку
именно там обязательно: разрыв соединения приходит ровно в этот момент.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.exc import DBAPIError

from document_worker.application.errors import SerializationConflictError
from document_worker.infrastructure.persistence.unit_of_work import (
    NestedUnitOfWorkError,
    SqlAlchemyUnitOfWork,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit


class _DriverError(Exception):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(f"driver error {sqlstate}")
        self.sqlstate = sqlstate


def _dbapi_error(sqlstate: str) -> DBAPIError:
    error = DBAPIError("COMMIT", None, _DriverError(sqlstate))
    error.connection_invalidated = False
    return error


class _FakeSavepoint:
    """Точка отката, которую база могла погасить сама."""

    def __init__(self, *, active: bool) -> None:
        self.is_active = active
        self.committed = 0
        self.rolled_back = 0

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        self.rolled_back += 1


class _FakeSession:
    """Сессия, которая падает на указанной операции."""

    def __init__(self, failing: str, *, savepoint_active: bool = True) -> None:
        self._failing = failing
        self.rolled_back = 0
        self.closed = 0
        self.savepoint = _FakeSavepoint(active=savepoint_active)

    async def begin(self) -> object:
        return object()

    async def begin_nested(self) -> _FakeSavepoint:
        return self.savepoint

    async def commit(self) -> None:
        self._maybe_fail("commit")

    async def flush(self) -> None:
        self._maybe_fail("flush")

    async def rollback(self) -> None:
        self.rolled_back += 1

    async def close(self) -> None:
        self.closed += 1

    def _maybe_fail(self, operation: str) -> None:
        if operation == self._failing:
            raise _dbapi_error("40001")


def _factory(session: _FakeSession) -> Callable[[], Any]:
    return lambda: session


async def test_commit_translates_driver_error() -> None:
    session = _FakeSession("commit")
    unit = SqlAlchemyUnitOfWork(_factory(session))  # type: ignore[arg-type]

    async with unit:
        with pytest.raises(SerializationConflictError):
            await unit.commit()

    assert session.rolled_back >= 1


async def test_flush_translates_driver_error() -> None:
    session = _FakeSession("flush")
    unit = SqlAlchemyUnitOfWork(_factory(session))  # type: ignore[arg-type]

    async with unit:
        with pytest.raises(SerializationConflictError):
            await unit.flush()


async def test_session_is_closed_even_when_the_block_fails() -> None:
    session = _FakeSession("never")
    unit = SqlAlchemyUnitOfWork(_factory(session))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        await _fails_inside(unit)

    assert session.closed == 1
    assert session.rolled_back == 1


async def test_successful_savepoint_is_released() -> None:
    session = _FakeSession("never")
    unit = SqlAlchemyUnitOfWork(_factory(session))  # type: ignore[arg-type]

    async with unit, unit.savepoint():
        pass

    assert session.savepoint.committed == 1


async def test_savepoint_already_closed_by_the_database_is_not_rolled_back() -> None:
    # Ошибка внутри точки отката могла погасить её саму: повторный откат
    # обратился бы к несуществующему savepoint.
    session = _FakeSession("never", savepoint_active=False)
    unit = SqlAlchemyUnitOfWork(_factory(session))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        await _fails_inside_savepoint(unit)

    assert session.savepoint.rolled_back == 0


async def test_savepoint_outside_the_block_is_an_error() -> None:
    unit = SqlAlchemyUnitOfWork(_factory(_FakeSession("never")))  # type: ignore[arg-type]

    with pytest.raises(NestedUnitOfWorkError):
        async with unit.savepoint():
            pass


async def _fails_inside(unit: SqlAlchemyUnitOfWork) -> None:
    async with unit:
        raise RuntimeError


async def _fails_inside_savepoint(unit: SqlAlchemyUnitOfWork) -> None:
    async with unit, unit.savepoint():
        raise RuntimeError
