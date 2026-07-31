"""Движок базы, фабрика сессий и фабрика единиц работы.

Переполнения пула нет: обработка упирается в процессор, и лишние соединения
сверх заявленного размера означали бы, что документов в работе больше, чем
сервис объявил. Пусть лучше ждёт тот, кто пришёл за соединением, чем база
получит нагрузку, которую никто не планировал.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from document_worker.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncEngine

    from document_worker.application.ports.unit_of_work import (
        UnitOfWork,
        UnitOfWorkFactory,
    )


def build_engine(
    dsn: str,
    *,
    pool_size: int,
    max_overflow: int,
    pool_timeout_s: float,
) -> AsyncEngine:
    """Создаёт движок с заданным пулом соединений."""
    return create_async_engine(
        dsn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout_s,
        pool_pre_ping=True,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Создаёт фабрику сессий поверх движка."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


def build_unit_of_work_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> UnitOfWorkFactory:
    """Создаёт фабрику единиц работы с таймаутом на транзакцию."""

    def factory(
        *,
        statement_timeout_ms: int,
        read_only: bool = False,
    ) -> AbstractAsyncContextManager[UnitOfWork]:
        return SqlAlchemyUnitOfWork(
            session_factory,
            statement_timeout_ms=statement_timeout_ms,
            read_only=read_only,
        )

    return factory
