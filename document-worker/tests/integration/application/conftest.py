"""Прикладные сервисы проверяются на настоящих репозиториях.

Фейк единицы работы пришлось бы учить семантике `ON CONFLICT`, guard-UPDATE и
отката — то есть повторять то, что уже проверено на PostgreSQL. Расхождение
такого фейка с оригиналом даёт зелёные тесты при красном бое.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from document_worker.application.config import ProcessingConfig, SourceConfig
from document_worker.domain.value_objects.identifiers import DocumentId
from document_worker.domain.value_objects.versioning import PipelineVersion
from document_worker.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork
from tests.fakes.system import FixedClock, SequentialIdGenerator

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from document_worker.application.ports.unit_of_work import (
        UnitOfWork,
        UnitOfWorkFactory,
    )

NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
PIPELINE_VERSION = PipelineVersion(1, 0, 0)
CONSUMER = "worker-1"


@pytest.fixture
def document_id() -> DocumentId:
    """Идентификатор документа, о котором идёт речь в тесте."""
    return DocumentId(uuid.uuid4())


@pytest.fixture
def clock() -> FixedClock:
    """Часы, которые двигает сам тест."""
    return FixedClock(NOW)


@pytest.fixture
def ids() -> SequentialIdGenerator:
    """Предсказуемые идентификаторы."""
    return SequentialIdGenerator()


@pytest.fixture
def config() -> ProcessingConfig:
    """Конфигурация обработки с короткими пределами."""
    return ProcessingConfig(
        pipeline_version=PIPELINE_VERSION,
        consumer_name=CONSUMER,
        document_timeout_s=60.0,
        claim_lease_s=900,
        source=SourceConfig(max_file_size_bytes=1024 * 1024, download_timeout_s=5.0),
    )


@pytest.fixture
def uow_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> UnitOfWorkFactory:
    """Фабрика единиц работы поверх откатываемой транзакции теста.

    Таймаут выражения задаётся движком, а тестовая транзакция уже открыта,
    поэтому здесь он не применяется.
    """

    def factory(
        *,
        statement_timeout_ms: int,
        read_only: bool = False,
    ) -> AbstractAsyncContextManager[UnitOfWork]:
        del statement_timeout_ms, read_only
        return SqlAlchemyUnitOfWork(session_factory)

    return factory
