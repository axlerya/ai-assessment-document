"""База с обеими схемами и обвязка сценария индексации.

Сценарию нужны и чужие таблицы (корпус чанков), и свои (индекс, эмбеддинги,
сообщения, outbox). Поэтому здесь своя база: в ней сначала накатываются
миграции document-worker, потом собственные — тот же порядок, что и при
развёртывании.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from alembic import command
from sqlalchemy.ext.asyncio import create_async_engine

from ai_worker.application.config import (
    ContextConfig,
    EmbeddingConfig,
    LlmConfig,
    OutboxConfig,
    ProcessingConfig,
    RerankConfig,
    RetrievalConfig,
)
from ai_worker.domain.constants import DENSE_DIMENSIONS
from ai_worker.domain.embedding.policy import DEFAULT_EMBEDDING_POLICY
from ai_worker.domain.errors import InvalidVector
from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector
from ai_worker.domain.value_objects.versioning import PromptVersion
from ai_worker.infrastructure.persistence.read_model.processed_chunks import (
    SqlAlchemyProcessedChunkReader,
)
from ai_worker.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork
from tests.conftest import (
    alembic_config,
    create_database,
    drop_database,
    dsn_for,
    unique_database_name,
)
from tests.support.foreign_schema import apply_foreign_schema, skip_unless_supported

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncConnection

    from ai_worker.domain.embedding.policy import EmbeddingPolicy

# Момент фиксирован: сценарий пишет отметки времени и сравнивает их между
# собой, и подменяемые часы — единственный способ проверить это тестом.
NOW = "2026-03-01T12:00:00+00:00"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def indexing_dsn(base_dsn: str) -> AsyncIterator[str]:
    """База с обеими схемами в порядке развёртывания."""
    skip_unless_supported()
    name = unique_database_name("ai_worker_indexing")
    await create_database(base_dsn, name)
    dsn = dsn_for(base_dsn, name)
    await asyncio.to_thread(apply_foreign_schema, dsn)
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")
    try:
        yield dsn
    finally:
        await drop_database(base_dsn, name)


@pytest_asyncio.fixture(loop_scope="session")
async def indexing_connection(indexing_dsn: str) -> AsyncIterator[AsyncConnection]:
    """Соединение во внешней транзакции: она откатывается после теста."""
    engine = create_async_engine(indexing_dsn)
    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                yield connection
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


@dataclass(slots=True)
class FrozenClock:
    """Часы, которые стоят: иначе отметки времени нечем сравнивать."""

    moment: datetime

    def now(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> None:
        from datetime import timedelta  # noqa: PLC0415 — только для теста

        self.moment = self.moment + timedelta(seconds=seconds)


@dataclass(slots=True)
class FakeEmbeddings:
    """Провайдер, заменённый на воспроизводимый генератор.

    Записывает, что и когда у него просили: сценарий обязан резать вход на
    пачки и не держать при этом открытой единицу работы.
    """

    connection: AsyncConnection | None = None
    unindexable: frozenset[str] = frozenset()
    batches: list[tuple[str, ...]] = field(default_factory=list)
    units_open: list[bool] = field(default_factory=list)

    async def embed_passages(
        self,
        texts: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[tuple[DenseVector, SparseVector]]:
        del timeout_s
        self.batches.append(tuple(texts))
        if self.connection is not None:
            self.units_open.append(self.connection.get_nested_transaction() is not None)
        return [self._one(text) for text in texts]

    async def embed_query(
        self,
        text: str,
        *,
        timeout_s: float,
    ) -> tuple[DenseVector, SparseVector]:  # pragma: no cover — не в этом сценарии
        del timeout_s
        return self._one(text)

    @property
    def embedded_texts(self) -> tuple[str, ...]:
        return tuple(text for batch in self.batches for text in batch)

    def _one(self, text: str) -> tuple[DenseVector, SparseVector]:
        if text in self.unindexable:
            # Так выглядит чанк, из которого модель не смогла построить
            # разреженное представление: одни служебные токены.
            raise InvalidVector("разреженный вектор пуст", context={})
        seed = uuid.uuid5(uuid.NAMESPACE_OID, text).int
        dense = tuple(
            ((seed >> (position % 64)) % 1000 + 1) / 1000.0
            for position in range(DENSE_DIMENSIONS)
        )
        sparse = {(seed >> shift) % 250_000: (shift + 1) / 10.0 for shift in range(4)}
        return DenseVector(dense), SparseVector.pruned(sparse)


def processing_config(
    *,
    policy: EmbeddingPolicy = DEFAULT_EMBEDDING_POLICY,
    batch_size: int = 2,
    consumer_name: str = "worker-1",
) -> ProcessingConfig:
    """Конфигурация сценария с подменяемыми пределами."""
    return ProcessingConfig(
        consumer_name=consumer_name,
        message_timeout_s=900.0,
        claim_lease_s=1800,
        embedding=EmbeddingConfig(
            policy=policy, batch_size=batch_size, timeout_s=120.0
        ),
        retrieval=RetrievalConfig(
            profile="hybrid-rrf-v1",
            top_k_dense=50,
            top_k_sparse=50,
            rrf_k=60,
            ef_search=100,
        ),
        rerank=RerankConfig(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=50,
            batch_size=8,
            timeout_s=120.0,
        ),
        context=ContextConfig(
            token_budget=8000, max_chunks=20, min_citable_confidence=0.6
        ),
        llm=LlmConfig(
            model="deepseek-ai/DeepSeek-V4-Flash",
            prompt_version=PromptVersion(1, 0, 0),
            timeout_s=120.0,
            max_output_tokens=4000,
        ),
        outbox=OutboxConfig(
            batch_size=100,
            poll_interval_s=0.5,
            lease_seconds=30,
            backoff_base_s=1.0,
            backoff_cap_s=300.0,
        ),
    )


@pytest.fixture
def reader(indexing_connection: AsyncConnection) -> SqlAlchemyProcessedChunkReader:
    """Чтение чужого корпуса тем же адаптером, что и в бою."""
    return SqlAlchemyProcessedChunkReader(indexing_connection)


@pytest.fixture
def uow_factory(
    indexing_connection: AsyncConnection,
) -> object:
    """Единица работы поверх соединения теста."""

    def factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork.on(indexing_connection)

    return factory
