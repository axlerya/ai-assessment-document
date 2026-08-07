"""Провайдер эмбеддингов: пачки, префиксы, нормализация и обрезка.

Прогон самой модели сюда не входит — он в `tests/slow`. Здесь проверяется то,
что провайдер делает вокруг модели и что обязано быть верным независимо от неё.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import math
import random
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from ai_worker.application.errors import EmbeddingBackendUnavailable
from ai_worker.domain.constants import DENSE_DIMENSIONS
from ai_worker.domain.embedding.policy import DEFAULT_EMBEDDING_POLICY
from ai_worker.infrastructure.embedding.bge_m3 import BgeM3EmbeddingProvider
from ai_worker.infrastructure.embedding.raw import RawEmbedding

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ai_worker.domain.embedding.policy import EmbeddingPolicy

pytestmark = pytest.mark.unit

PASSAGES = ("договор аренды", "акт приёма-передачи", "счёт на оплату", "приложение №1")


def _seeded(text: str) -> random.Random:
    # Воспроизводимость важнее криптостойкости: фейку нужен один и тот же
    # вектор на один и тот же текст.
    return random.Random(hashlib.sha256(text.encode("utf-8")).digest()[:8])  # noqa: S311


@dataclass(slots=True)
class FakeRuntime:
    """Модель, заменённая на воспроизводимый генератор.

    Считает вызовы и запомненные бюджеты: провайдер обязан резать вход на
    пачки и делить между ними один таймаут, а не выдавать каждой полный.
    """

    sparse_weights: int = 4
    dense_scale: float = 3.0
    delay_s: float = 0.0
    failure: Exception | None = None
    batches: list[tuple[str, ...]] = field(default_factory=list)
    budgets: list[float] = field(default_factory=list)

    async def encode(
        self,
        texts: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[RawEmbedding]:
        self.batches.append(tuple(texts))
        self.budgets.append(timeout_s)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.failure is not None:
            raise self.failure
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> RawEmbedding:
        source = _seeded(text)
        dense = tuple(
            source.uniform(-self.dense_scale, self.dense_scale)
            for _ in range(DENSE_DIMENSIONS)
        )
        sparse: Mapping[int, float] = {
            source.randrange(250_000): source.uniform(0.01, 1.0)
            for _ in range(self.sparse_weights)
        }
        return RawEmbedding(dense=dense, sparse=sparse)


def _provider(
    runtime: FakeRuntime,
    *,
    batch_size: int = 2,
    policy: EmbeddingPolicy | None = None,
) -> BgeM3EmbeddingProvider:
    return BgeM3EmbeddingProvider(
        policy=policy or DEFAULT_EMBEDDING_POLICY,
        runtime=runtime,
        batch_size=batch_size,
    )


async def test_empty_input_does_not_touch_the_model() -> None:
    runtime = FakeRuntime()

    assert await _provider(runtime).embed_passages([], timeout_s=1.0) == ()
    assert runtime.batches == []


async def test_passages_are_encoded_in_batches_of_the_configured_size() -> None:
    # Пачка целиком лежит в памяти рабочего процесса: без разбиения документ на
    # тысячу чанков уходит в модель одним куском.
    runtime = FakeRuntime()

    await _provider(runtime, batch_size=3).embed_passages(PASSAGES, timeout_s=10.0)

    assert [len(batch) for batch in runtime.batches] == [3, 1]


async def test_batch_size_does_not_change_the_vectors() -> None:
    # Размер пачки — параметр производительности, поэтому в хэш политики он не
    # входит; значит, на результат он влиять не имеет права.
    one = await _provider(FakeRuntime(), batch_size=1).embed_passages(
        PASSAGES, timeout_s=10.0
    )
    many = await _provider(FakeRuntime(), batch_size=4).embed_passages(
        PASSAGES, timeout_s=10.0
    )

    assert one == many


async def test_order_of_results_follows_the_order_of_texts() -> None:
    runtime = FakeRuntime()

    vectors = await _provider(runtime, batch_size=2).embed_passages(
        PASSAGES, timeout_s=10.0
    )
    single = await _provider(FakeRuntime(), batch_size=2).embed_passages(
        [PASSAGES[2]], timeout_s=10.0
    )

    assert len(vectors) == len(PASSAGES)
    assert vectors[2] == single[0]


async def test_passage_prefix_is_applied() -> None:
    runtime = FakeRuntime()
    policy = dataclasses.replace(DEFAULT_EMBEDDING_POLICY, passage_prefix="документ: ")

    await _provider(runtime, policy=policy).embed_passages(["акт"], timeout_s=1.0)

    assert runtime.batches == [("документ: акт",)]


async def test_query_prefix_is_applied() -> None:
    # Запрос и фрагмент — разные вызовы именно потому, что префиксы у них
    # разные; общий метод скрыл бы это различие.
    runtime = FakeRuntime()
    policy = dataclasses.replace(DEFAULT_EMBEDDING_POLICY, query_prefix="вопрос: ")

    await _provider(runtime, policy=policy).embed_query("кто арендатор", timeout_s=1.0)

    assert runtime.batches == [("вопрос: кто арендатор",)]


async def test_dense_vector_has_declared_dimension() -> None:
    dense, _ = await _provider(FakeRuntime()).embed_query("запрос", timeout_s=1.0)

    assert len(dense.values) == DENSE_DIMENSIONS


async def test_dense_vector_is_normalized_when_policy_says_so() -> None:
    policy = dataclasses.replace(DEFAULT_EMBEDDING_POLICY, normalize=True)

    dense, _ = await _provider(FakeRuntime(), policy=policy).embed_query(
        "запрос", timeout_s=1.0
    )

    assert math.isclose(math.hypot(*dense.values), 1.0, rel_tol=1e-9)


async def test_dense_vector_is_left_as_is_when_policy_says_not_to() -> None:
    policy = dataclasses.replace(DEFAULT_EMBEDDING_POLICY, normalize=False)

    dense, _ = await _provider(FakeRuntime(), policy=policy).embed_query(
        "запрос", timeout_s=1.0
    )

    assert not math.isclose(math.hypot(*dense.values), 1.0, rel_tol=1e-3)


async def test_sparse_weights_are_pruned_to_the_limit() -> None:
    policy = dataclasses.replace(DEFAULT_EMBEDDING_POLICY, sparse_top_k=3)
    runtime = FakeRuntime(sparse_weights=32)

    _, sparse = await _provider(runtime, policy=policy).embed_query(
        "запрос", timeout_s=1.0
    )

    assert len(sparse.weights) == 3


async def test_timeout_budget_is_shared_between_batches() -> None:
    # Полный таймаут на каждую пачку означал бы, что заявленный предел на вызов
    # превышается во столько раз, сколько пачек.
    runtime = FakeRuntime(delay_s=0.05)

    await _provider(runtime, batch_size=1).embed_passages(PASSAGES[:2], timeout_s=5.0)

    assert runtime.budgets[1] < runtime.budgets[0]


async def test_exhausted_budget_stops_before_the_next_batch() -> None:
    runtime = FakeRuntime(delay_s=0.05)

    with pytest.raises(EmbeddingBackendUnavailable, match="таймаут"):
        await _provider(runtime, batch_size=1).embed_passages(PASSAGES, timeout_s=0.06)

    assert len(runtime.batches) < len(PASSAGES)


async def test_model_timeout_becomes_a_transient_error() -> None:
    runtime = FakeRuntime(failure=TimeoutError())

    with pytest.raises(EmbeddingBackendUnavailable, match="таймаут"):
        await _provider(runtime).embed_query("запрос", timeout_s=1.0)


async def test_broken_worker_becomes_a_transient_error() -> None:
    # Рабочий процесс, убитый нехваткой памяти, — это повтор, а не разбор.
    runtime = FakeRuntime(failure=BrokenProcessPool("worker died"))

    with pytest.raises(EmbeddingBackendUnavailable, match="процесс"):
        await _provider(runtime).embed_query("запрос", timeout_s=1.0)
