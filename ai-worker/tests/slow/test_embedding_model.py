"""Контракт настоящей bge-m3: воспроизводимость и пригодность векторов.

Эти проверки нельзя заменить фейком: они говорят о самой модели, а не о коде
вокруг неё. Ошибка здесь означает, что индекс наполнен не тем, чем ищут.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from ai_worker.domain.constants import DENSE_DIMENSIONS, SPARSE_VOCABULARY_SIZE
from ai_worker.domain.embedding.policy import DEFAULT_EMBEDDING_POLICY
from ai_worker.infrastructure.embedding.bge_m3 import BgeM3EmbeddingProvider

if TYPE_CHECKING:
    from ai_worker.infrastructure.embedding.runtime import PooledEncoderRuntime

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

TIMEOUT_S = 600.0

CONTRACT = "Арендатор обязуется вносить плату не позднее пятого числа месяца."
RESTATED = "Плата за аренду вносится арендатором до пятого числа каждого месяца."
UNRELATED = "Погода в Мурманске в ноябре редко бывает солнечной."


def _provider(
    runtime: PooledEncoderRuntime, *, batch_size: int = 2
) -> BgeM3EmbeddingProvider:
    return BgeM3EmbeddingProvider(
        policy=DEFAULT_EMBEDDING_POLICY,
        runtime=runtime,
        batch_size=batch_size,
    )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


async def test_same_text_gives_the_same_vector(runtime: PooledEncoderRuntime) -> None:
    # Без этого свойства ни идемпотентная индексация, ни сравнение прогонов
    # оценки не имеют смысла.
    first = await _provider(runtime).embed_passages([CONTRACT], timeout_s=TIMEOUT_S)
    second = await _provider(runtime).embed_passages([CONTRACT], timeout_s=TIMEOUT_S)

    assert first == second


async def test_batching_does_not_change_vectors(runtime: PooledEncoderRuntime) -> None:
    # Дополнение пачки паддингом не должно менять чужой вектор, иначе документ
    # индексируется по-разному в зависимости от того, сколько чанков осталось.
    texts = [CONTRACT, RESTATED, UNRELATED]
    alone = await _provider(runtime, batch_size=1).embed_passages(
        texts, timeout_s=TIMEOUT_S
    )
    together = await _provider(runtime, batch_size=3).embed_passages(
        texts, timeout_s=TIMEOUT_S
    )

    for one, many in zip(alone, together, strict=True):
        assert (
            max(abs(a - b) for a, b in zip(one[0].values, many[0].values, strict=True))
            < 1e-4
        )


async def test_dense_vector_has_declared_dimension(
    runtime: PooledEncoderRuntime,
) -> None:
    dense, _ = await _provider(runtime).embed_query(CONTRACT, timeout_s=TIMEOUT_S)

    assert len(dense.values) == DENSE_DIMENSIONS
    assert math.isclose(math.hypot(*dense.values), 1.0, rel_tol=1e-5)


async def test_sparse_weights_stay_inside_the_vocabulary(
    runtime: PooledEncoderRuntime,
) -> None:
    _, sparse = await _provider(runtime).embed_query(CONTRACT, timeout_s=TIMEOUT_S)

    assert sparse.weights
    assert all(0 <= index < SPARSE_VOCABULARY_SIZE for index, _ in sparse.weights)
    assert all(weight > 0 for _, weight in sparse.weights)


async def test_restatement_is_closer_than_an_unrelated_text(
    runtime: PooledEncoderRuntime,
) -> None:
    # Проверка того, что в индекс кладётся смысл, а не шум: без неё неверно
    # собранный пулинг прошёл бы все остальные тесты.
    vectors = await _provider(runtime).embed_passages(
        [CONTRACT, RESTATED, UNRELATED], timeout_s=TIMEOUT_S
    )
    source, restated, unrelated = (dense.values for dense, _ in vectors)

    assert _cosine(source, restated) > _cosine(source, unrelated)


async def test_text_longer_than_the_limit_is_truncated_not_rejected(
    runtime: PooledEncoderRuntime,
) -> None:
    # Чанк выше предела токенов — штатный случай плотной таблицы, а не отказ.
    long_text = " ".join([CONTRACT] * 400)

    dense, sparse = await _provider(runtime).embed_query(long_text, timeout_s=TIMEOUT_S)

    assert len(dense.values) == DENSE_DIMENSIONS
    assert len(sparse.weights) <= DEFAULT_EMBEDDING_POLICY.sparse_top_k
