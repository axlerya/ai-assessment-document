"""Векторы: ширина плотного и предел разреженного заданы хранилищем."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_worker.domain.constants import (
    DENSE_DIMENSIONS,
    SPARSE_TOP_K,
    SPARSE_VOCABULARY_SIZE,
)
from ai_worker.domain.errors import InvalidVector
from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector

pytestmark = pytest.mark.unit


def _dense(length: int = DENSE_DIMENSIONS) -> tuple[float, ...]:
    return tuple(0.01 for _ in range(length))


def test_dense_vector_keeps_its_values() -> None:
    values = _dense()

    assert DenseVector(values).values == values


@pytest.mark.parametrize("length", [0, 1, DENSE_DIMENSIONS - 1, DENSE_DIMENSIONS + 1])
def test_dense_vector_rejects_wrong_dimension(length: int) -> None:
    # Ширина колонки `vector(1024)` задана миграцией: вектор другой длины не
    # сохранится, и узнать об этом на записи, а не на построении, значит
    # потерять весь прогон индексации документа.
    with pytest.raises(InvalidVector):
        DenseVector(_dense(length))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_dense_vector_rejects_nan_and_inf(bad: float) -> None:
    # NaN внутри вектора делает любое расстояние NaN, и поиск молча возвращает
    # мусор в произвольном порядке.
    values = (*_dense(DENSE_DIMENSIONS - 1), bad)

    with pytest.raises(InvalidVector):
        DenseVector(values)


def test_dense_vectors_with_the_same_values_are_the_same_value() -> None:
    assert DenseVector(_dense()) == DenseVector(_dense())


def test_sparse_vector_is_stored_in_ascending_index_order() -> None:
    # Литерал `sparsevec` требует возрастания индексов; канонический порядок
    # здесь избавляет адаптер от собственной сортировки.
    vector = SparseVector.pruned({7: 0.5, 2: 0.9, 5: 0.1})

    assert [index for index, _ in vector.weights] == [2, 5, 7]


def test_sparse_vector_is_pruned_to_the_limit() -> None:
    weights = {index: float(index + 1) for index in range(SPARSE_TOP_K + 250)}

    vector = SparseVector.pruned(weights)

    assert len(vector.weights) == SPARSE_TOP_K


def test_pruning_keeps_the_heaviest_weights() -> None:
    weights = {index: float(index) for index in range(1, SPARSE_TOP_K + 11)}

    vector = SparseVector.pruned(weights)

    kept = {index for index, _ in vector.weights}
    assert min(kept) == 11
    assert max(kept) == SPARSE_TOP_K + 10


def test_pruning_is_stable_for_equal_weights() -> None:
    # При равных весах порядок обязан задаваться индексом: иначе один и тот же
    # чанк давал бы разные векторы между прогонами, и оценка качества
    # перестала бы быть воспроизводимой.
    weights = dict.fromkeys(range(SPARSE_TOP_K + 50), 0.5)

    first = SparseVector.pruned(weights)
    second = SparseVector.pruned(dict(reversed(list(weights.items()))))

    assert first == second
    assert [index for index, _ in first.weights] == list(range(SPARSE_TOP_K))


def test_sparse_vector_below_the_limit_is_kept_whole() -> None:
    weights = dict.fromkeys(range(12), 0.3)

    assert len(SparseVector.pruned(weights).weights) == 12


def test_empty_sparse_vector_is_rejected() -> None:
    # Чанк без единого токена не существует: его текст не бывает пустым.
    with pytest.raises(InvalidVector):
        SparseVector.pruned({})


@pytest.mark.parametrize("index", [-1, SPARSE_VOCABULARY_SIZE])
def test_index_outside_the_vocabulary_is_rejected(index: int) -> None:
    with pytest.raises(InvalidVector):
        SparseVector.pruned({index: 0.5})


def test_vector_longer_than_the_limit_is_rejected_on_construction() -> None:
    # Обрезка — не единственный вход: строку читает и маппер из базы, и вектор
    # сверх предела означал бы индекс, который отказался бы строиться.
    too_many = tuple((index, 0.5) for index in range(SPARSE_TOP_K + 1))

    with pytest.raises(InvalidVector):
        SparseVector(too_many)


def test_unordered_weights_are_rejected_on_construction() -> None:
    # Порядок — часть представления: литерал `sparsevec` требует возрастания
    # индексов, и нарушение вскрылось бы уже на записи в базу.
    with pytest.raises(InvalidVector):
        SparseVector(((5, 0.9), (2, 0.4)))


def test_repeated_index_is_rejected_on_construction() -> None:
    with pytest.raises(InvalidVector):
        SparseVector(((2, 0.9), (2, 0.4)))


@pytest.mark.parametrize("weight", [0.0, -0.5, math.nan, math.inf])
def test_non_positive_and_broken_weights_are_rejected(weight: float) -> None:
    # Разреженный выход модели даёт только положительные веса; ноль означал бы
    # токен, которого в чанке нет, и раздувал бы вектор до предела впустую.
    with pytest.raises(InvalidVector):
        SparseVector.pruned({4: weight})


_INDEXES = st.integers(min_value=0, max_value=SPARSE_VOCABULARY_SIZE - 1)
_WEIGHTS = st.floats(
    min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False
)


@given(st.dictionaries(_INDEXES, _WEIGHTS, min_size=1, max_size=64))
@settings(max_examples=50, deadline=None)
def test_pruning_never_exceeds_the_limit(weights: dict[int, float]) -> None:
    assert len(SparseVector.pruned(weights).weights) <= SPARSE_TOP_K


@given(st.dictionaries(_INDEXES, _WEIGHTS, min_size=1, max_size=64))
@settings(max_examples=50, deadline=None)
def test_pruning_does_not_depend_on_input_order(weights: dict[int, float]) -> None:
    shuffled = dict(sorted(weights.items(), key=lambda pair: -pair[0]))

    assert SparseVector.pruned(weights) == SparseVector.pruned(shuffled)


@given(st.dictionaries(_INDEXES, _WEIGHTS, min_size=1, max_size=64))
@settings(max_examples=50, deadline=None)
def test_result_is_always_sorted_by_index(weights: dict[int, float]) -> None:
    indexes = [index for index, _ in SparseVector.pruned(weights).weights]

    assert indexes == sorted(indexes)
    assert len(set(indexes)) == len(indexes)
