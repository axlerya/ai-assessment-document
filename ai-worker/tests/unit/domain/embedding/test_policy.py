"""Политика версий эмбеддингов: чем считали и под какой версией это лежит."""

from __future__ import annotations

import dataclasses

import pytest

from ai_worker.domain.constants import DENSE_DIMENSIONS, SPARSE_TOP_K
from ai_worker.domain.embedding.policy import (
    DEFAULT_EMBEDDING_POLICY,
    EMBEDDING_VERSION,
    KNOWN_EMBEDDING_HASHES,
    EmbeddingPolicy,
)
from ai_worker.domain.errors import InvalidEmbeddingPolicy
from ai_worker.domain.value_objects.versioning import EmbeddingVersion

pytestmark = pytest.mark.unit


def test_declared_policy_is_registered() -> None:
    # Реестр обязан описывать ту политику, с которой сервис уходит в бой.
    DEFAULT_EMBEDDING_POLICY.ensure_registered()


def test_declared_version_is_the_one_in_the_registry() -> None:
    assert str(EMBEDDING_VERSION) in KNOWN_EMBEDDING_HASHES


def test_hash_does_not_depend_on_the_run() -> None:
    assert DEFAULT_EMBEDDING_POLICY.params_hash() == (
        DEFAULT_EMBEDDING_POLICY.params_hash()
    )


def test_policy_hash_change_without_version_bump_is_rejected() -> None:
    # Ровно тот случай, ради которого реестр и заведён: правка предела токенов
    # через `.env` сложила бы векторы разной геометрии в один namespace.
    changed = dataclasses.replace(DEFAULT_EMBEDDING_POLICY, max_input_tokens=512)

    with pytest.raises(InvalidEmbeddingPolicy, match="не соответствуют"):
        changed.ensure_registered()


def test_unknown_version_is_rejected() -> None:
    unknown = dataclasses.replace(
        DEFAULT_EMBEDDING_POLICY, version=EmbeddingVersion(9, 0, 0)
    )

    with pytest.raises(InvalidEmbeddingPolicy, match="реестре"):
        unknown.ensure_registered()


_CHANGED_PARAMETERS = [
    pytest.param(
        dataclasses.replace(
            DEFAULT_EMBEDDING_POLICY, version=EmbeddingVersion(2, 0, 0)
        ),
        id="version",
    ),
    pytest.param(
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, model_name="BAAI/bge-m3-other"),
        id="model_name",
    ),
    pytest.param(
        dataclasses.replace(
            DEFAULT_EMBEDDING_POLICY, normalize=not DEFAULT_EMBEDDING_POLICY.normalize
        ),
        id="normalize",
    ),
    pytest.param(
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, max_input_tokens=512),
        id="max_input_tokens",
    ),
    pytest.param(
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, sparse_top_k=500),
        id="sparse_top_k",
    ),
    pytest.param(
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, query_prefix="запрос: "),
        id="query_prefix",
    ),
    pytest.param(
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, passage_prefix="документ: "),
        id="passage_prefix",
    ),
]


@pytest.mark.parametrize("changed", _CHANGED_PARAMETERS)
def test_every_parameter_that_changes_the_vector_changes_the_hash(
    changed: EmbeddingPolicy,
) -> None:
    # Параметр вне хэша — это тихая смена геометрии под старой версией.
    assert changed.params_hash() != DEFAULT_EMBEDDING_POLICY.params_hash()


def test_policy_carries_the_identity_of_its_vectors() -> None:
    identity = DEFAULT_EMBEDDING_POLICY.identity

    assert identity.version == DEFAULT_EMBEDDING_POLICY.version
    assert identity.model_name == DEFAULT_EMBEDDING_POLICY.model_name


def test_dimensions_must_match_the_column() -> None:
    with pytest.raises(InvalidEmbeddingPolicy, match="ширина"):
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, dimensions=DENSE_DIMENSIONS - 1)


def test_sparse_limit_above_the_index_ceiling_is_rejected() -> None:
    # Иначе `CREATE INDEX` падает не на записи вектора, а на построении индекса.
    with pytest.raises(InvalidEmbeddingPolicy, match="разреж"):
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, sparse_top_k=SPARSE_TOP_K + 1)


def test_empty_sparse_limit_is_rejected() -> None:
    with pytest.raises(InvalidEmbeddingPolicy, match="разреж"):
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, sparse_top_k=0)


def test_non_positive_input_limit_is_rejected() -> None:
    with pytest.raises(InvalidEmbeddingPolicy, match="токен"):
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, max_input_tokens=0)


def test_model_name_is_required() -> None:
    with pytest.raises(InvalidEmbeddingPolicy, match="модел"):
        dataclasses.replace(DEFAULT_EMBEDDING_POLICY, model_name="  ")


def test_policy_has_no_defaults() -> None:
    # Пропуск поля обязан быть ошибкой конструктора, видимой mypy, а не тихим
    # значением, разъезжающимся с настройками сервиса.
    with_defaults = [
        field.name
        for field in dataclasses.fields(EmbeddingPolicy)
        if field.default is not dataclasses.MISSING
        or field.default_factory is not dataclasses.MISSING
    ]

    assert not with_defaults, f"поля со значением по умолчанию: {with_defaults}"
