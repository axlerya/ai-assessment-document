"""Бюджет токенов и хэш параметров чанкования."""

from __future__ import annotations

import pytest

from document_worker.domain.chunking.policy import (
    CHUNKING_VERSION,
    DEFAULT_CHUNKING_POLICY,
    KNOWN_PARAMS_HASHES,
    ChunkingPolicy,
)
from document_worker.domain.constants import MAX_CHUNK_TOKENS
from document_worker.domain.errors import InvalidChunkingPolicy
from document_worker.domain.value_objects.versioning import ChunkingVersion
from tests.unit.domain.chunking.support import default_policy

pytestmark = pytest.mark.unit


def test_policy_has_no_default_values() -> None:
    # Единственный источник чисел — настройки; тихий дефолт разъезжается
    # с .env.example и вскрывается уже в бою.
    with pytest.raises(TypeError):
        ChunkingPolicy()  # type: ignore[call-arg]


def test_policy_rejects_max_tokens_above_entity_ceiling() -> None:
    # Иначе InvariantViolation сущности прилетит на первом плотном документе,
    # то есть в терминальной части обработки, а не на старте.
    with pytest.raises(InvalidChunkingPolicy):
        default_policy(max_tokens=MAX_CHUNK_TOKENS + 1)


def test_policy_rejects_overlap_not_smaller_than_min_tokens() -> None:
    with pytest.raises(InvalidChunkingPolicy):
        default_policy(overlap_tokens=DEFAULT_CHUNKING_POLICY.min_tokens)


def test_policy_rejects_target_above_max() -> None:
    with pytest.raises(InvalidChunkingPolicy):
        default_policy(target_tokens=900, max_tokens=800)


def test_policy_rejects_non_positive_minimum() -> None:
    with pytest.raises(InvalidChunkingPolicy):
        default_policy(min_tokens=0, overlap_tokens=0)


def test_policy_rejects_empty_encoding() -> None:
    with pytest.raises(InvalidChunkingPolicy):
        default_policy(encoding="")


def test_params_hash_changes_when_any_policy_field_changes() -> None:
    changed = default_policy(overlap_tokens=DEFAULT_CHUNKING_POLICY.overlap_tokens + 1)

    assert changed.params_hash() != DEFAULT_CHUNKING_POLICY.params_hash()


def test_params_hash_matches_recorded_value() -> None:
    # Захардкоженный хэш падает при любом изменении параметров, сделанном без
    # осознанного инкремента версии чанкования.
    assert (
        DEFAULT_CHUNKING_POLICY.params_hash()
        == KNOWN_PARAMS_HASHES[str(CHUNKING_VERSION)]
    )


def test_default_parameters_are_registered() -> None:
    DEFAULT_CHUNKING_POLICY.ensure_registered()


def test_changed_budget_without_new_version_is_refused() -> None:
    # Правка бюджета через .env без инкремента версии сложила бы чанки разных
    # параметров в один namespace chunking_version — незаметно и необратимо.
    with pytest.raises(InvalidChunkingPolicy, match="параметры"):
        default_policy(target_tokens=350).ensure_registered()


def test_unknown_version_is_refused() -> None:
    with pytest.raises(InvalidChunkingPolicy, match="версия"):
        default_policy(version=ChunkingVersion(9, 0, 0)).ensure_registered()
