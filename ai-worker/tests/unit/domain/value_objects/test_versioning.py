"""Версии: четыре величины, четыре типа, один формат."""

from __future__ import annotations

import pytest

from ai_worker.domain.errors import (
    InvalidChunkingVersion,
    InvalidEmbeddingVersion,
    InvalidPipelineVersion,
    InvalidPromptVersion,
    InvalidVersion,
)
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PipelineVersion,
    PromptVersion,
)

pytestmark = pytest.mark.unit

# Свои — EmbeddingVersion и PromptVersion; чужие — ChunkingVersion и
# PipelineVersion: они приходят от document-worker и только разбираются.
ALL_VERSIONS = (EmbeddingVersion, PromptVersion, ChunkingVersion, PipelineVersion)


@pytest.mark.parametrize("version_class", ALL_VERSIONS)
def test_version_parses_and_prints_back(version_class: type) -> None:
    assert str(version_class.parse("2.11.3")) == "2.11.3"


@pytest.mark.parametrize("version_class", ALL_VERSIONS)
@pytest.mark.parametrize(
    "raw",
    [
        "01.0.0",  # ведущий ноль даёт вторую запись той же версии
        "1.0",
        "1.0.0.0",
        "v1.0.0",
        "1.0.0-rc1",
        "proc-1.4.0",
        "",
        " 1.0.0",
    ],
)
def test_only_bare_semver_is_accepted(version_class: type, raw: str) -> None:
    # На версии держится вся идемпотентность: две записи одной величины
    # означают два namespace вместо одного.
    with pytest.raises(InvalidVersion):
        version_class.parse(raw)


@pytest.mark.parametrize("version_class", ALL_VERSIONS)
def test_zero_major_is_rejected(version_class: type) -> None:
    with pytest.raises(InvalidVersion):
        version_class(0, 1, 0)


@pytest.mark.parametrize(
    ("version_class", "error_class"),
    [
        (EmbeddingVersion, InvalidEmbeddingVersion),
        (PromptVersion, InvalidPromptVersion),
        (ChunkingVersion, InvalidChunkingVersion),
        (PipelineVersion, InvalidPipelineVersion),
    ],
)
def test_each_version_reports_its_own_error(
    version_class: type,
    error_class: type[InvalidVersion],
) -> None:
    # По коду ошибки видно, какая именно версия испорчена: одна общая ошибка
    # заставляла бы разбирать текст сообщения.
    with pytest.raises(error_class):
        version_class.parse("нет")


def test_version_types_are_not_interchangeable() -> None:
    assert EmbeddingVersion(1, 0, 0) != PromptVersion(1, 0, 0)


def test_versions_are_ordered() -> None:
    assert EmbeddingVersion(1, 2, 0) < EmbeddingVersion(1, 10, 0)
    assert EmbeddingVersion(2, 0, 0) > EmbeddingVersion(1, 99, 99)


def test_newer_than_compares_all_three_parts() -> None:
    assert EmbeddingVersion(1, 0, 1).is_newer_than(EmbeddingVersion(1, 0, 0))
    assert not EmbeddingVersion(1, 0, 0).is_newer_than(EmbeddingVersion(1, 0, 0))


def test_compatibility_is_decided_by_the_major_part() -> None:
    # Смена мажорной части открывает новый namespace: старые эмбеддинги
    # остаются, переиндексация идёт рядом.
    assert EmbeddingVersion(1, 4, 2).is_compatible_with(EmbeddingVersion(1, 0, 0))
    assert not EmbeddingVersion(2, 0, 0).is_compatible_with(EmbeddingVersion(1, 0, 0))


def test_highest_of_picks_the_greatest_version() -> None:
    # Так выбирается версия чанкования документа, когда их несколько
    # (ADR-0008): берётся наибольшая, а не первая попавшаяся.
    versions = (
        ChunkingVersion(1, 0, 0),
        ChunkingVersion(2, 1, 0),
        ChunkingVersion(1, 9, 9),
    )

    assert ChunkingVersion.highest_of(versions) == ChunkingVersion(2, 1, 0)


def test_highest_of_rejects_an_empty_choice() -> None:
    with pytest.raises(InvalidVersion):
        ChunkingVersion.highest_of(())
