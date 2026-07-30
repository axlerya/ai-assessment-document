"""Тесты версий пайплайна и чанкования."""

from __future__ import annotations

import pytest

from document_worker.domain.errors import (
    InvalidChunkingVersion,
    InvalidPipelineVersion,
)
from document_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    PipelineVersion,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.0.0", (1, 0, 0)),
        ("2.13.4", (2, 13, 4)),
        ("999.999.999", (999, 999, 999)),
    ],
)
def test_parse_reads_semver(raw: str, expected: tuple[int, int, int]) -> None:
    version = PipelineVersion.parse(raw)

    assert (version.major, version.minor, version.patch) == expected


@pytest.mark.parametrize("raw", ["1.0.0", "2.13.4", "10.0.1"])
def test_str_roundtrips_through_parse(raw: str) -> None:
    assert str(PipelineVersion.parse(raw)) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "1.0",
        "1.0.0.0",
        "v1.0.0",
        "1.0.0-rc1",
        "proc-1.4.0",
        "01.0.0",
        "1..0",
        "",
        " 1.0.0",
    ],
)
def test_parse_rejects_non_semver(raw: str) -> None:
    with pytest.raises(InvalidPipelineVersion):
        PipelineVersion.parse(raw)


@pytest.mark.parametrize(
    ("major", "minor", "patch"),
    [(0, 0, 0), (1000, 0, 0), (1, 1000, 0), (1, 0, 1000), (1, -1, 0)],
)
def test_rejects_parts_outside_bounds(major: int, minor: int, patch: int) -> None:
    with pytest.raises(InvalidPipelineVersion):
        PipelineVersion(major, minor, patch)


def test_versions_of_same_major_are_compatible() -> None:
    assert PipelineVersion(2, 1, 0).is_compatible_with(PipelineVersion(2, 9, 3))


def test_versions_of_different_major_are_not_compatible() -> None:
    assert not PipelineVersion(2, 1, 0).is_compatible_with(PipelineVersion(3, 0, 0))


@pytest.mark.parametrize(
    ("newer", "older"),
    [
        ((2, 0, 0), (1, 9, 9)),
        ((1, 2, 0), (1, 1, 9)),
        ((1, 1, 2), (1, 1, 1)),
    ],
)
def test_is_newer_than_compares_parts_left_to_right(
    newer: tuple[int, int, int],
    older: tuple[int, int, int],
) -> None:
    assert PipelineVersion(*newer).is_newer_than(PipelineVersion(*older))
    assert not PipelineVersion(*older).is_newer_than(PipelineVersion(*newer))


def test_equal_version_is_not_newer_than_itself() -> None:
    version = PipelineVersion(1, 0, 0)

    assert not version.is_newer_than(PipelineVersion(1, 0, 0))


def test_versions_are_ordered() -> None:
    versions = [
        PipelineVersion(1, 2, 0),
        PipelineVersion(1, 0, 3),
        PipelineVersion(2, 0, 0),
    ]

    assert sorted(versions) == [
        PipelineVersion(1, 0, 3),
        PipelineVersion(1, 2, 0),
        PipelineVersion(2, 0, 0),
    ]


def test_chunking_version_is_not_equal_to_pipeline_version() -> None:
    # mypy тоже отвергает такое сравнение — это и есть проверяемое свойство.
    assert ChunkingVersion(1, 0, 0) != PipelineVersion(1, 0, 0)  # type: ignore[comparison-overlap]


def test_chunking_version_raises_its_own_error() -> None:
    with pytest.raises(InvalidChunkingVersion):
        ChunkingVersion.parse("1.0")


def test_version_is_immutable() -> None:
    version = PipelineVersion(1, 0, 0)

    with pytest.raises(AttributeError):
        version.major = 2  # type: ignore[misc]
