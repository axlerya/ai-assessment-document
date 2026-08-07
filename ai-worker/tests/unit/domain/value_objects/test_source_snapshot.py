"""Снимок источника: что именно произвёл document-worker."""

from __future__ import annotations

import pytest

from ai_worker.domain.value_objects.enums import SourceStatus
from ai_worker.domain.value_objects.source_snapshot import SourceSnapshot
from ai_worker.domain.value_objects.versioning import ChunkingVersion, PipelineVersion

pytestmark = pytest.mark.unit


def _snapshot(status: SourceStatus) -> SourceSnapshot:
    return SourceSnapshot(
        pipeline_version=PipelineVersion(1, 0, 0),
        chunking_version=ChunkingVersion(1, 0, 0),
        status=status,
    )


def test_fully_processed_document_is_complete() -> None:
    assert _snapshot(SourceStatus.PROCESSED).is_complete


def test_partially_processed_document_is_indexed_but_marked() -> None:
    # Выбросить его значило бы потерять из корпуса любой скан с одной
    # нечитаемой строкой — то есть основной сценарий ТЗ.
    snapshot = _snapshot(SourceStatus.PARTIALLY_PROCESSED)

    assert not snapshot.is_complete


def test_snapshot_keeps_both_versions_of_its_source() -> None:
    # Версия чанкования выбирает корпус, версия обработки объясняет качество:
    # порознь ни одна не отвечает на вопрос «что мы индексируем».
    snapshot = _snapshot(SourceStatus.PROCESSED)

    assert snapshot.chunking_version == ChunkingVersion(1, 0, 0)
    assert snapshot.pipeline_version == PipelineVersion(1, 0, 0)
