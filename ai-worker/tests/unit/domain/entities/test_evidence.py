"""Доказательство: чанк вместе с тем, как он был найден."""

from __future__ import annotations

import pytest

from ai_worker.domain.entities.evidence import Evidence
from ai_worker.domain.errors import InvariantViolation
from ai_worker.domain.value_objects.enums import ExtractionMethod
from ai_worker.domain.value_objects.scores import RrfScore, Score
from tests.factories import make_chunk, make_quality

pytestmark = pytest.mark.unit


def _evidence(
    *,
    dense_rank: int | None = 1,
    sparse_rank: int | None = 4,
    rerank_score: Score | None = None,
    quality: object | None = None,
) -> Evidence:
    chunk = make_chunk(quality=quality) if quality is not None else make_chunk()
    return Evidence(
        chunk=chunk,
        dense_rank=dense_rank,
        dense_score=None if dense_rank is None else Score(0.81),
        sparse_rank=sparse_rank,
        sparse_score=None if sparse_rank is None else Score(6.4),
        rrf_score=RrfScore(0.031),
        rerank_score=rerank_score,
    )


def test_evidence_keeps_the_chunk_it_points_at() -> None:
    chunk = make_chunk()

    evidence = Evidence(
        chunk=chunk,
        dense_rank=1,
        dense_score=Score(0.9),
        sparse_rank=None,
        sparse_score=None,
        rrf_score=RrfScore(0.016),
        rerank_score=None,
    )

    assert evidence.chunk is chunk
    assert evidence.ref == chunk.ref


def test_evidence_must_come_from_at_least_one_branch() -> None:
    # Попадание, не найденное ни плотной, ни разреженной ветвью, означает
    # склейку по неверному ключу — то есть чужой чанк в выдаче.
    with pytest.raises(InvariantViolation):
        _evidence(dense_rank=None, sparse_rank=None)


@pytest.mark.parametrize("rank", [0, -1])
def test_ranks_start_at_one(rank: int) -> None:
    with pytest.raises(InvariantViolation):
        _evidence(dense_rank=rank)


def test_rank_and_score_come_together() -> None:
    # Ранг без оценки нельзя ни объяснить, ни воспроизвести при разборе выдачи.
    with pytest.raises(InvariantViolation):
        Evidence(
            chunk=make_chunk(),
            dense_rank=1,
            dense_score=None,
            sparse_rank=None,
            sparse_score=None,
            rrf_score=RrfScore(0.016),
            rerank_score=None,
        )


def test_evidence_found_by_both_branches_is_recognized_as_such() -> None:
    assert _evidence().found_by_both_branches
    assert not _evidence(sparse_rank=None).found_by_both_branches


def test_reranking_records_the_new_score() -> None:
    reranked = _evidence().reranked(Score(3.14))

    assert reranked.rerank_score == Score(3.14)
    # Исходные ранги остаются: без них не объяснить, что реранкер переставил.
    assert reranked.dense_rank == 1
    assert reranked.sparse_rank == 4


def test_reranking_twice_is_refused() -> None:
    # Второй проход означал бы, что порядок собран из двух разных прогонов.
    with pytest.raises(InvariantViolation):
        _evidence(rerank_score=Score(1.0)).reranked(Score(2.0))


def test_ordering_score_is_the_rerank_score_when_it_exists() -> None:
    # Контекст собирается по итогам реранкинга, а до него — по слиянию.
    assert _evidence(rerank_score=Score(2.5)).ordering_score == Score(2.5)
    assert _evidence().ordering_score == Score(0.031)


def test_evidence_carries_the_quality_of_its_chunk() -> None:
    unreliable = make_quality(
        method=ExtractionMethod.OCR, confidence=0.31, illegible_span_count=2
    )

    evidence = _evidence(quality=unreliable)

    assert evidence.chunk.quality.illegible_span_count == 2
