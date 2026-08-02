"""Политика сравнения с baseline и вид отчёта."""

from __future__ import annotations

import pytest

from eval.run_eval import (
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_THRESHOLD,
    Report,
    _compare,
    _markdown,
)

pytestmark = pytest.mark.unit

BASELINE = {
    "corpus_hash": "abc",
    "aggregate": {"cer": 0.10, "boundary_f1": 0.50, "hallucination_rate": 0.0},
}


def _report(**aggregate: float) -> Report:
    totals = {"cer": 0.10, "boundary_f1": 0.50, "hallucination_rate": 0.0}
    totals.update(aggregate)
    return Report(
        corpus_hash="abc",
        corpus_version="1.0.0",
        chunking_version="1.0.0",
        font_version="dejavu-2.37",
        environment={"python": "3.12.0"},
        aggregate=totals,
        by_category={
            "digital_pdf": {
                "cer": 0.0,
                "wer": 0.0,
                "boundary_f1": 0.4,
                "pages": 13.0,
            }
        },
        pages=(),
    )


def test_unchanged_metrics_pass() -> None:
    assert _compare(_report(), BASELINE) == EXIT_OK


def test_small_drift_is_within_tolerance() -> None:
    # Порядок распознавания и время не повторяются между машинами: без запаса
    # гейт краснел бы от смены раннера, а не от качества.
    assert _compare(_report(cer=0.115, boundary_f1=0.48), BASELINE) == EXIT_OK


def test_growing_error_is_a_regression() -> None:
    assert _compare(_report(cer=0.15), BASELINE) == EXIT_REGRESSION


def test_falling_f1_is_a_regression() -> None:
    assert _compare(_report(boundary_f1=0.40), BASELINE) == EXIT_REGRESSION


def test_invariant_has_no_tolerance() -> None:
    # Выдуманный текст — дефект, а не деградация: допуска у него нет.
    assert _compare(_report(hallucination_rate=0.001), BASELINE) == EXIT_THRESHOLD


def test_markdown_shows_every_metric_and_category() -> None:
    rendered = _markdown(_report())

    assert "`cer`" in rendered
    assert "`digital_pdf`" in rendered
    assert "dejavu-2.37" in rendered
