"""Гейты качества: абсолютные пороги, отсутствие регресса и защита корпуса.

По умолчанию выключены маркером `eval`: прогон занимает минуты и упирается в
распознавание, а красная сборка от чужой загрузки машины обесценила бы гейт.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from eval.run_eval import LOWER_IS_BETTER, TOLERANCE_BETTER, TOLERANCE_WORSE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.eval.conftest import Measured

pytestmark = pytest.mark.eval

BASELINE = Path(__file__).resolve().parents[2] / "eval" / "baseline.json"


def _baseline() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(BASELINE.read_text(encoding="utf-8"))
    return loaded


async def test_corpus_hash_matches_baseline(measured: Measured) -> None:
    # Без этой проверки достаточно убрать зашумлённые сканы, и все метрики
    # зеленеют — а по одному отчёту этого не видно.
    manifest, _, _, _ = measured

    assert manifest.corpus_hash == _baseline()["corpus_hash"]


async def test_hallucination_rate_is_zero(measured: Measured) -> None:
    # Инвариант устава: придумывать текст на месте нечитаемого запрещено.
    _, totals, _, _ = measured

    assert totals["hallucination_rate"] == 0.0


async def test_chunk_page_linkage_errors_are_zero(measured: Measured) -> None:
    # Инвариант цитирования: текст чанка — точный срез текста своей страницы.
    _, totals, _, _ = measured

    assert totals["chunk_page_linkage_errors"] == 0.0


async def test_all_absolute_thresholds_are_satisfied(measured: Measured) -> None:
    _, totals, categories, _ = measured
    thresholds = _baseline()["thresholds"]

    for name, bound in thresholds["aggregate"].items():
        _assert_within(name, totals[name], bound)
    for category, bounds in thresholds["by_category"].items():
        for name, bound in bounds.items():
            _assert_within(f"{category}.{name}", categories[category][name], bound)


async def test_no_regression_against_baseline(measured: Measured) -> None:
    _, totals, _, _ = measured
    before: Mapping[str, float] = _baseline()["aggregate"]

    for name, was in before.items():
        now = totals[name]
        if name in LOWER_IS_BETTER:
            assert now <= was + TOLERANCE_WORSE, name
        else:
            assert now >= was - TOLERANCE_BETTER, name


async def test_confidence_does_not_contradict_actual_accuracy(
    measured: Measured,
) -> None:
    # Порог откалиброван по факту и сейчас близок к нулю: уверенность RapidOCR
    # качество страницы не предсказывает. Гейт держит хотя бы отсутствие
    # обратной связи — фильтр по уверенности у соседних сервисов не должен
    # отбрасывать как раз хорошие страницы.
    _, totals, _, _ = measured
    bound = _baseline()["thresholds"]["aggregate"]["confidence_calibration_rho"]

    assert totals["confidence_calibration_rho"] >= bound["min"]


def _assert_within(name: str, value: float, limits: Mapping[str, float]) -> None:
    if "max" in limits:
        assert value <= limits["max"], f"{name}={value:.4f} > {limits['max']}"
    if "min" in limits:
        assert value >= limits["min"], f"{name}={value:.4f} < {limits['min']}"
