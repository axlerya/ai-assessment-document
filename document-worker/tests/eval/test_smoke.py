"""Быстрый прогон стенда: инварианты на подмножестве корпуса.

Гоняется на каждом PR с меткой — полный корпус занимает минуты и упирается в
распознавание, а инварианты ломаются и на двух документах.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.eval.conftest import Measured

pytestmark = pytest.mark.eval

MIN_TEXT_LAYER_ACCURACY = 0.95


async def test_invariants_hold_on_the_smoke_subset(smoke_measured: Measured) -> None:
    _, totals, _, _ = smoke_measured

    assert totals["hallucination_rate"] == 0.0
    assert totals["chunk_page_linkage_errors"] == 0.0


async def test_extraction_method_is_chosen_correctly(smoke_measured: Measured) -> None:
    _, totals, _, _ = smoke_measured

    assert totals["text_layer_detection_accuracy"] >= MIN_TEXT_LAYER_ACCURACY


async def test_digital_pdf_is_read_without_errors(smoke_measured: Measured) -> None:
    # Текстовый слой читается точно: остаток — разница переносов, которую
    # нормализация уже сняла.
    _, _, categories, _ = smoke_measured

    assert categories["digital_pdf"]["cer"] <= 0.01
