"""Чтение номера попытки из заголовков доставки."""

from __future__ import annotations

import pytest

from document_worker.presentation.messaging.headers import (
    HEADER_ATTEMPT,
    current_attempt,
    without_death_headers,
)

pytestmark = pytest.mark.unit


def test_first_delivery_has_no_attempts_behind_it() -> None:
    assert current_attempt({}) == 0


@pytest.mark.parametrize("raw", [3, "3"])
def test_attempt_is_read_as_a_number(raw: object) -> None:
    assert current_attempt({HEADER_ATTEMPT: raw}) == 3


@pytest.mark.parametrize("raw", ["много", None, True, [1], -5])
def test_unreadable_attempt_starts_the_count_over(raw: object) -> None:
    # Значение приходит извне, и доверять ему нельзя: непонятное считаем
    # первой доставкой, а не поводом отказать.
    assert current_attempt({HEADER_ATTEMPT: raw}) == 0


def test_broker_death_headers_are_dropped() -> None:
    headers = {"x-death": [{"count": 1}], "x-first-death-queue": "q", "x-attempt": 2}

    assert without_death_headers(headers) == {HEADER_ATTEMPT: 2}
