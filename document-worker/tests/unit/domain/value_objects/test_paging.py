"""Тесты номера страницы."""

from __future__ import annotations

import pytest

from document_worker.domain.constants import MAX_PAGES
from document_worker.domain.errors import InvalidPageNumber
from document_worker.domain.value_objects.paging import PageNumber

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [0, -1, MAX_PAGES + 1])
def test_rejects_value_outside_bounds(value: int) -> None:
    with pytest.raises(InvalidPageNumber):
        PageNumber(value)


@pytest.mark.parametrize("value", [1, 42, MAX_PAGES])
def test_accepts_value_within_bounds(value: int) -> None:
    assert int(PageNumber(value)) == value


def test_page_numbers_are_ordered() -> None:
    assert sorted([PageNumber(3), PageNumber(1), PageNumber(2)]) == [
        PageNumber(1),
        PageNumber(2),
        PageNumber(3),
    ]


def test_next_page_returns_following_number() -> None:
    assert PageNumber(1).next() == PageNumber(2)


def test_next_page_beyond_limit_raises() -> None:
    with pytest.raises(InvalidPageNumber):
        PageNumber(MAX_PAGES).next()


def test_page_number_is_immutable() -> None:
    page = PageNumber(1)

    with pytest.raises(AttributeError):
        page.value = 2  # type: ignore[misc]
