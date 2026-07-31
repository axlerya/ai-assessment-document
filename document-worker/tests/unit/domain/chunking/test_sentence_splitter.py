"""Границы предложений внутри блока."""

from __future__ import annotations

import pytest

from document_worker.domain.chunking.sentence_splitter import SentenceSplitter

pytestmark = pytest.mark.unit

SPLITTER = SentenceSplitter()


def test_splits_on_sentence_end() -> None:
    text = "Первое предложение. Второе предложение. Третье предложение."

    assert SPLITTER.boundaries(text) == (
        text.index("Второе"),
        text.index("Третье"),
    )


def test_does_not_split_after_known_abbreviation() -> None:
    assert SPLITTER.boundaries("Согласно ст. 12 ГК РФ стороны отвечают.") == ()


def test_does_not_split_after_initials() -> None:
    assert SPLITTER.boundaries("Иванов И. И. подписал договор.") == ()


def test_no_boundaries_without_punctuation() -> None:
    assert SPLITTER.boundaries("сплошной текст без единого знака препинания") == ()


def test_boundary_after_closing_quote() -> None:
    text = 'Стороны согласовали "цену". Далее следует условие.'

    assert SPLITTER.boundaries(text) == (text.index("Далее"),)
