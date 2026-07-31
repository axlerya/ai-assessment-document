"""Сборка чанков: границы, перекрытие, нумерация.

Смена секции — жёсткое условие разрыва: правило устава «chunk не должен
объединять несвязанные разделы только ради достижения фиксированного размера»
живёт именно здесь, а не в комментарии.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from document_worker.domain.chunking.chunk_assembler import fitting_prefix_length
from document_worker.domain.chunking.pipeline import build_pipeline
from document_worker.domain.constants import MAX_CHUNK_OVERLAP_CHARS
from tests.unit.domain.chunking.support import (
    FakeTokenCounter,
    default_policy,
    text_layer_page,
)

if TYPE_CHECKING:
    from document_worker.domain.chunking.chunk_assembler import ChunkDraft

pytestmark = pytest.mark.unit

# Бюджет уменьшен, чтобы границы срабатывали на текстах в несколько строк:
# токен здесь — четыре символа, значит цель это 80 символов, предел — 120.
SMALL_POLICY = default_policy(
    target_tokens=20, max_tokens=30, min_tokens=10, overlap_tokens=8
)

SENTENCES = (
    "Первое предложение здесь. Второе предложение здесь. "
    "Третье предложение здесь. Четвёртое предложение тут. Пятое предложение тут."
)


def drafts_of(*contents: str) -> tuple[ChunkDraft, ...]:
    """Черновики чанков для страниц с заданными текстами."""
    pages = [
        text_layer_page(content, number=number)
        for number, content in enumerate(contents, start=1)
    ]
    return build_pipeline(SMALL_POLICY, FakeTokenCounter()).run(pages)


def test_never_merges_two_different_articles_into_one_chunk() -> None:
    drafts = drafts_of(
        "Статья 11. Первое\n"
        "Текст первой статьи достаточно короткий.\n"
        "Статья 12. Второе\n"
        "Текст второй статьи достаточно короткий."
    )

    assert len(drafts) == 2
    assert [draft.heading_path for draft in drafts] == [
        ("Статья 11",),
        ("Статья 12",),
    ]


def test_never_crosses_page_boundary() -> None:
    first, second = "Текст первой страницы целиком.", "Текст второй страницы целиком."
    pages = [
        text_layer_page(first, number=1),
        text_layer_page(second, number=2),
    ]
    drafts = build_pipeline(SMALL_POLICY, FakeTokenCounter()).run(pages)

    assert [int(draft.page_number) for draft in drafts] == [1, 2]
    assert [draft.text for draft in drafts] == [first, second]


def test_offsets_reproduce_source_text_exactly() -> None:
    content = SENTENCES
    page = text_layer_page(content)

    for draft in build_pipeline(SMALL_POLICY, FakeTokenCounter()).run([page]):
        assert draft.span.slice_of(content) == draft.text


def test_text_length_equals_offset_difference() -> None:
    for draft in drafts_of(SENTENCES):
        assert len(draft.text) == draft.span.end - draft.span.start


def test_respects_max_tokens_after_overlap() -> None:
    for draft in drafts_of(SENTENCES):
        assert draft.token_count <= SMALL_POLICY.max_tokens
        assert draft.overlap_prefix_chars <= MAX_CHUNK_OVERLAP_CHARS


def test_adds_overlap_only_inside_one_page_and_section() -> None:
    drafts = drafts_of(SENTENCES)

    assert drafts[0].overlap_prefix_chars == 0
    assert any(draft.overlap_prefix_chars > 0 for draft in drafts[1:])


def test_does_not_add_overlap_across_section_boundary() -> None:
    drafts = drafts_of(
        "Статья 11. Первое\n"
        "Текст первой статьи достаточно короткий.\n"
        "Статья 12. Второе\n"
        "Текст второй статьи достаточно короткий."
    )

    assert [draft.overlap_prefix_chars for draft in drafts] == [0, 0]


def test_overlap_never_makes_two_chunks_start_at_same_offset() -> None:
    # Совпадение начал двух чанков одной страницы падает с 23505 по
    # uq__document_chunks__page__start уже в терминальной части обработки.
    starts = [draft.span.start for draft in drafts_of(SENTENCES)]

    assert starts == sorted(set(starts))


def test_table_chunk_is_never_merged_with_paragraph() -> None:
    drafts = drafts_of(
        "| Наименование | Цена |\n"
        "| Бумага | 120 |\n"
        "Текст абзаца, который идёт сразу после таблицы."
    )

    assert len(drafts) == 2
    assert drafts[0].text.startswith("|")
    assert drafts[1].text.startswith("Текст")


def test_table_longer_than_max_tokens_splits_on_row_boundaries() -> None:
    # Строки с отступом: разрез идёт сразу за переводом строки, и без обрезки
    # края следующая часть начиналась бы с пробелов.
    rows = "\n".join(f"  | Позиция {number} | 100 |" for number in range(12))

    drafts = drafts_of(rows)

    assert len(drafts) > 1
    for draft in drafts:
        assert draft.text.startswith("|")
        assert draft.text.endswith("|")


def test_keeps_section_heading_out_of_chunk_text() -> None:
    drafts = drafts_of(
        "Статья 12. Ответственность\nИсполнитель отвечает за просрочку поставки."
    )

    assert len(drafts) == 1
    assert "Статья 12" not in drafts[0].text
    assert drafts[0].heading_path[-1] == "Статья 12"


def test_chunk_ordinals_are_dense_and_ordered() -> None:
    drafts = drafts_of(SENTENCES)

    assert [draft.ordinal for draft in drafts] == list(range(len(drafts)))


def test_ordinals_restart_on_every_page() -> None:
    drafts = drafts_of("Текст первой страницы.", "Текст второй страницы.")

    assert [draft.ordinal for draft in drafts] == [0, 0]


def test_emits_single_chunk_for_one_word_document() -> None:
    drafts = drafts_of("Договор")

    assert len(drafts) == 1
    assert drafts[0].token_count >= 1
    assert drafts[0].quality.is_retrievable is True


def test_merges_short_chunk_with_neighbour_inside_same_section() -> None:
    drafts = drafts_of(SENTENCES)

    assert all(
        draft.token_count >= SMALL_POLICY.min_tokens or len(drafts) == 1
        for draft in drafts
    )


def test_does_not_merge_short_chunk_across_page() -> None:
    drafts = drafts_of("Короткий хвост.", "Начало следующей страницы.")

    assert len(drafts) == 2


def test_single_long_line_without_paragraphs_yields_multiple_chunks() -> None:
    content = "а" * 500

    drafts = drafts_of(content)

    assert len(drafts) > 1
    assert all(draft.token_count <= SMALL_POLICY.max_tokens for draft in drafts)
    assert "".join(draft.text for draft in drafts) == content


def test_blank_chunks_are_dropped() -> None:
    assert drafts_of("   \n\n  \t ") == ()


def test_two_short_paragraphs_of_one_section_become_one_chunk() -> None:
    drafts = drafts_of("Первый абзац.\n\nВторой абзац.")

    assert len(drafts) == 1
    assert drafts[0].text == "Первый абзац.\n\nВторой абзац."


def test_merges_short_paragraph_with_its_neighbour() -> None:
    drafts = drafts_of("а" * 100 + "\n\nДа и всё.")

    assert len(drafts) == 1
    assert drafts[0].text.endswith("Да и всё.")


def test_falls_back_to_word_boundaries_when_a_sentence_is_too_long() -> None:
    # Предложение длиннее предела: резать по границам предложений нечем,
    # и каскад спускается на уровень слов.
    long_sentence = "Слово" + " слово" * 40 + "."

    drafts = drafts_of(f"{long_sentence} Второе предложение здесь.")

    assert len(drafts) > 1
    assert all(draft.token_count <= SMALL_POLICY.max_tokens for draft in drafts)


def test_overlap_is_refused_when_sentence_is_longer_than_the_char_cap() -> None:
    # Предложение целиком не влезает в потолок перекрытия, а резать его пополам
    # запрещено: перекрытия нет вовсе.
    policy = default_policy(
        target_tokens=250, max_tokens=300, min_tokens=200, overlap_tokens=150
    )
    sentence = "Слово" + " слово" * 73 + "."
    page = text_layer_page(" ".join([sentence] * 4))

    drafts = build_pipeline(policy, FakeTokenCounter()).run([page])

    assert len(drafts) > 1
    assert all(draft.overlap_prefix_chars == 0 for draft in drafts)


def test_fitting_prefix_uses_logarithmic_number_of_count_calls() -> None:
    # Наращивание по символу дало бы двадцать тысяч вызовов токенизатора
    # на один жёсткий разрез.
    text = "я" * 20_000
    counter = FakeTokenCounter()

    length = fitting_prefix_length(text, counter=counter, limit=30)

    assert 0 < length < len(text)
    assert counter.calls <= 2 * math.log2(len(text))
