"""Тесты карты смещений нормализованного текста."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from document_worker.domain.errors import InvariantViolation
from document_worker.domain.normalization.offsets import (
    OffsetMap,
    OffsetMapBuilder,
    OffsetSegment,
    RuleAction,
)
from document_worker.domain.value_objects.text import TextSpan

pytestmark = pytest.mark.unit

# Перечисление закрыто: любое действие, способное дописать текст, ломает
# структурный запрет выдумывания.
EXPECTED_ACTIONS = {"keep", "drop", "map", "unfold", "collapse"}


def _built(*specs: tuple[int, int, RuleAction]) -> OffsetMap:
    builder = OffsetMapBuilder()
    for source, target, action in specs:
        builder.add(source=source, target=target, action=action)
    return builder.build()


def test_rule_action_enum_has_no_generative_action() -> None:
    assert {action.value for action in RuleAction} == EXPECTED_ACTIONS


def test_identity_map_when_no_edits() -> None:
    mapping = OffsetMap.identity(10)

    assert [mapping.project_offset(position) for position in range(11)] == list(
        range(11)
    )


def test_identity_map_of_empty_text() -> None:
    mapping = OffsetMap.identity(0)

    assert mapping.project_offset(0) == 0
    assert mapping.segments == ()


def test_dropped_range_collapses_to_its_left_edge() -> None:
    # "abXcd" → "abcd": символ на позиции 2 удалён.
    mapping = _built(
        (2, 2, RuleAction.KEEP),
        (1, 0, RuleAction.DROP),
        (2, 2, RuleAction.KEEP),
    )

    assert mapping.project_offset(2) == 2
    assert mapping.project_offset(3) == 2
    assert mapping.project_offset(5) == 4


def test_unfolded_char_expands_target_range() -> None:
    # "ﬁx" → "fix"
    mapping = _built((1, 2, RuleAction.UNFOLD), (1, 1, RuleAction.KEEP))

    assert mapping.project_offset(0) == 0
    assert mapping.project_offset(1) == 2
    assert mapping.project_offset(2) == 3


def test_collapsed_range_maps_to_single_position() -> None:
    # "a   b" → "a b"
    mapping = _built(
        (1, 1, RuleAction.KEEP),
        (3, 1, RuleAction.COLLAPSE),
        (1, 1, RuleAction.KEEP),
    )

    assert mapping.project_offset(1) == 1
    assert mapping.project_offset(2) == 1
    assert mapping.project_offset(4) == 2


def test_project_span_shifts_both_bounds() -> None:
    mapping = _built(
        (2, 2, RuleAction.KEEP),
        (1, 0, RuleAction.DROP),
        (2, 2, RuleAction.KEEP),
    )

    assert mapping.project_span(TextSpan(3, 5)) == TextSpan(2, 4)


def test_project_span_returns_none_when_fragment_is_destroyed() -> None:
    mapping = _built(
        (2, 2, RuleAction.KEEP),
        (2, 0, RuleAction.DROP),
        (1, 1, RuleAction.KEEP),
    )

    assert mapping.project_span(TextSpan(2, 4)) is None


def test_project_span_keeps_empty_span_empty() -> None:
    mapping = OffsetMap.identity(5)

    assert mapping.project_span(TextSpan(2, 2)) == TextSpan(2, 2)


def test_compose_chains_two_maps() -> None:
    # "aXbYc" → "abYc" → "abc"
    first = _built(
        (1, 1, RuleAction.KEEP),
        (1, 0, RuleAction.DROP),
        (3, 3, RuleAction.KEEP),
    )
    second = _built(
        (2, 2, RuleAction.KEEP),
        (1, 0, RuleAction.DROP),
        (1, 1, RuleAction.KEEP),
    )

    composed = first.compose(second)

    assert composed.source_length == 5
    assert composed.target_length == 3
    assert composed.project_offset(4) == 2


def test_compose_rejects_length_mismatch() -> None:
    first = OffsetMap.identity(5)
    second = OffsetMap.identity(4)

    with pytest.raises(InvariantViolation):
        first.compose(second)


def test_compose_with_identity_changes_nothing() -> None:
    mapping = _built(
        (2, 2, RuleAction.KEEP),
        (1, 0, RuleAction.DROP),
        (2, 2, RuleAction.KEEP),
    )

    composed = mapping.compose(OffsetMap.identity(mapping.target_length))

    assert [composed.project_offset(index) for index in range(6)] == [
        mapping.project_offset(index) for index in range(6)
    ]


def test_rejects_segments_not_covering_source() -> None:
    with pytest.raises(InvariantViolation):
        OffsetMap(
            source_length=5,
            target_length=2,
            segments=(OffsetSegment(0, 2, 0, 2, RuleAction.KEEP),),
        )


def test_rejects_insertion_from_nowhere() -> None:
    with pytest.raises(InvariantViolation):
        OffsetMap(
            source_length=1,
            target_length=2,
            segments=(
                OffsetSegment(0, 0, 0, 1, RuleAction.MAP),
                OffsetSegment(0, 1, 1, 2, RuleAction.KEEP),
            ),
        )


def test_rejects_drop_segment_with_non_empty_target() -> None:
    with pytest.raises(InvariantViolation):
        OffsetMap(
            source_length=1,
            target_length=1,
            segments=(OffsetSegment(0, 1, 0, 1, RuleAction.DROP),),
        )


def test_rejects_target_gap_between_segments() -> None:
    with pytest.raises(InvariantViolation):
        OffsetMap(
            source_length=2,
            target_length=3,
            segments=(
                OffsetSegment(0, 1, 0, 1, RuleAction.KEEP),
                OffsetSegment(1, 2, 2, 3, RuleAction.KEEP),
            ),
        )


def test_builder_ignores_empty_piece() -> None:
    builder = OffsetMapBuilder()
    builder.add(source=0, target=0, action=RuleAction.KEEP)
    builder.add(source=1, target=1, action=RuleAction.KEEP)

    assert builder.build().source_length == 1


def test_compose_carries_unfolded_segment_through() -> None:
    # "ﬁx" → "fix" → "fx": разворот лигатуры, затем удаление символа.
    expansion = _built((1, 2, RuleAction.UNFOLD), (1, 1, RuleAction.KEEP))
    removal = _built(
        (1, 1, RuleAction.KEEP),
        (1, 0, RuleAction.DROP),
        (1, 1, RuleAction.KEEP),
    )

    composed = expansion.compose(removal)

    assert composed.target_length == 2
    assert composed.project_offset(1) == 1


def test_compose_marks_fully_removed_segment_as_dropped() -> None:
    expansion = _built((1, 2, RuleAction.UNFOLD), (1, 1, RuleAction.KEEP))
    removal = _built((2, 0, RuleAction.DROP), (1, 1, RuleAction.KEEP))

    composed = expansion.compose(removal)

    assert composed.segments[0].action is RuleAction.DROP
    assert composed.target_length == 1


def test_rejects_overlapping_segments() -> None:
    with pytest.raises(InvariantViolation):
        OffsetMap(
            source_length=3,
            target_length=3,
            segments=(
                OffsetSegment(0, 2, 0, 2, RuleAction.KEEP),
                OffsetSegment(1, 3, 2, 3, RuleAction.KEEP),
            ),
        )


def test_offsets_are_counted_in_code_points_not_utf16_units() -> None:
    # Эмодзи вне BMP занимает две UTF-16 единицы, но одну кодовую точку.
    text = "a\U0001f600b"
    mapping = OffsetMap.identity(len(text))

    assert len(text) == 3
    assert mapping.project_offset(2) == 2


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=0, max_value=4),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_project_offset_is_monotonic(pairs: list[tuple[int, int]]) -> None:
    builder = OffsetMapBuilder()
    for source, target in pairs:
        action = RuleAction.DROP if target == 0 else RuleAction.MAP
        builder.add(source=source, target=target, action=action)
    mapping = builder.build()

    projected = [
        mapping.project_offset(position)
        for position in range(mapping.source_length + 1)
    ]

    assert projected == sorted(projected)


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=0, max_value=4),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_bisect_lookup_matches_linear_reference(pairs: list[tuple[int, int]]) -> None:
    builder = OffsetMapBuilder()
    for source, target in pairs:
        action = RuleAction.DROP if target == 0 else RuleAction.MAP
        builder.add(source=source, target=target, action=action)
    mapping = builder.build()

    for position in range(mapping.source_length + 1):
        assert mapping.project_offset(position) == _linear_projection(mapping, position)


def _linear_projection(mapping: OffsetMap, position: int) -> int:
    """Эталонная реализация проекции без двоичного поиска."""
    if position >= mapping.source_length:
        return mapping.target_length
    for segment in mapping.segments:
        if segment.source_start <= position < segment.source_end:
            source_span = segment.source_end - segment.source_start
            target_span = segment.target_end - segment.target_start
            if source_span == target_span:
                return segment.target_start + (position - segment.source_start)
            return segment.target_start
    return mapping.target_length
