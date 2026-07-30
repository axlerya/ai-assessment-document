"""Тесты формата маркера неразборчивого фрагмента."""

from __future__ import annotations

import pytest

from document_worker.domain.markers import MARKER_RE, IllegibleMarker, MarkedText
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import IllegibleReason
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan

pytestmark = pytest.mark.unit


def _illegible(
    start: int, end: int, *, raw: str, line: int | None = None
) -> IllegibleSpan:
    return IllegibleSpan(
        span=TextSpan(start, end),
        confidence=OcrConfidence(0.31),
        reason=IllegibleReason.LOW_OCR_CONFIDENCE,
        raw_text=raw,
        line_number=line,
    )


def test_marker_matches_declared_format() -> None:
    marker = IllegibleMarker(
        line_number=14,
        span=TextSpan(0, 0),
        confidence=OcrConfidence(0.31),
    )

    assert marker.render() == "[НЕРАЗБОРЧИВО: строка 14, confidence=0.31]"


def test_marker_without_line_number_uses_char_locator() -> None:
    marker = IllegibleMarker(
        line_number=None,
        span=TextSpan(120, 138),
        confidence=OcrConfidence(0.31),
    )

    assert marker.render() == "[НЕРАЗБОРЧИВО: символы 120-138, confidence=0.31]"


def test_confidence_is_rendered_with_two_decimals() -> None:
    marker = IllegibleMarker(
        line_number=1,
        span=TextSpan(0, 0),
        confidence=OcrConfidence(0.5),
    )

    assert marker.render().endswith("confidence=0.50]")


@pytest.mark.parametrize(
    "marker",
    [
        IllegibleMarker(
            line_number=14, span=TextSpan(0, 0), confidence=OcrConfidence(0.31)
        ),
        IllegibleMarker(
            line_number=None, span=TextSpan(120, 138), confidence=OcrConfidence(0.07)
        ),
    ],
)
def test_marker_regex_parses_rendered_marker_back(marker: IllegibleMarker) -> None:
    assert IllegibleMarker.parse(marker.render()) == marker


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "просто текст",
        "[НЕРАЗБОРЧИВО: строка 14]",
        "[НЕРАЗБОРЧИВО: строка 14, confidence=0.3]",
        "[UNREADABLE: line 14, confidence=0.31]",
    ],
)
def test_parse_rejects_foreign_string(raw: str) -> None:
    with pytest.raises(ValueError, match="маркер"):
        IllegibleMarker.parse(raw)


def test_marker_is_built_from_illegible_span() -> None:
    span = _illegible(5, 8, raw="абв", line=3)

    marker = IllegibleMarker.of(span)

    assert marker.line_number == 3
    assert marker.span == TextSpan(5, 8)
    assert marker.confidence == OcrConfidence(0.31)


def test_marked_text_replaces_span_with_marker() -> None:
    text = "начало абв конец"
    marked = MarkedText(text=text, spans=(_illegible(7, 10, raw="абв", line=1),))

    assert marked.render() == "начало [НЕРАЗБОРЧИВО: строка 1, confidence=0.31] конец"


def test_marked_text_renders_several_spans_without_shifting_offsets() -> None:
    text = "аbc и dеf тут"
    marked = MarkedText(
        text=text,
        spans=(
            _illegible(0, 3, raw="аbc"),
            _illegible(6, 9, raw="dеf"),
        ),
    )

    rendered = marked.render()

    assert rendered.startswith("[НЕРАЗБОРЧИВО: символы 0-3, confidence=0.31]")
    assert "[НЕРАЗБОРЧИВО: символы 6-9, confidence=0.31]" in rendered
    assert rendered.endswith(" тут")


def test_zero_length_span_inserts_marker_without_removing_text() -> None:
    span = IllegibleSpan(
        span=TextSpan(0, 0),
        confidence=OcrConfidence.ZERO,
        reason=IllegibleReason.NO_TEXT_RECOGNIZED,
        raw_text="",
    )
    marked = MarkedText(text="текст", spans=(span,))

    assert marked.render() == "[НЕРАЗБОРЧИВО: символы 0-0, confidence=0.00]текст"


def test_marked_text_without_spans_returns_original_text() -> None:
    assert MarkedText(text="текст", spans=()).render() == "текст"


def test_strip_markers_returns_text_without_markers() -> None:
    text = "начало абв конец"
    marked = MarkedText(text=text, spans=(_illegible(7, 10, raw="абв", line=1),))

    assert MarkedText.strip_markers(marked.render()) == "начало  конец"


def test_regular_text_contains_no_markers() -> None:
    assert MARKER_RE.search("обычный текст договора без пометок") is None
