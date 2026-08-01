"""Сборка сырого текста страницы из слов распознавателя.

Единственный сборщик текста страницы: он принимает только то, что вернул
движок, и увеличивает длину на длину слова. Текста, которого движок не выдавал,
добавить здесь просто нечем — это и есть барьер против домысливания.
"""

from __future__ import annotations

import pytest

from document_worker.application.dto.ocr import ConfidenceSource, RecognizedWordDTO
from document_worker.application.errors import IllegibleSpanLostError
from document_worker.application.services.page_text import (
    PageTextAssembler,
    reproject_spans,
)
from document_worker.domain.normalization.normalizer import TextNormalizer
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ExtractionMethod, IllegibleReason
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan

pytestmark = pytest.mark.unit

ASSEMBLER = PageTextAssembler()
NORMALIZER = TextNormalizer()
BOX = BoundingBox(0.1, 0.1, 0.2, 0.2)


def word(
    text: str,
    *,
    line: int = 0,
    index: int = 0,
    confidence: float = 0.9,
) -> RecognizedWordDTO:
    return RecognizedWordDTO(
        text=text,
        confidence=confidence,
        bbox=BOX,
        line_index=line,
        word_index=index,
        confidence_source=ConfidenceSource.WORD,
    )


def test_words_of_one_line_are_joined_by_a_single_space() -> None:
    assembled = ASSEMBLER.assemble(
        [word("Договор", index=0), word("поставки", index=1)]
    )

    assert assembled.content == "Договор поставки"


def test_lines_are_separated_by_a_newline() -> None:
    assembled = ASSEMBLER.assemble([word("Первая"), word("Вторая", line=1)])

    assert assembled.content == "Первая\nВторая"


def test_every_word_span_points_at_its_own_text() -> None:
    # Смещения слов — то, по чему политика читаемости строит диапазоны:
    # съехавший спан пометит неразборчивым соседнее слово.
    assembled = ASSEMBLER.assemble(
        [word("Первая"), word("строка", index=1), word("Вторая", line=1)]
    )

    for recognized in assembled.words:
        assert recognized.span.slice_of(assembled.content) == recognized.text


def test_page_text_is_concatenation_of_word_texts_only() -> None:
    words = [word("Договор"), word("поставки", index=1), word("товаров", line=1)]

    assembled = ASSEMBLER.assemble(words)

    assert "".join(assembled.content.split()) == "".join(item.text for item in words)


def test_empty_word_list_gives_empty_page() -> None:
    assembled = ASSEMBLER.assemble([])

    assert assembled.content == ""
    assert assembled.words == ()


def test_line_numbers_start_at_one() -> None:
    assembled = ASSEMBLER.assemble([word("Строка", line=0)])

    assert assembled.words[0].line_number == 1


def test_span_is_reprojected_after_whitespace_collapse() -> None:
    # Нормализация схлопывает пробелы и сдвигает всё, что идёт следом:
    # диапазон, оставленный на прежнем месте, укажет не туда.
    raw = "Договор   поставки товаров"
    normalized = NORMALIZER.normalize(raw, source=ExtractionMethod.OCR)
    span = IllegibleSpan(
        span=TextSpan(raw.index("товаров"), len(raw)),
        confidence=OcrConfidence(0.3),
        reason=IllegibleReason.LOW_OCR_CONFIDENCE,
        raw_text="товаров",
    )

    reprojected = reproject_spans((span,), normalized)

    assert reprojected[0].span.slice_of(normalized.content) == "товаров"
    assert reprojected[0].raw_text == "товаров"


def test_zero_length_span_survives_reprojection() -> None:
    normalized = NORMALIZER.normalize("", source=ExtractionMethod.OCR)
    span = IllegibleSpan(
        span=TextSpan(0, 0),
        confidence=OcrConfidence.ZERO,
        reason=IllegibleReason.NO_TEXT_RECOGNIZED,
        raw_text="",
    )

    assert reproject_spans((span,), normalized) == (span,)


def test_span_destroyed_by_normalization_is_an_error() -> None:
    # Молча сохранённый съехавший диапазон всплыл бы только при цитировании
    # оператору — то есть тогда, когда исправлять поздно.
    # Фрагмент целиком состоял из мягких переносов, которые нормализация
    # удаляет: переносить нечего.
    raw = "Договор ­­ поставки"
    normalized = NORMALIZER.normalize(raw, source=ExtractionMethod.OCR)
    span = IllegibleSpan(
        span=TextSpan(8, 10),
        confidence=OcrConfidence(0.3),
        reason=IllegibleReason.LOW_OCR_CONFIDENCE,
        raw_text=raw[8:10],
    )

    with pytest.raises(IllegibleSpanLostError):
        reproject_spans((span,), normalized)
