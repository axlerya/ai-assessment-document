"""Сырой текст страницы и перенос диапазонов в канонический текст.

Единственный сборщик текста страницы. Он принимает только то, что вернул
движок, и двигает курсор на длину слова; текста, которого движок не выдавал,
добавить сюда нечем — это и есть барьер против домысливания.

Политика читаемости работает на сыром тексте: там позиции слов известны точно и
совпадают с боксами. Нормализация необратима, и запуск политики после неё
означал бы поиск плохих слов в тексте, где границы слов уже сдвинулись.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from document_worker.application.errors import IllegibleSpanLostError
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.text import RecognizedWord, TextSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.application.dto.ocr import RecognizedWordDTO
    from document_worker.domain.normalization.normalizer import NormalizedText
    from document_worker.domain.value_objects.text import IllegibleSpan

WORD_SEPARATOR = " "
LINE_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class AssembledPageText:
    """Сырой текст страницы вместе со словами, из которых он собран."""

    content: str
    words: tuple[RecognizedWord, ...]


class PageTextAssembler:
    """Складывает текст страницы из слов распознавателя."""

    def assemble(self, words: Sequence[RecognizedWordDTO]) -> AssembledPageText:
        """Собирает текст и переносит слова в доменные значения."""
        parts: list[str] = []
        assembled: list[RecognizedWord] = []
        cursor = 0
        previous_line: int | None = None
        for word in words:
            separator = _separator(previous_line, word.line_index)
            parts.append(separator)
            cursor += len(separator)
            parts.append(word.text)
            assembled.append(
                RecognizedWord(
                    text=word.text,
                    confidence=OcrConfidence(word.confidence),
                    span=TextSpan(cursor, cursor + len(word.text)),
                    bbox=word.bbox,
                    # Номера строк в документе считаются с единицы, индексы
                    # движка — с нуля.
                    line_number=word.line_index + 1,
                )
            )
            cursor += len(word.text)
            previous_line = word.line_index
        return AssembledPageText(content="".join(parts), words=tuple(assembled))


def reproject_spans(
    spans: Sequence[IllegibleSpan],
    normalized: NormalizedText,
) -> tuple[IllegibleSpan, ...]:
    """Переносит диапазоны из сырого текста в канонический.

    Текст фрагмента берётся срезом канонического текста: иначе инвариант
    сущности «сохранённый фрагмент совпадает со срезом» рвался бы на каждом
    диапазоне, внутри которого нормализация заменила неразрывный пробел.

    Raises:
        IllegibleSpanLostError: Диапазон уничтожен нормализацией целиком.
    """
    return tuple(_reprojected(span, normalized) for span in spans)


def _reprojected(span: IllegibleSpan, normalized: NormalizedText) -> IllegibleSpan:
    projected = normalized.offsets.project_span(span.span)
    if projected is None:
        # Молча сохранённый съехавший диапазон всплыл бы только при цитировании
        # оператору — то есть тогда, когда исправлять поздно.
        raise IllegibleSpanLostError(
            "диапазон неразборчивости потерян нормализацией",
            context={"reason": span.reason.value, "start": span.span.start},
        )
    return replace(
        span, span=projected, raw_text=projected.slice_of(normalized.content)
    )


def _separator(previous_line: int | None, line: int) -> str:
    if previous_line is None:
        return ""
    return LINE_SEPARATOR if line != previous_line else WORD_SEPARATOR
