"""Текстовые диапазоны, распознанные слова и неразборчивые фрагменты.

Смещения — относительно текста своей страницы, в кодовых точках Unicode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.constants import (
    MAX_ILLEGIBLE_CONFIDENCE,
    MAX_PAGE_TEXT_LENGTH,
)
from document_worker.domain.errors import (
    InvalidIllegibleSpan,
    InvalidRecognizedWord,
    InvalidTextSpan,
)
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import IllegibleReason

if TYPE_CHECKING:
    from document_worker.domain.value_objects.geometry import BoundingBox

MIN_LINE_NUMBER = 1


@dataclass(frozen=True, slots=True, order=True)
class TextSpan:
    """Полуинтервал [start, end) в тексте страницы. Пустой диапазон допустим."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Проверяет порядок и границы."""
        if self.start < 0:
            raise InvalidTextSpan(
                "начало диапазона отрицательно",
                context={"start": self.start},
            )
        if self.end < self.start:
            raise InvalidTextSpan(
                "конец диапазона раньше начала",
                context={"start": self.start, "end": self.end},
            )
        if self.end > MAX_PAGE_TEXT_LENGTH:
            raise InvalidTextSpan(
                f"диапазон выходит за предел {MAX_PAGE_TEXT_LENGTH}",
                context={"end": self.end},
            )

    @property
    def length(self) -> int:
        """Длина в кодовых точках."""
        return self.end - self.start

    @property
    def is_empty(self) -> bool:
        """Пуст ли диапазон."""
        return self.length == 0

    def overlaps(self, other: TextSpan) -> bool:
        """Пересекаются ли диапазоны. Пустой не пересекается ни с чем."""
        if self.is_empty or other.is_empty:
            return False
        return self.start < other.end and other.start < self.end

    def contains(self, other: TextSpan) -> bool:
        """Целиком ли другой диапазон лежит внутри этого."""
        return self.start <= other.start and other.end <= self.end

    def shift(self, delta: int) -> TextSpan:
        """Сдвигает обе границы."""
        return TextSpan(self.start + delta, self.end + delta)

    def slice_of(self, text: str) -> str:
        """Срез текста по этому диапазону."""
        return text[self.start : self.end]

    def within(self, text_length: int) -> bool:
        """Помещается ли диапазон в текст такой длины."""
        return self.end <= text_length


@dataclass(frozen=True, slots=True)
class RecognizedWord:
    """Слово, выданное распознавателем, со своей уверенностью и позицией."""

    text: str
    confidence: OcrConfidence
    span: TextSpan
    bbox: BoundingBox | None = None
    line_number: int | None = None

    def __post_init__(self) -> None:
        """Сверяет длину текста с диапазоном."""
        if self.span.length != len(self.text):
            raise InvalidRecognizedWord(
                "длина слова не совпадает с диапазоном",
                context={
                    "text_length": len(self.text),
                    "span_length": self.span.length,
                },
            )
        if not self.text and self.confidence != OcrConfidence.ZERO:
            raise InvalidRecognizedWord(
                "пустое слово не может иметь ненулевую уверенность",
                context={"confidence": self.confidence.value},
            )
        if self.line_number is not None and self.line_number < MIN_LINE_NUMBER:
            raise InvalidRecognizedWord(
                "номер строки начинается с единицы",
                context={"line_number": self.line_number},
            )


@dataclass(frozen=True, slots=True)
class IllegibleSpan:
    """Диапазон, который не удалось уверенно прочитать.

    Готовой строки маркера здесь нет: маркер рендерится на выдаче.
    """

    span: TextSpan
    confidence: OcrConfidence
    reason: IllegibleReason
    raw_text: str
    bbox: BoundingBox | None = None
    line_number: int | None = None

    def __post_init__(self) -> None:
        """Сверяет причину с содержимым диапазона."""
        if self.confidence.value > MAX_ILLEGIBLE_CONFIDENCE:
            raise InvalidIllegibleSpan(
                f"уверенно прочитанный фрагмент нельзя пометить неразборчивым: "
                f"порог {MAX_ILLEGIBLE_CONFIDENCE}",
                context={"confidence": self.confidence.value},
            )
        if self.line_number is not None and self.line_number < MIN_LINE_NUMBER:
            raise InvalidIllegibleSpan(
                "номер строки начинается с единицы",
                context={"line_number": self.line_number},
            )
        if self.reason is IllegibleReason.NO_TEXT_RECOGNIZED:
            self._require_nothing_recognized()
        if self.reason.is_technical:
            self._require_nothing_recognized()
            if not self.span.is_empty:
                raise InvalidIllegibleSpan(
                    "технический сбой не оставляет диапазона в тексте",
                    context={"reason": self.reason.value, "length": self.span.length},
                )

    def _require_nothing_recognized(self) -> None:
        # Непустой raw_text здесь означал бы выдуманный текст.
        if self.raw_text:
            raise InvalidIllegibleSpan(
                "распознаватель ничего не выдал, а текст фрагмента не пуст",
                context={"reason": self.reason.value, "raw_text": self.raw_text},
            )
        if self.confidence != OcrConfidence.ZERO:
            raise InvalidIllegibleSpan(
                "нераспознанный фрагмент не может иметь ненулевую уверенность",
                context={
                    "reason": self.reason.value,
                    "confidence": self.confidence.value,
                },
            )
