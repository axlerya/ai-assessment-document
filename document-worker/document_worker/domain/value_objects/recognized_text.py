"""Канонический текст страницы.

Текст хранится нормализованным и без маркеров: маркеры — представление, они
рендерятся на выдаче. Отсюда единая база отсчёта для всех смещений.

Подставить текст через этот тип нельзя: объект frozen, метода замены содержимого
нет, а несовпадение сохранённого фрагмента со срезом даёт FabricatedTextDetected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from document_worker.domain.constants import MAX_PAGE_TEXT_LENGTH
from document_worker.domain.errors import (
    FabricatedTextDetected,
    InvalidTextSpan,
    InvariantViolation,
)
from document_worker.domain.markers import MARKER_RE
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan

if TYPE_CHECKING:
    from document_worker.domain.value_objects.enums import IllegibleReason

# Способы, у которых уверенности не существует: у текстового слоя она не нужна,
# у нечитаемой страницы её нечему измерять. Ноль тут исказил бы средние так же,
# как единица от текстового слоя.
_METHODS_WITHOUT_CONFIDENCE = frozenset(
    {ExtractionMethod.TEXT_LAYER, ExtractionMethod.NONE}
)


@dataclass(frozen=True, slots=True)
class RecognizedText:
    """Текст страницы вместе со способом извлечения и неразборчивыми фрагментами."""

    content: str
    method: ExtractionMethod
    confidence: OcrConfidence | None
    illegible_spans: tuple[IllegibleSpan, ...] = field(default=())

    def __post_init__(self) -> None:
        """Проверяет согласованность способа, уверенности и диапазонов."""
        self._validate_content()
        self._validate_method_and_confidence()
        self._validate_spans()

    def _validate_content(self) -> None:
        if len(self.content) > MAX_PAGE_TEXT_LENGTH:
            raise InvariantViolation(
                f"текст страницы длиннее {MAX_PAGE_TEXT_LENGTH} кодовых точек",
                context={"length": len(self.content)},
            )
        if MARKER_RE.search(self.content) is not None:
            raise InvariantViolation(
                "канонический текст не содержит маркеров неразборчивости",
                context={"method": self.method.value},
            )

    def _validate_method_and_confidence(self) -> None:
        without_confidence = self.method in _METHODS_WITHOUT_CONFIDENCE
        if without_confidence and self.confidence is not None:
            raise InvariantViolation(
                f"у способа {self.method.value} уверенности не существует",
                context={"method": self.method.value},
            )
        if not without_confidence and self.confidence is None:
            raise InvariantViolation(
                f"способ {self.method.value} обязан нести уверенность",
                context={"method": self.method.value},
            )
        if self.method is ExtractionMethod.TEXT_LAYER and self.illegible_spans:
            raise InvariantViolation(
                "текстовый слой не даёт неразборчивых фрагментов",
                context={"spans": len(self.illegible_spans)},
            )
        if self.method is ExtractionMethod.NONE and self.content:
            raise InvariantViolation(
                "страница без способа извлечения не может иметь текст",
                context={"length": len(self.content)},
            )

    def _validate_spans(self) -> None:
        previous_end = 0
        for span in self.illegible_spans:
            if not span.span.within(len(self.content)):
                raise InvalidTextSpan(
                    "диапазон выходит за пределы текста страницы",
                    context={"end": span.span.end, "content_length": len(self.content)},
                )
            if span.span.start < previous_end:
                raise InvariantViolation(
                    "диапазоны неразборчивости пересекаются или не отсортированы",
                    context={"start": span.span.start, "previous_end": previous_end},
                )
            expected = "" if span.span.is_empty else span.span.slice_of(self.content)
            if span.raw_text != expected:
                raise FabricatedTextDetected(
                    "сохранённый фрагмент не совпадает со срезом текста страницы",
                    context={"raw_text": span.raw_text, "slice": expected},
                )
            previous_end = span.span.end

    @classmethod
    def nothing_recognized(
        cls,
        *,
        method: ExtractionMethod,
        reason: IllegibleReason,
    ) -> Self:
        """Страница, на которой распознаватель не выдал ничего.

        Raises:
            InvariantViolation: Для текстового слоя такой ситуации не бывает.
        """
        if method is ExtractionMethod.TEXT_LAYER:
            raise InvariantViolation(
                "текстовый слой не может ничего не распознать",
                context={"method": method.value},
            )
        confidence = None if method is ExtractionMethod.NONE else OcrConfidence.ZERO
        return cls(
            content="",
            method=method,
            confidence=confidence,
            illegible_spans=(
                IllegibleSpan(
                    span=TextSpan(0, 0),
                    confidence=OcrConfidence.ZERO,
                    reason=reason,
                    raw_text="",
                ),
            ),
        )

    @property
    def has_illegible(self) -> bool:
        """Есть ли неразборчивые фрагменты."""
        return bool(self.illegible_spans)

    @property
    def char_count(self) -> int:
        """Длина текста в кодовых точках."""
        return len(self.content)

    @property
    def illegible_char_count(self) -> int:
        """Сколько символов покрыто неразборчивыми фрагментами."""
        return sum(span.span.length for span in self.illegible_spans)

    @property
    def illegible_ratio(self) -> float:
        """Доля неразборчивого текста; для пустой страницы — ноль."""
        if not self.content:
            return 0.0
        return self.illegible_char_count / len(self.content)
