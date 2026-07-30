"""Маркер неразборчивого фрагмента — единственный формат на весь проект.

Маркер это представление, а не хранение: в каноническом тексте страницы его
нет и быть не может. Рендерится только на выдаче.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.text import TextSpan

if TYPE_CHECKING:
    from document_worker.domain.value_objects.text import IllegibleSpan

MARKER_TEMPLATE = "[НЕРАЗБОРЧИВО: {locator}, confidence={confidence:.2f}]"
LOCATOR_BY_LINE = "строка {line}"
LOCATOR_BY_CHARS = "символы {start}-{end}"

MARKER_RE: re.Pattern[str] = re.compile(
    r"\[НЕРАЗБОРЧИВО: "
    r"(?:строка (?P<line>\d+)|символы (?P<start>\d+)-(?P<end>\d+))"
    r", confidence=(?P<confidence>\d\.\d{2})\]"
)


@dataclass(frozen=True, slots=True)
class IllegibleMarker:
    """Пометка о нераспознанном фрагменте.

    Рендер использует номер строки, если он известен, иначе диапазон символов.
    Поэтому `parse(render())` возвращает исходный маркер только когда вторая
    координата в нейтральном состоянии: пустой диапазон либо `line_number=None`.
    """

    line_number: int | None
    span: TextSpan
    confidence: OcrConfidence

    @classmethod
    def of(cls, span: IllegibleSpan) -> Self:
        """Строит маркер по неразборчивому фрагменту."""
        return cls(
            line_number=span.line_number,
            span=span.span,
            confidence=span.confidence,
        )

    def render(self) -> str:
        """Собирает строку маркера."""
        if self.line_number is not None:
            locator = LOCATOR_BY_LINE.format(line=self.line_number)
        else:
            locator = LOCATOR_BY_CHARS.format(start=self.span.start, end=self.span.end)
        return MARKER_TEMPLATE.format(locator=locator, confidence=self.confidence.value)

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Разбирает строку маркера обратно.

        Raises:
            ValueError: Строка не является маркером.
        """
        match = MARKER_RE.fullmatch(raw)
        if match is None:
            msg = f"{raw!r} не является маркером неразборчивого фрагмента"
            raise ValueError(msg)
        confidence = OcrConfidence(float(match.group("confidence")))
        line = match.group("line")
        if line is not None:
            return cls(
                line_number=int(line),
                span=TextSpan(0, 0),
                confidence=confidence,
            )
        return cls(
            line_number=None,
            span=TextSpan(int(match.group("start")), int(match.group("end"))),
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class MarkedText:
    """Текст страницы вместе с его неразборчивыми фрагментами."""

    text: str
    spans: tuple[IllegibleSpan, ...]

    def render(self) -> str:
        """Подставляет маркеры вместо срезов неразборчивых фрагментов.

        Замены идут справа налево, поэтому смещения ещё не обработанных
        фрагментов остаются валидными. Фрагмент нулевой длины вставляет маркер
        в позицию, ничего не удаляя.
        """
        rendered = self.text
        for span in sorted(self.spans, key=lambda item: item.span.start, reverse=True):
            marker = IllegibleMarker.of(span).render()
            rendered = rendered[: span.span.start] + marker + rendered[span.span.end :]
        return rendered

    @staticmethod
    def strip_markers(rendered: str) -> str:
        """Убирает все маркеры из отрендеренного текста."""
        return MARKER_RE.sub("", rendered)
