"""Карта смещений: где оказался каждый символ после нормализации.

Объект транзиентный — живёт на время обработки страницы и в БД не сохраняется.
Единица смещения — кодовая точка Unicode.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from document_worker.domain.errors import InvariantViolation
from document_worker.domain.value_objects.text import TextSpan

if TYPE_CHECKING:
    from collections.abc import Sequence


class RuleAction(StrEnum):
    """Закрытое множество действий над символом.

    Действий «дополнить», «предположить», «восстановить» здесь нет и появиться
    они могут только расширением этого перечисления — что ловит тест.
    """

    KEEP = "keep"
    DROP = "drop"
    MAP = "map"
    UNFOLD = "unfold"
    COLLAPSE = "collapse"


@dataclass(frozen=True, slots=True)
class OffsetSegment:
    """Кусок исходного текста и его место в результате."""

    source_start: int
    source_end: int
    target_start: int
    target_end: int
    action: RuleAction

    @property
    def source_length(self) -> int:
        """Длина куска в исходном тексте."""
        return self.source_end - self.source_start

    @property
    def target_length(self) -> int:
        """Длина куска в результате."""
        return self.target_end - self.target_start


@dataclass(frozen=True, slots=True)
class OffsetMap:
    """Соответствие между исходным и нормализованным текстом."""

    source_length: int
    target_length: int
    segments: tuple[OffsetSegment, ...]
    _source_starts: tuple[int, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        """Проверяет покрытие, монотонность и отсутствие вставок."""
        self._validate()
        object.__setattr__(
            self,
            "_source_starts",
            tuple(segment.source_start for segment in self.segments),
        )

    def _validate(self) -> None:
        source_cursor = 0
        target_cursor = 0
        for segment in self.segments:
            if segment.source_start != source_cursor:
                raise InvariantViolation(
                    "сегменты карты смещений не покрывают исходный текст",
                    context={
                        "expected": source_cursor,
                        "actual": segment.source_start,
                    },
                )
            if segment.source_length <= 0:
                raise InvariantViolation(
                    "вставка текста из ниоткуда непредставима",
                    context={"source_start": segment.source_start},
                )
            if segment.target_start != target_cursor or segment.target_length < 0:
                raise InvariantViolation(
                    "сегменты карты смещений не покрывают результат",
                    context={
                        "expected": target_cursor,
                        "actual": segment.target_start,
                    },
                )
            if segment.action is RuleAction.DROP and segment.target_length != 0:
                raise InvariantViolation(
                    "удалённый фрагмент не может занимать место в результате",
                    context={"target_length": segment.target_length},
                )
            source_cursor = segment.source_end
            target_cursor = segment.target_end

        if source_cursor != self.source_length or target_cursor != self.target_length:
            raise InvariantViolation(
                "карта смещений обрывается раньше конца текста",
                context={
                    "source": f"{source_cursor}/{self.source_length}",
                    "target": f"{target_cursor}/{self.target_length}",
                },
            )

    @classmethod
    def identity(cls, length: int) -> Self:
        """Карта текста, который не изменился."""
        segments = (
            (OffsetSegment(0, length, 0, length, RuleAction.KEEP),) if length else ()
        )
        return cls(source_length=length, target_length=length, segments=segments)

    def project_offset(self, position: int) -> int:
        """Переводит позицию исходного текста в позицию результата.

        Позиция внутри удалённого куска схлопывается к его левому краю.
        """
        if position <= 0:
            return 0
        if position >= self.source_length:
            return self.target_length
        segment = self.segments[bisect_right(self._source_starts, position) - 1]
        if segment.source_length == segment.target_length:
            return segment.target_start + (position - segment.source_start)
        return segment.target_start

    def project_span(self, span: TextSpan) -> TextSpan | None:
        """Переводит диапазон; возвращает None, если он уничтожен целиком."""
        projected = TextSpan(
            self.project_offset(span.start), self.project_offset(span.end)
        )
        if projected.is_empty and not span.is_empty:
            return None
        return projected

    def compose(self, other: OffsetMap) -> OffsetMap:
        """Склеивает две карты в одну.

        Raises:
            InvariantViolation: Результат первой карты не совпадает с входом второй.
        """
        if self.target_length != other.source_length:
            raise InvariantViolation(
                "карты смещений не стыкуются",
                context={
                    "target": self.target_length,
                    "source": other.source_length,
                },
            )
        builder = OffsetMapBuilder()
        for segment in self.segments:
            _compose_segment(builder, segment, other)
        return builder.build()


def _compose_segment(
    builder: OffsetMapBuilder,
    segment: OffsetSegment,
    other: OffsetMap,
) -> None:
    """Переносит один сегмент через вторую карту.

    Кусок один к одному разбирается посимвольно: иначе удаление внутри него
    сдвинуло бы всё, что идёт следом.
    """
    if segment.target_length == 0:
        builder.add(source=segment.source_length, target=0, action=RuleAction.DROP)
        return
    if segment.source_length == segment.target_length:
        for offset in range(segment.source_length):
            position = segment.target_start + offset
            length = other.project_offset(position + 1) - other.project_offset(position)
            builder.add(
                source=1,
                target=length,
                action=segment.action if length else RuleAction.DROP,
            )
        return
    start = other.project_offset(segment.target_start)
    end = other.project_offset(segment.target_end)
    builder.add(
        source=segment.source_length,
        target=end - start,
        action=segment.action if end > start else RuleAction.DROP,
    )


class OffsetMapBuilder:
    """Накапливает сегменты по мере обработки текста правилом."""

    __slots__ = ("_segments", "_source_cursor", "_target_cursor")

    def __init__(self) -> None:
        """Создаёт пустой накопитель."""
        self._segments: list[OffsetSegment] = []
        self._source_cursor = 0
        self._target_cursor = 0

    def add(self, *, source: int, target: int, action: RuleAction) -> None:
        """Добавляет кусок длиной `source`, занявший `target` в результате."""
        if source <= 0:
            return
        self._segments.append(
            OffsetSegment(
                source_start=self._source_cursor,
                source_end=self._source_cursor + source,
                target_start=self._target_cursor,
                target_end=self._target_cursor + target,
                action=action,
            )
        )
        self._source_cursor += source
        self._target_cursor += target

    def build(self) -> OffsetMap:
        """Собирает карту из накопленных сегментов."""
        return OffsetMap(
            source_length=self._source_cursor,
            target_length=self._target_cursor,
            segments=tuple(_merged(self._segments)),
        )


def _merged(segments: Sequence[OffsetSegment]) -> list[OffsetSegment]:
    """Склеивает соседние куски с одинаковым действием и равными длинами."""
    merged: list[OffsetSegment] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        mergeable = (
            previous is not None
            and previous.action is segment.action
            and previous.source_length == previous.target_length
            and segment.source_length == segment.target_length
        )
        if previous is not None and mergeable:
            merged[-1] = OffsetSegment(
                source_start=previous.source_start,
                source_end=segment.source_end,
                target_start=previous.target_start,
                target_end=segment.target_end,
                action=previous.action,
            )
        else:
            merged.append(segment)
    return merged
