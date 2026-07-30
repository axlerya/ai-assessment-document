"""Снимки качества обработки и вердикты политик.

Живут в value objects, а не в модулях политик: на них ссылаются и политика,
и агрегат документа, и события.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from document_worker.domain.errors import InvariantViolation
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    DocumentStatus,
    ExtractionMethod,
    PageStatus,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.value_objects.paging import PageNumber
    from document_worker.domain.value_objects.text import IllegibleSpan

_METHODS_WITHOUT_CONFIDENCE = frozenset(
    {ExtractionMethod.TEXT_LAYER, ExtractionMethod.NONE}
)


@dataclass(frozen=True, slots=True)
class PageOutcome:
    """Итог обработки одной страницы — вход для агрегатов документа."""

    page_number: PageNumber
    status: PageStatus
    method: ExtractionMethod
    confidence: OcrConfidence | None
    char_count: int
    illegible_char_count: int

    def __post_init__(self) -> None:
        """Сверяет способ, статус и счётчики символов."""
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
        # Делает разбиение страниц по способам и по статусам полным
        # и непересекающимся, поэтому счётчики события всегда сходятся.
        if (self.method is ExtractionMethod.NONE) != (self.status is PageStatus.FAILED):
            raise InvariantViolation(
                "отсутствие способа и статус failed это одно и то же состояние",
                context={"method": self.method.value, "status": self.status.value},
            )
        if self.char_count < 0 or self.illegible_char_count < 0:
            raise InvariantViolation(
                "счётчики символов отрицательны",
                context={
                    "chars": self.char_count,
                    "illegible": self.illegible_char_count,
                },
            )
        if self.illegible_char_count > self.char_count:
            raise InvariantViolation(
                "неразборчивых символов больше, чем всего",
                context={
                    "chars": self.char_count,
                    "illegible": self.illegible_char_count,
                },
            )


@dataclass(frozen=True, slots=True)
class DocumentQualityStats:
    """Агрегированное качество документа.

    Доли считаются на лету: как поля они могли бы разойтись со счётчиками.
    """

    pages_total: int
    pages_text_layer: int
    pages_ocr: int
    pages_hybrid: int
    pages_failed: int
    pages_extracted: int
    pages_partially_illegible: int
    pages_illegible: int
    pages_failed_status: int
    total_chars: int
    illegible_chars: int
    mean_ocr_confidence: OcrConfidence | None

    def __post_init__(self) -> None:
        """Проверяет, что оба разбиения страниц сходятся с общим числом."""
        by_method = (
            self.pages_text_layer
            + self.pages_ocr
            + self.pages_hybrid
            + self.pages_failed
        )
        if by_method != self.pages_total:
            raise InvariantViolation(
                "счётчики по способам извлечения не сходятся с числом страниц",
                context={"by_method": by_method, "total": self.pages_total},
            )
        by_status = (
            self.pages_extracted
            + self.pages_partially_illegible
            + self.pages_illegible
            + self.pages_failed_status
        )
        if by_status != self.pages_total:
            raise InvariantViolation(
                "счётчики по статусам не сходятся с числом страниц",
                context={"by_status": by_status, "total": self.pages_total},
            )

    @classmethod
    def from_outcomes(cls, outcomes: Sequence[PageOutcome]) -> Self:
        """Собирает снимок по итогам страниц.

        Raises:
            InvariantViolation: Пустой список или повторяющиеся номера страниц.
        """
        if not outcomes:
            raise InvariantViolation("документ без страниц не имеет снимка качества")
        numbers = [outcome.page_number for outcome in outcomes]
        if len(set(numbers)) != len(numbers):
            raise InvariantViolation(
                "номера страниц повторяются",
                context={"pages": len(numbers)},
            )

        by_method = Counter(outcome.method for outcome in outcomes)
        by_status = Counter(outcome.status for outcome in outcomes)
        ocr_pages = [
            (outcome.confidence, outcome.char_count)
            for outcome in outcomes
            if outcome.confidence is not None
        ]
        return cls(
            pages_total=len(outcomes),
            pages_text_layer=by_method[ExtractionMethod.TEXT_LAYER],
            pages_ocr=by_method[ExtractionMethod.OCR],
            pages_hybrid=by_method[ExtractionMethod.HYBRID],
            pages_failed=by_method[ExtractionMethod.NONE],
            pages_extracted=by_status[PageStatus.EXTRACTED],
            pages_partially_illegible=by_status[PageStatus.PARTIALLY_ILLEGIBLE],
            pages_illegible=by_status[PageStatus.ILLEGIBLE],
            pages_failed_status=by_status[PageStatus.FAILED],
            total_chars=sum(outcome.char_count for outcome in outcomes),
            illegible_chars=sum(outcome.illegible_char_count for outcome in outcomes),
            mean_ocr_confidence=OcrConfidence.weighted_mean(ocr_pages),
        )

    @property
    def ocr_coverage(self) -> float:
        """Доля страниц, прошедших через распознавание."""
        return (self.pages_ocr + self.pages_hybrid) / self.pages_total

    @property
    def usable_page_ratio(self) -> float:
        """Доля страниц, пригодных для чанкования."""
        return (
            self.pages_extracted + self.pages_partially_illegible
        ) / self.pages_total

    @property
    def failed_page_ratio(self) -> float:
        """Доля страниц, которые не удалось прочитать."""
        return self.pages_failed_status / self.pages_total

    @property
    def illegible_char_ratio(self) -> float:
        """Доля неразборчивого текста; для документа без текста — ноль."""
        if self.total_chars == 0:
            return 0.0
        return self.illegible_chars / self.total_chars


@dataclass(frozen=True, slots=True)
class PageLegibilityVerdict:
    """Решение о читаемости одной страницы."""

    status: PageStatus
    mean_confidence: OcrConfidence
    illegible_spans: tuple[IllegibleSpan, ...]
    illegible_ratio: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Не даёт объявить страницу читаемой при наличии неразборчивых мест."""
        if (self.status is PageStatus.EXTRACTED) != (not self.illegible_spans):
            raise InvariantViolation(
                "читаемость страницы противоречит списку неразборчивых фрагментов",
                context={
                    "status": self.status.value,
                    "spans": len(self.illegible_spans),
                },
            )


@dataclass(frozen=True, slots=True)
class DocumentStatusVerdict:
    """Решение о терминальном статусе документа.

    Проблемные страницы перечислены по категориям, а не одним списком: событие
    о частичной обработке требует именно такого разбиения.
    """

    status: DocumentStatus
    stats: DocumentQualityStats
    reasons: tuple[str, ...] = ()
    partially_illegible_pages: tuple[PageNumber, ...] = ()
    illegible_pages: tuple[PageNumber, ...] = ()
    failed_pages: tuple[PageNumber, ...] = ()

    @property
    def problem_pages(self) -> tuple[PageNumber, ...]:
        """Все страницы, из-за которых документ не полностью обработан."""
        return tuple(
            sorted(
                {
                    *self.partially_illegible_pages,
                    *self.illegible_pages,
                    *self.failed_pages,
                }
            )
        )

    def __post_init__(self) -> None:
        """Сверяет статус со списком проблемных страниц."""
        if not self.status.is_terminal:
            raise InvariantViolation(
                "вердикт объявляет только терминальный статус",
                context={"status": self.status.value},
            )
        # Низкое среднее confidence это причина без проблемных страниц: все
        # страницы прочитаны, но доверять документу как полному источнику нельзя.
        if self.status is DocumentStatus.PARTIALLY_PROCESSED and not (
            self.problem_pages or self.reasons
        ):
            raise InvariantViolation(
                "частичная обработка без единой причины бессмысленна",
                context={"status": self.status.value},
            )
        if self.status is DocumentStatus.PROCESSED and self.problem_pages:
            raise InvariantViolation(
                "полностью обработанный документ не имеет проблемных страниц",
                context={"pages": len(self.problem_pages)},
            )
