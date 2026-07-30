"""Политика итогового статуса документа — единственный её источник.

Ни агрегат, ни use case, ни маппер статус не вычисляют: `Document.complete()`
принимает готовый вердикт.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.domain.value_objects.enums import DocumentStatus, PageStatus
from document_worker.domain.value_objects.quality import (
    DocumentQualityStats,
    DocumentStatusVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.value_objects.paging import PageNumber
    from document_worker.domain.value_objects.quality import PageOutcome

REASON_EMPTY_DOCUMENT = "empty_document"
REASON_INCOMPLETE_PAGE_SET = "incomplete_page_set"
REASON_TOO_MANY_FAILED_PAGES = "too_many_failed_pages"
REASON_TOO_FEW_USABLE_PAGES = "too_few_usable_pages"
REASON_NO_EXTRACTABLE_TEXT = "no_extractable_text"
REASON_ILLEGIBLE_PAGES = "illegible_pages"
REASON_FAILED_PAGES = "failed_pages"
REASON_LOW_MEAN_CONFIDENCE = "low_mean_confidence"


@dataclass(frozen=True, slots=True)
class DocumentStatusPolicy:
    """Считает итоговый статус документа по итогам его страниц."""

    max_failed_page_ratio: float = 0.50
    min_usable_page_ratio: float = 0.30
    min_document_mean_confidence: float = 0.60
    min_extracted_chars: int = 200

    def evaluate(
        self,
        outcomes: Sequence[PageOutcome],
        *,
        declared_page_count: int | None,
    ) -> DocumentStatusVerdict:
        """Возвращает вердикт о терминальном статусе документа."""
        if not outcomes or not declared_page_count:
            return self._failed_without_stats(REASON_EMPTY_DOCUMENT, outcomes)

        stats = DocumentQualityStats.from_outcomes(outcomes)
        pages = _pages_by_status(outcomes)

        failure = self._first_failure(stats, len(outcomes), declared_page_count)
        if failure is not None:
            return _verdict(DocumentStatus.FAILED, stats, (failure,), pages)

        if self._is_fully_processed(stats):
            return _verdict(DocumentStatus.PROCESSED, stats, (), _NO_PAGES)

        return _verdict(
            DocumentStatus.PARTIALLY_PROCESSED,
            stats,
            self._degradation_reasons(stats),
            pages,
        )

    def _failed_without_stats(
        self,
        reason: str,
        outcomes: Sequence[PageOutcome],
    ) -> DocumentStatusVerdict:
        # Снимок качества требует хотя бы одной страницы, поэтому для пустого
        # документа он строится из заглушки.
        stats = (
            DocumentQualityStats.from_outcomes(outcomes)
            if outcomes
            else _EMPTY_DOCUMENT_STATS
        )
        return _verdict(
            DocumentStatus.FAILED, stats, (reason,), _pages_by_status(outcomes)
        )

    def _first_failure(
        self,
        stats: DocumentQualityStats,
        page_count: int,
        declared_page_count: int,
    ) -> str | None:
        if page_count != declared_page_count:
            return REASON_INCOMPLETE_PAGE_SET
        if stats.failed_page_ratio >= self.max_failed_page_ratio:
            return REASON_TOO_MANY_FAILED_PAGES
        if stats.usable_page_ratio < self.min_usable_page_ratio:
            return REASON_TOO_FEW_USABLE_PAGES
        if stats.total_chars < self.min_extracted_chars:
            return REASON_NO_EXTRACTABLE_TEXT
        return None

    def _is_fully_processed(self, stats: DocumentQualityStats) -> bool:
        # Устав читается буквально: 99 прочитанных страниц из 100 дают
        # частичную обработку, а не полную.
        if stats.pages_failed_status or stats.pages_illegible:
            return False
        if stats.pages_partially_illegible:
            return False
        confidence = stats.mean_ocr_confidence
        return confidence is None or not confidence.is_below(
            self.min_document_mean_confidence
        )

    def _degradation_reasons(self, stats: DocumentQualityStats) -> tuple[str, ...]:
        reasons: list[str] = []
        if stats.pages_illegible or stats.pages_partially_illegible:
            reasons.append(REASON_ILLEGIBLE_PAGES)
        if stats.pages_failed_status:
            reasons.append(REASON_FAILED_PAGES)
        confidence = stats.mean_ocr_confidence
        if confidence is not None and confidence.is_below(
            self.min_document_mean_confidence
        ):
            reasons.append(REASON_LOW_MEAN_CONFIDENCE)
        return tuple(reasons)


class _PagesByStatus:
    """Номера страниц, разложенные по категориям проблем."""

    __slots__ = ("failed", "illegible", "partially_illegible")

    def __init__(
        self,
        partially_illegible: tuple[PageNumber, ...],
        illegible: tuple[PageNumber, ...],
        failed: tuple[PageNumber, ...],
    ) -> None:
        self.partially_illegible = partially_illegible
        self.illegible = illegible
        self.failed = failed


_NO_PAGES = _PagesByStatus((), (), ())

_EMPTY_DOCUMENT_STATS = DocumentQualityStats(
    pages_total=0,
    pages_text_layer=0,
    pages_ocr=0,
    pages_hybrid=0,
    pages_failed=0,
    pages_extracted=0,
    pages_partially_illegible=0,
    pages_illegible=0,
    pages_failed_status=0,
    total_chars=0,
    illegible_chars=0,
    mean_ocr_confidence=None,
)


def _pages_by_status(outcomes: Sequence[PageOutcome]) -> _PagesByStatus:
    def numbers(status: PageStatus) -> tuple[PageNumber, ...]:
        return tuple(
            outcome.page_number for outcome in outcomes if outcome.status is status
        )

    return _PagesByStatus(
        partially_illegible=numbers(PageStatus.PARTIALLY_ILLEGIBLE),
        illegible=numbers(PageStatus.ILLEGIBLE),
        failed=numbers(PageStatus.FAILED),
    )


def _verdict(
    status: DocumentStatus,
    stats: DocumentQualityStats,
    reasons: tuple[str, ...],
    pages: _PagesByStatus,
) -> DocumentStatusVerdict:
    return DocumentStatusVerdict(
        status=status,
        stats=stats,
        reasons=reasons,
        partially_illegible_pages=pages.partially_illegible,
        illegible_pages=pages.illegible,
        failed_pages=pages.failed,
    )
