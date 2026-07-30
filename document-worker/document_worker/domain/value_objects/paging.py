"""Номер страницы документа."""

from __future__ import annotations

from dataclasses import dataclass

from document_worker.domain.constants import MAX_PAGES, MIN_PAGE_NUMBER
from document_worker.domain.errors import InvalidPageNumber


@dataclass(frozen=True, slots=True, order=True)
class PageNumber:
    """Номер страницы, нумерация с единицы."""

    value: int

    def __post_init__(self) -> None:
        """Проверяет границы номера."""
        if not MIN_PAGE_NUMBER <= self.value <= MAX_PAGES:
            raise InvalidPageNumber(
                f"номер страницы вне {MIN_PAGE_NUMBER}..{MAX_PAGES}",
                context={"value": self.value, "limit": MAX_PAGES},
            )

    def __int__(self) -> int:
        """Возвращает номер как int."""
        return self.value

    def next(self) -> PageNumber:
        """Следующий номер страницы."""
        return PageNumber(self.value + 1)
