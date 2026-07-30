"""Контекст логирования.

Источник истины для correlation_id — явная передача в командах. Эти
переменные существуют только чтобы обогащать записи лога.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_document_id: ContextVar[str | None] = ContextVar("document_id", default=None)


@contextmanager
def logging_context(
    *,
    correlation_id: str | None = None,
    document_id: str | None = None,
) -> Iterator[None]:
    """Подставляет значения в лог на время блока и всегда возвращает прежние."""
    correlation_token = _correlation_id.set(correlation_id)
    document_token = _document_id.set(document_id)
    try:
        yield
    finally:
        _correlation_id.reset(correlation_token)
        _document_id.reset(document_token)


def current_logging_context() -> dict[str, str]:
    """Текущие значения для обогащения записи лога."""
    values = {
        "correlation_id": _correlation_id.get(),
        "document_id": _document_id.get(),
    }
    return {key: value for key, value in values.items() if value is not None}
