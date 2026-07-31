"""Сквозной идентификатор запроса в контексте логов.

Контекст обязан очищаться между сообщениями: корутина обработчика переиспользует
задачу, и оставшийся идентификатор приписал бы чужие записи предыдущему
документу — расследование инцидента пошло бы по ложному следу.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator

FIELD_CORRELATION_ID = "correlation_id"
FIELD_DOCUMENT_ID = "document_id"
FIELD_EVENT_ID = "event_id"
FIELD_ATTEMPT = "attempt"


@contextlib.contextmanager
def message_context(
    *,
    correlation_id: str,
    document_id: str,
    event_id: str,
    attempt: int,
) -> Iterator[None]:
    """Привязывает идентификаторы сообщения к логам внутри блока."""
    structlog.contextvars.bind_contextvars(
        **{
            FIELD_CORRELATION_ID: correlation_id,
            FIELD_DOCUMENT_ID: document_id,
            FIELD_EVENT_ID: event_id,
            FIELD_ATTEMPT: attempt,
        }
    )
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(
            FIELD_CORRELATION_ID,
            FIELD_DOCUMENT_ID,
            FIELD_EVENT_ID,
            FIELD_ATTEMPT,
        )
