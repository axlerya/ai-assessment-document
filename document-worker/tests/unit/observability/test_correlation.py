"""Сквозной идентификатор в контексте логов и настройка логирования."""

from __future__ import annotations

import pytest
import structlog

from document_worker.observability.correlation import (
    FIELD_CORRELATION_ID,
    message_context,
)
from document_worker.observability.logging import (
    UnsafeLogLevelError,
    configure_logging,
)

pytestmark = pytest.mark.unit

CORRELATION_ID = "0f0c2a4e-4f0f-4a3e-9d3a-2c9a0f0c2a4e"


def _bound() -> dict[str, object]:
    return dict(structlog.contextvars.get_contextvars())


def test_identifiers_are_bound_inside_the_block() -> None:
    with message_context(
        correlation_id=CORRELATION_ID,
        document_id="doc",
        event_id="evt",
        attempt=2,
    ):
        assert _bound()[FIELD_CORRELATION_ID] == CORRELATION_ID


def test_context_is_cleared_between_messages() -> None:
    # Оставшийся идентификатор приписал бы чужие записи предыдущему документу,
    # и расследование инцидента пошло бы по ложному следу.
    with message_context(
        correlation_id=CORRELATION_ID, document_id="doc", event_id="evt", attempt=1
    ):
        pass

    assert FIELD_CORRELATION_ID not in _bound()


def test_context_is_cleared_even_when_handling_fails() -> None:
    with (
        pytest.raises(RuntimeError),
        message_context(
            correlation_id=CORRELATION_ID,
            document_id="doc",
            event_id="evt",
            attempt=1,
        ),
    ):
        raise RuntimeError("обработка упала")

    assert FIELD_CORRELATION_ID not in _bound()


def test_logging_is_configured_without_error() -> None:
    configure_logging(level="INFO", environment="local")

    assert structlog.get_logger() is not None


def test_configuring_debug_in_production_is_refused() -> None:
    with pytest.raises(UnsafeLogLevelError):
        configure_logging(level="DEBUG", environment="production")
