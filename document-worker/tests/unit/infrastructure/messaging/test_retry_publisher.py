"""Границы лестницы повторов."""

from __future__ import annotations

import pytest

from document_worker.infrastructure.messaging.retry_publisher import RetryPublisher
from document_worker.infrastructure.messaging.topology import (
    RETRY_LADDER,
    build_topology,
)

pytestmark = pytest.mark.unit

CONSUMER_TIMEOUT_MS = 7_200_000


@pytest.mark.parametrize("attempt", [0, len(RETRY_LADDER) + 1])
async def test_attempt_outside_the_ladder_is_rejected(attempt: int) -> None:
    # Ступени под такой номер нет, и публиковать копию некуда: молча уронить
    # её в никуда хуже, чем отказать.
    publisher = RetryPublisher(
        broker=None,  # type: ignore[arg-type]
        topology=build_topology(consumer_timeout_ms=CONSUMER_TIMEOUT_MS),
    )

    with pytest.raises(ValueError, match="вне лестницы"):
        await publisher.schedule(b"{}", {}, attempt=attempt)
