"""Системные время и идентификаторы."""

from __future__ import annotations

from datetime import UTC

import pytest

from document_worker.application.ports.system import Clock, IdGenerator
from document_worker.infrastructure.system.clock import SystemClock, Uuid4IdGenerator

pytestmark = pytest.mark.unit


def test_adapters_satisfy_their_ports() -> None:
    assert isinstance(SystemClock(), Clock)
    assert isinstance(Uuid4IdGenerator(), IdGenerator)


def test_clock_answers_in_utc_with_a_zone() -> None:
    # Наивное время в базе означает молчаливый сдвиг на часовой пояс машины.
    moment = SystemClock().now()

    assert moment.tzinfo is not None
    assert moment.utcoffset() == UTC.utcoffset(None)


def test_clock_moves_forward() -> None:
    clock = SystemClock()

    assert clock.now() <= clock.now()


def test_generator_issues_distinct_identifiers() -> None:
    generator = Uuid4IdGenerator()

    assert generator.new_uuid() != generator.new_uuid()
