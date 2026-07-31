"""Сборка ASGI-приложения: маршруты проб."""

from __future__ import annotations

import pytest

from document_worker.bootstrap.app import (
    HTTP_OK,
    HTTP_UNAVAILABLE,
    LIVENESS_PATH,
    METRICS_PATH,
    READINESS_PATH,
    build_metrics_route,
    build_readiness_route,
    liveness_route,
)
from document_worker.observability.metrics import Metrics

pytestmark = pytest.mark.unit


def test_probes_live_on_their_own_paths() -> None:
    assert LIVENESS_PATH == "/health/live"
    assert READINESS_PATH == "/health/ready"
    assert METRICS_PATH == "/metrics"


def test_probe_routes_are_built() -> None:
    assert liveness_route is not None
    assert build_readiness_route([]) is not None
    assert build_metrics_route(Metrics()) is not None


def test_probe_answers_are_the_two_expected_codes() -> None:
    assert (HTTP_OK, HTTP_UNAVAILABLE) == (200, 503)
