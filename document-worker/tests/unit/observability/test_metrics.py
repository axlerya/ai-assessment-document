"""Реестр метрик."""

from __future__ import annotations

import pytest

from document_worker.observability.metrics import (
    FORBIDDEN_LABELS,
    Metrics,
    ensure_labels_are_bounded,
)

pytestmark = pytest.mark.unit

DECLARED_SERIES = (
    "docworker_documents_processed_total",
    "docworker_pages_processed_total",
    "docworker_page_processing_duration_seconds",
    "docworker_messages_consumed_total",
    "docworker_messages_dlq_total",
    "docworker_errors_total",
    "docworker_outbox_pending",
    "docworker_outbox_lag_seconds",
)


def test_every_declared_series_is_exposed() -> None:
    metrics = Metrics()
    metrics.documents_processed.labels(status="processed").inc()
    metrics.pages_processed.labels(
        extraction_method="text_layer", status="extracted"
    ).inc()
    metrics.page_duration_seconds.labels(extraction_method="text_layer").observe(0.2)
    metrics.messages_consumed.labels(outcome="processed").inc()
    metrics.messages_dlq.labels(error_code="corrupted_document").inc()
    metrics.errors.labels(error_type="StorageUnavailableError", error_code="x").inc()
    metrics.outbox_pending.set(3)
    metrics.outbox_lag_seconds.set(12.5)

    rendered = metrics.render().decode()

    for series in DECLARED_SERIES:
        assert series in rendered


def test_no_series_carries_an_unbounded_label() -> None:
    # Идентификатор документа в лейбле — это отдельный временной ряд на каждый
    # документ: хранилище метрик умирает на первой же тысяче.
    ensure_labels_are_bounded(Metrics().label_names())


@pytest.mark.parametrize("label", sorted(FORBIDDEN_LABELS))
def test_unbounded_label_is_rejected(label: str) -> None:
    with pytest.raises(ValueError, match="кардинальности"):
        ensure_labels_are_bounded([label, "status"])


def test_registries_do_not_share_state() -> None:
    # Общий реестр ломает тесты повторной регистрацией и смешивает счётчики.
    first = Metrics()
    first.outbox_pending.set(7)

    assert "docworker_outbox_pending 7.0" not in Metrics().render().decode()
