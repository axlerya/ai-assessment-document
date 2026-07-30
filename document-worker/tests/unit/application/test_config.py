"""Тесты согласованности конфигурации обработки."""

from __future__ import annotations

import pytest

from document_worker.application.config import (
    OutboxConfig,
    ProcessingConfig,
    SourceConfig,
    TransactionConfig,
)
from document_worker.domain.constants import MAX_FILE_SIZE_BYTES, MAX_PAGES
from document_worker.domain.errors import InvariantViolation
from document_worker.domain.value_objects.versioning import PipelineVersion

pytestmark = pytest.mark.unit


def _processing(**overrides: object) -> ProcessingConfig:
    defaults: dict[str, object] = {
        "pipeline_version": PipelineVersion(1, 0, 0),
        "consumer_name": "document-worker-1",
        "document_timeout_s": 3600.0,
    }
    defaults.update(overrides)
    return ProcessingConfig(**defaults)  # type: ignore[arg-type]


def test_source_defaults_match_domain_limits() -> None:
    source = SourceConfig()

    assert source.max_file_size_bytes == MAX_FILE_SIZE_BYTES
    assert source.max_pages == MAX_PAGES


def test_source_rejects_limit_above_domain_maximum() -> None:
    with pytest.raises(InvariantViolation):
        SourceConfig(max_pages=MAX_PAGES + 1)


def test_source_rejects_file_size_above_domain_maximum() -> None:
    with pytest.raises(InvariantViolation):
        SourceConfig(max_file_size_bytes=MAX_FILE_SIZE_BYTES + 1)


def test_source_supports_only_pdf() -> None:
    assert SourceConfig().supported_mime_types == frozenset({"application/pdf"})


def test_transaction_timeouts_are_positive() -> None:
    with pytest.raises(InvariantViolation):
        TransactionConfig(claim_ms=0)


def test_outbox_backoff_cap_is_not_below_base() -> None:
    with pytest.raises(InvariantViolation):
        OutboxConfig(backoff_base_s=10.0, backoff_cap_s=1.0)


def test_outbox_batch_size_is_positive() -> None:
    with pytest.raises(InvariantViolation):
        OutboxConfig(batch_size=0)


def test_processing_requires_positive_document_timeout() -> None:
    with pytest.raises(InvariantViolation):
        _processing(document_timeout_s=0.0)


def test_processing_config_is_frozen() -> None:
    config = _processing()

    with pytest.raises(AttributeError):
        config.consumer_name = "другой"  # type: ignore[misc]


def test_processing_config_carries_nested_sections() -> None:
    config = _processing()

    assert config.source.max_pages == MAX_PAGES
    assert config.tx.claim_ms > 0
    assert config.outbox.batch_size > 0
