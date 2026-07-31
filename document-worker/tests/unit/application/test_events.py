"""Перевод доменных событий в строки outbox.

В полезной нагрузке обязаны остаться только примитивы JSON: значения-объекты
и перечисления домена сериализатор шины не понимает.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from document_worker.application.events import to_outbox_event
from document_worker.domain.events import DocumentProcessed, DocumentProcessingFailed
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ProcessingStage
from document_worker.domain.value_objects.identifiers import DocumentId
from document_worker.domain.value_objects.versioning import PipelineVersion
from tests.factories import new_correlation_id

if TYPE_CHECKING:
    from document_worker.domain.value_objects.identifiers import CorrelationId

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
VERSION = PipelineVersion(1, 0, 0)


def _processed(
    *,
    document_id: DocumentId | None = None,
    correlation_id: CorrelationId | None = None,
) -> DocumentProcessed:
    return DocumentProcessed(
        document_id=document_id or DocumentId.generate(),
        correlation_id=correlation_id or new_correlation_id(),
        pipeline_version=VERSION,
        occurred_at=NOW,
        pages_total=2,
        pages_text_layer=1,
        pages_ocr=1,
        pages_hybrid=0,
        pages_failed=0,
        chunks_total=7,
        total_chars=4096,
        mean_ocr_confidence=OcrConfidence(0.9312),
        ocr_coverage=0.5,
        processing_duration_ms=1200,
    )


def test_event_type_is_also_the_routing_key() -> None:
    # Второго источника имени не заводится: расхождение типа и ключа означало
    # бы событие, которое некому доставить.
    record = to_outbox_event(_processed())

    assert record.event_type == DocumentProcessed.event_type
    assert record.routing_key == record.event_type


def test_event_id_repeats_for_the_same_document_and_version() -> None:
    document_id = DocumentId.generate()

    first = to_outbox_event(_processed(document_id=document_id))
    second = to_outbox_event(_processed(document_id=document_id))

    assert first.event_id == second.event_id


def test_payload_is_plain_json() -> None:
    record = to_outbox_event(_processed())

    assert json.loads(json.dumps(record.payload)) == record.payload


def test_confidence_becomes_a_number() -> None:
    # Строкой её не сравнить и не усреднить на стороне потребителя.
    record = to_outbox_event(_processed())

    assert record.payload["mean_ocr_confidence"] == pytest.approx(0.9312)


def test_moments_become_strings_with_their_zone() -> None:
    record = to_outbox_event(_processed())

    assert record.payload["occurred_at"] == NOW.isoformat()


def test_identifiers_and_version_become_strings() -> None:
    document_id = DocumentId.generate()
    correlation_id = new_correlation_id()

    record = to_outbox_event(
        _processed(document_id=document_id, correlation_id=correlation_id)
    )

    assert record.payload["document_id"] == str(document_id)
    assert record.payload["correlation_id"] == correlation_id.value
    assert record.payload["pipeline_version"] == str(VERSION)


def test_stage_of_a_failure_becomes_its_value() -> None:
    event = DocumentProcessingFailed(
        document_id=DocumentId.generate(),
        correlation_id=new_correlation_id(),
        pipeline_version=VERSION,
        occurred_at=NOW,
        error_code="corrupted_document",
        error_message="файл не читается",
        stage=ProcessingStage.TEXT_EXTRACTION,
        attempt=2,
        pages_total=None,
        pages_persisted=0,
    )

    record = to_outbox_event(event)

    assert record.payload["stage"] == ProcessingStage.TEXT_EXTRACTION.value
    assert record.payload["pages_total"] is None
