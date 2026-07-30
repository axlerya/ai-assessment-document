"""Тесты доменных идентификаторов."""

from __future__ import annotations

import uuid

import pytest

from document_worker.domain.errors import InvalidCorrelationId, InvalidIdentifier
from document_worker.domain.value_objects.identifiers import (
    ChunkId,
    CorrelationId,
    DocumentId,
    EventId,
    JobId,
    PageId,
)
from document_worker.domain.value_objects.versioning import PipelineVersion

pytestmark = pytest.mark.unit

ID_TYPES = [DocumentId, PageId, ChunkId, JobId, EventId]

DOCUMENT_ID = DocumentId(uuid.UUID("11111111-1111-5111-9111-111111111111"))
PIPELINE_VERSION = PipelineVersion(1, 0, 0)


@pytest.mark.parametrize("id_type", ID_TYPES)
def test_rejects_nil_uuid(id_type: type[DocumentId]) -> None:
    with pytest.raises(InvalidIdentifier):
        id_type(uuid.UUID(int=0))


def test_document_id_parse_rejects_non_uuid_string() -> None:
    with pytest.raises(InvalidIdentifier):
        DocumentId.parse("не-uuid")


def test_document_id_parse_accepts_canonical_form() -> None:
    raw = "6f1c0f8e-6a1e-5b2a-9f3c-2d4e5a6b7c81"

    assert str(DocumentId.parse(raw)) == raw


def test_generate_produces_distinct_identifiers() -> None:
    assert DocumentId.generate() != DocumentId.generate()


def test_page_id_and_chunk_id_are_not_interchangeable() -> None:
    value = uuid.UUID("22222222-2222-5222-9222-222222222222")

    assert PageId(value) != ChunkId(value)


def test_event_id_deterministic_is_stable_for_same_inputs() -> None:
    first = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processed",
    )
    second = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processed",
    )

    assert first == second


def test_event_id_deterministic_differs_for_other_pipeline_version() -> None:
    first = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processed",
    )
    second = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PipelineVersion(2, 0, 0),
        event_type="document.processed",
    )

    assert first != second


def test_event_id_deterministic_differs_for_other_event_type() -> None:
    processed = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processed",
    )
    failed = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processing.failed",
    )

    assert processed != failed


def test_event_id_deterministic_differs_for_other_document() -> None:
    other_document = DocumentId(uuid.UUID("33333333-3333-5333-9333-333333333333"))

    first = EventId.deterministic(
        document_id=DOCUMENT_ID,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processed",
    )
    second = EventId.deterministic(
        document_id=other_document,
        pipeline_version=PIPELINE_VERSION,
        event_type="document.processed",
    )

    assert first != second


@pytest.mark.parametrize(
    "raw",
    [
        "с пробелом",
        "",
        "короче8",
        "x" * 129,
        "с/слэшем",
        "кириллица",
        "with\ttab",
    ],
)
def test_correlation_id_rejects_invalid_value(raw: str) -> None:
    with pytest.raises(InvalidCorrelationId):
        CorrelationId(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "abcd1234",
        "req-2026-07-30",
        "trace:0af7651916cd43dd",
        "a.b_c-d:e",
        "x" * 128,
    ],
)
def test_correlation_id_accepts_valid_value(raw: str) -> None:
    assert str(CorrelationId(raw)) == raw


def test_identifier_is_immutable() -> None:
    document_id = DocumentId.generate()

    with pytest.raises(AttributeError):
        document_id.value = uuid.uuid4()  # type: ignore[misc]
