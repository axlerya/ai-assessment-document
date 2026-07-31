"""Перевод сообщения в команду прикладного слоя.

Через границу проходит dataclass, а не модель Pydantic: иначе тип транспорта
протёк бы в use case и потянул за собой всю библиотеку.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from document_worker.presentation.messaging.contracts.commands import (
    ProcessDocumentRequestedV1,
)
from document_worker.presentation.messaging.mappers import to_process_document_command

pytestmark = pytest.mark.unit

DOCUMENT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
DEFAULT_BUCKET = "documents"


def _message(**overrides: object) -> ProcessDocumentRequestedV1:
    payload: dict[str, object] = {
        "event_id": str(uuid.uuid4()),
        "document_id": str(DOCUMENT_ID),
        "object_key": f"documents/{DOCUMENT_ID}/source.pdf",
        "mime_type": "application/pdf",
        "occurred_at": NOW.isoformat(),
    }
    payload.update(overrides)
    return ProcessDocumentRequestedV1.model_validate(payload)


def test_command_carries_what_the_message_said() -> None:
    message = _message()

    command = to_process_document_command(
        message, default_bucket=DEFAULT_BUCKET, attempt=1, max_attempts=5
    )

    assert command.document_id.value == DOCUMENT_ID
    assert command.event_id.value == message.event_id
    assert command.correlation_id.value == str(message.correlation_id)
    assert command.object_ref.key == message.object_key
    assert command.mime_type.value == "application/pdf"
    assert command.occurred_at == NOW


def test_default_bucket_is_used_when_the_message_omits_it() -> None:
    command = to_process_document_command(
        _message(), default_bucket=DEFAULT_BUCKET, attempt=1, max_attempts=5
    )

    assert command.object_ref.bucket == DEFAULT_BUCKET


def test_bucket_from_the_message_wins() -> None:
    command = to_process_document_command(
        _message(bucket="legal-archive"),
        default_bucket=DEFAULT_BUCKET,
        attempt=1,
        max_attempts=5,
    )

    assert command.object_ref.bucket == "legal-archive"


def test_attempt_and_its_budget_travel_into_the_command() -> None:
    # Решение об исчерпании бюджета принимает application, а не подписчик.
    command = to_process_document_command(
        _message(), default_bucket=DEFAULT_BUCKET, attempt=5, max_attempts=5
    )

    assert command.attempt == 5
    assert command.is_last_attempt


def test_command_before_the_last_attempt_has_budget_left() -> None:
    command = to_process_document_command(
        _message(), default_bucket=DEFAULT_BUCKET, attempt=4, max_attempts=5
    )

    assert not command.is_last_attempt
