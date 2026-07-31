"""Входящий контракт команды обработки.

Всё, что не прошло здесь, до базы не доходит: сообщение отвергается разбором
и уходит в DLQ. Поэтому проверки тут про форму сообщения, а не про предметную
область — список поддерживаемых типов файлов живёт в доменной политике.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from document_worker.domain.constants import MAX_OBJECT_KEY_LENGTH
from document_worker.presentation.messaging.contracts.commands import (
    ProcessDocumentRequestedV1,
)

pytestmark = pytest.mark.unit

DOCUMENT_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": str(uuid.uuid4()),
        "document_id": str(DOCUMENT_ID),
        "object_key": f"documents/{DOCUMENT_ID}/source.pdf",
        "mime_type": "application/pdf",
        "occurred_at": NOW.isoformat(),
    }
    payload.update(overrides)
    return payload


def test_valid_message_is_parsed() -> None:
    message = ProcessDocumentRequestedV1.model_validate(_payload())

    assert message.document_id == DOCUMENT_ID
    assert message.object_key == f"documents/{DOCUMENT_ID}/source.pdf"


def test_extra_unknown_field_is_accepted() -> None:
    # Продюсер вправе добавить поле, и это не должно ломать потребителя.
    message = ProcessDocumentRequestedV1.model_validate(
        _payload(priority="high", tenant="acme")
    )

    assert message.mime_type == "application/pdf"


def test_message_is_frozen() -> None:
    message = ProcessDocumentRequestedV1.model_validate(_payload())

    with pytest.raises(ValidationError):
        message.document_id = uuid.uuid4()  # type: ignore[misc]


def test_correlation_id_is_generated_when_absent() -> None:
    # Сообщение без сквозного идентификатора не отвергается: трассируемость
    # не покупается ценой потерянной работы.
    message = ProcessDocumentRequestedV1.model_validate(_payload())

    assert isinstance(message.correlation_id, uuid.UUID)


def test_correlation_id_from_the_message_is_kept() -> None:
    correlation_id = uuid.uuid4()

    message = ProcessDocumentRequestedV1.model_validate(
        _payload(correlation_id=str(correlation_id))
    )

    assert message.correlation_id == correlation_id


def test_occurred_at_is_normalized_to_utc() -> None:
    # Расхождение часов продюсера сообщение не отклоняет — только приводится.
    moment = datetime(2026, 7, 31, 12, 0, tzinfo=timezone(timedelta(hours=3)))

    message = ProcessDocumentRequestedV1.model_validate(
        _payload(occurred_at=moment.isoformat())
    )

    assert message.occurred_at == NOW
    assert message.occurred_at.tzinfo == UTC


def test_occurred_at_without_a_zone_is_rejected() -> None:
    # Момент без зоны нечем сравнить с моментами базы, которые все в UTC.
    with pytest.raises(ValidationError):
        ProcessDocumentRequestedV1.model_validate(
            _payload(occurred_at="2026-07-31T09:00:00")
        )


@pytest.mark.parametrize(
    "key",
    [
        "documents/{document_id}/../../etc/passwd",
        "documents/{document_id}/..",
        "/documents/{document_id}/source.pdf",
        "documents/{document_id}//source.pdf",
        "documents/{document_id}/sub\\source.pdf",
    ],
)
def test_object_key_traversal_is_rejected(key: str) -> None:
    # Ключ приходит извне, и обработать по нему чужой файл значит записать в
    # документ содержимое другого.
    with pytest.raises(ValidationError):
        ProcessDocumentRequestedV1.model_validate(
            _payload(object_key=key.format(document_id=DOCUMENT_ID))
        )


def test_object_key_must_belong_to_the_document() -> None:
    other = uuid.uuid4()

    with pytest.raises(ValidationError):
        ProcessDocumentRequestedV1.model_validate(
            _payload(object_key=f"documents/{other}/source.pdf")
        )


def test_unknown_schema_version_is_rejected() -> None:
    # Повтор бессмыслен: новая версия кода не появится за сорок минут.
    with pytest.raises(ValidationError):
        ProcessDocumentRequestedV1.model_validate(_payload(schema_version=2))


@pytest.mark.parametrize("mime_type", ["application", "APPLICATION/PDF", "a/b/c", ""])
def test_syntactically_invalid_mime_type_is_rejected(mime_type: str) -> None:
    with pytest.raises(ValidationError):
        ProcessDocumentRequestedV1.model_validate(_payload(mime_type=mime_type))


def test_unsupported_but_well_formed_mime_type_is_accepted() -> None:
    # Список поддерживаемых типов — доменное правило, и в транспортном
    # контракте оно стало бы невидимым для остальных слоёв.
    message = ProcessDocumentRequestedV1.model_validate(_payload(mime_type="image/png"))

    assert message.mime_type == "image/png"


@pytest.mark.parametrize("bucket", ["ab", "Documents", "documents_1", "a" * 64])
def test_malformed_bucket_is_rejected(bucket: str) -> None:
    with pytest.raises(ValidationError):
        ProcessDocumentRequestedV1.model_validate(_payload(bucket=bucket))


def test_bucket_is_optional() -> None:
    message = ProcessDocumentRequestedV1.model_validate(_payload())

    assert message.bucket is None


def test_object_key_longer_than_the_storage_allows_is_rejected() -> None:
    # Предел длины ключа один на весь сервис и приходит из домена.
    tail = "a" * MAX_OBJECT_KEY_LENGTH

    with pytest.raises(ValidationError):
        ProcessDocumentRequestedV1.model_validate(
            _payload(object_key=f"documents/{DOCUMENT_ID}/{tail}")
        )
