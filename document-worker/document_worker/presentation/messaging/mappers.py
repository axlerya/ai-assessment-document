"""Перевод сообщения в команду прикладного слоя.

Через границу проходит dataclass, а не модель Pydantic: иначе тип транспорта
протёк бы в use case и потянул за собой всю библиотеку.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_worker.application.dto.commands import ProcessDocumentCommand
from document_worker.domain.value_objects.identifiers import (
    CorrelationId,
    DocumentId,
    EventId,
)
from document_worker.domain.value_objects.storage import MimeType, ObjectRef

if TYPE_CHECKING:
    from document_worker.presentation.messaging.contracts.commands import (
        ProcessDocumentRequestedV1,
    )


def to_process_document_command(
    message: ProcessDocumentRequestedV1,
    *,
    default_bucket: str,
    attempt: int,
    max_attempts: int,
) -> ProcessDocumentCommand:
    """Собирает команду обработки из входящего сообщения."""
    return ProcessDocumentCommand(
        event_id=EventId(message.event_id),
        document_id=DocumentId(message.document_id),
        correlation_id=CorrelationId(str(message.correlation_id)),
        object_ref=ObjectRef(
            bucket=message.bucket or default_bucket, key=message.object_key
        ),
        mime_type=MimeType(message.mime_type),
        occurred_at=message.occurred_at,
        attempt=attempt,
        max_attempts=max_attempts,
    )
