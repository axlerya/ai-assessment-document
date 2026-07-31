"""Команды use cases. correlation_id есть в каждой — он не берётся из окружения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from document_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
        EventId,
    )
    from document_worker.domain.value_objects.paging import PageNumber
    from document_worker.domain.value_objects.storage import MimeType, ObjectRef


@dataclass(frozen=True, slots=True)
class ProcessDocumentCommand:
    """Обработать документ целиком."""

    event_id: EventId
    document_id: DocumentId
    correlation_id: CorrelationId
    object_ref: ObjectRef
    mime_type: MimeType
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ExtractDocumentTextCommand:
    """Определить способ извлечения и прочитать текстовый слой."""

    document_id: DocumentId
    correlation_id: CorrelationId
    source_path: Path


@dataclass(frozen=True, slots=True)
class ProcessDocumentPageCommand:
    """Прочитать одну страницу и сохранить её."""

    document_id: DocumentId
    correlation_id: CorrelationId
    page_number: PageNumber
    source_path: str


@dataclass(frozen=True, slots=True)
class CreateDocumentChunksCommand:
    """Разбить прочитанные страницы документа на чанки."""

    document_id: DocumentId
    correlation_id: CorrelationId


@dataclass(frozen=True, slots=True)
class CompleteDocumentProcessingCommand:
    """Зафиксировать терминальный результат и событие о нём."""

    document_id: DocumentId
    correlation_id: CorrelationId
    event_id: EventId


@dataclass(frozen=True, slots=True)
class FailDocumentProcessingCommand:
    """Зафиксировать отказ обработки."""

    document_id: DocumentId
    correlation_id: CorrelationId
    event_id: EventId
    error_code: str
    error_message: str


@dataclass(frozen=True, slots=True)
class PublishOutboxEventsCommand:
    """Опубликовать накопленные события."""

    correlation_id: CorrelationId
    limit: int
