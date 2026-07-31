"""Команды use cases. correlation_id есть в каждой — он не берётся из окружения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from document_worker.application.dto.extraction import (
        DocumentExtraction,
        PagePlanEntryDTO,
    )
    from document_worker.domain.value_objects.enums import ProcessingStage
    from document_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
        EventId,
        JobId,
    )
    from document_worker.domain.value_objects.storage import (
        Checksum,
        FileSize,
        MimeType,
        ObjectRef,
    )


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
    job_id: JobId
    entry: PagePlanEntryDTO
    extraction: DocumentExtraction


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
    job_id: JobId
    page_count: int
    chunks_total: int
    # Размер и сумма выясняются скачиванием, а строку документа создаёт сервис
    # приёма файлов и их не знает.
    source_size: FileSize
    source_checksum: Checksum


@dataclass(frozen=True, slots=True)
class FailDocumentProcessingCommand:
    """Зафиксировать отказ обработки."""

    document_id: DocumentId
    correlation_id: CorrelationId
    event_id: EventId
    job_id: JobId | None
    error_code: str
    error_message: str
    stage: ProcessingStage
    pages_persisted: int = 0


@dataclass(frozen=True, slots=True)
class PublishOutboxEventsCommand:
    """Опубликовать накопленные события."""

    correlation_id: CorrelationId
    limit: int
