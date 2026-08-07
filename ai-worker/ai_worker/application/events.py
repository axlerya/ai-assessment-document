"""Перевод завершённого прогона в строку outbox.

Домен про доставку не знает, поэтому ключ маршрутизации и представление
полезной нагрузки живут здесь. `event_id` не выдумывается: он детерминирован по
документу и версии эмбеддингов, и повторное завершение даёт тот же ключ,
который гасит уникальное ограничение outbox.

`event_id` кладётся и в тело: потребитель дедуплицирует по нему, а не по
заголовку, и база это требует проверкой `payload ->> 'event_id'` (ADR-0007).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ai_worker.application.dto.messaging import OutboxEventDTO
from ai_worker.domain.value_objects.identifiers import EventId

if TYPE_CHECKING:
    from datetime import datetime

    from ai_worker.domain.entities.document_index import DocumentIndex
    from ai_worker.domain.value_objects.versioning import ChunkingVersion

DOCUMENT_INDEXED: Final[str] = "document.indexed"
SCHEMA_VERSION: Final[int] = 1


def document_indexed_event(
    index: DocumentIndex,
    *,
    chunking_version: ChunkingVersion,
    duration_ms: int,
    occurred_at: datetime,
) -> OutboxEventDTO:
    """Готовит событие об успешно проиндексированном документе."""
    event_id = EventId.for_indexing(
        document_id=index.document_id, embedding_version=index.embedding.version
    )
    # Сквозной идентификатор обязателен для строки outbox. Если сообщение его
    # не принесло, им становится доставка, породившая работу: она и есть то,
    # к чему привязывается разбор.
    correlation = index.correlation_id or index.source_event_id
    return OutboxEventDTO(
        event_id=event_id.value,
        aggregate_id=index.document_id.value,
        event_type=DOCUMENT_INDEXED,
        # Топологию объявляет messaging; здесь ключ совпадает с типом события,
        # и второго источника имени не заводится.
        routing_key=DOCUMENT_INDEXED,
        payload={
            "event_id": str(event_id),
            "schema_version": SCHEMA_VERSION,
            "status": "indexed",
            "document_id": str(index.document_id),
            "embedding_version": str(index.embedding.version),
            "chunking_version": str(chunking_version),
            "model_name": index.embedding.model_name,
            "chunks_total": index.chunks_total,
            "chunks_embedded": index.chunks_embedded,
            "chunks_failed": index.chunks_failed,
            "duration_ms": duration_ms,
            "occurred_at": occurred_at.isoformat(),
            "correlation_id": str(correlation),
        },
        correlation_id=str(correlation),
        occurred_at=occurred_at,
    )
