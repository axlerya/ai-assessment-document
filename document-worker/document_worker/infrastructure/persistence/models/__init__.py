"""ORM-модели. Импортируются все, иначе metadata соберётся неполной."""

from __future__ import annotations

from document_worker.infrastructure.persistence.models.chunk import DocumentChunkRow
from document_worker.infrastructure.persistence.models.document import DocumentRow
from document_worker.infrastructure.persistence.models.job import ProcessingJobRow
from document_worker.infrastructure.persistence.models.message import (
    ProcessedMessageRow,
)
from document_worker.infrastructure.persistence.models.outbox import OutboxEventRow
from document_worker.infrastructure.persistence.models.page import (
    DocumentPageRow,
    IllegibleSpanRow,
)

__all__ = [
    "DocumentChunkRow",
    "DocumentPageRow",
    "DocumentRow",
    "IllegibleSpanRow",
    "OutboxEventRow",
    "ProcessedMessageRow",
    "ProcessingJobRow",
]
