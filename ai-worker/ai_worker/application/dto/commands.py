"""Команды сценариев.

Сообщение брокера сюда не доезжает: подписчик переводит его в команду, и
сценарий не знает ни про AMQP, ни про Pydantic. Поля здесь — только то, без
чего работу не начать; версия чанкования и состояние документа читаются из
базы (ADR-0008), а не переносятся из тела события.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from ai_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
        EventId,
    )


@dataclass(frozen=True, slots=True)
class IndexDocumentCommand:
    """Просьба проиндексировать обработанный документ."""

    event_id: EventId
    document_id: DocumentId
    correlation_id: CorrelationId | None
    occurred_at: datetime
