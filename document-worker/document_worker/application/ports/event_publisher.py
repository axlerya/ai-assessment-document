"""Публикация событий в шину."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from document_worker.application.dto.results import OutboxRecordDTO


@runtime_checkable
class EventPublisher(Protocol):
    """Отправляет готовое событие.

    Подтверждения публикации обязательны: неподтверждённая отправка это
    недоступность брокера, а не успех.
    """

    async def publish(self, record: OutboxRecordDTO) -> None:
        """Публикует одно событие."""
        ...
