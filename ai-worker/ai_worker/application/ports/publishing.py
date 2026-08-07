"""Публикация исходящих событий."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ai_worker.application.dto.messaging import OutboxRecordDTO


@runtime_checkable
class EventPublisher(Protocol):
    """Отправляет накопленное событие в шину."""

    async def publish(self, record: OutboxRecordDTO) -> None:
        """Публикует событие и ждёт подтверждения.

        Неподтверждённая отправка — это недоступность брокера, а не успех:
        отметить такое событие опубликованным значит потерять его навсегда.
        """
        ...
