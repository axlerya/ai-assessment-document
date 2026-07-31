"""Что подписчику нужно снаружи.

Протоколы объявлены здесь, а реализованы в инфраструктуре: presentation и
infrastructure — соседние слои, и импорт между ними запрещён. Структурная
типизация связывает их без единого импорта, а конкретные объекты подставляет
композиционный корень.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from document_worker.application.dto.commands import ProcessDocumentCommand
    from document_worker.application.dto.results import ProcessDocumentResult


@runtime_checkable
class DocumentProcessor(Protocol):
    """Прикладной сценарий обработки документа."""

    async def execute(self, command: ProcessDocumentCommand) -> ProcessDocumentResult:
        """Обрабатывает одно сообщение целиком."""
        ...


@runtime_checkable
class MessageRetrier(Protocol):
    """Отложенный повтор и отправка на разбор."""

    async def schedule(
        self,
        body: bytes,
        headers: dict[str, object],
        *,
        attempt: int,
    ) -> None:
        """Кладёт копию сообщения на ступень задержки по номеру попытки."""
        ...

    async def send_to_dlq(self, body: bytes, headers: dict[str, object]) -> None:
        """Кладёт копию сообщения в очередь разбора."""
        ...
