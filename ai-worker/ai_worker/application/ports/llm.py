"""Вызов языковой модели.

Порт не знает ни про температуру, ни про формат сообщений: это устройство
конкретного поставщика. Он знает ровно то, что обещает сервису — инструкция,
запрос и контекст на входе, разобранный ответ и стоимость на выходе.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class LlmCompletion:
    """Ответ модели вместе с тем, во что он обошёлся."""

    content: Mapping[str, Any]
    input_tokens: int
    output_tokens: int


@runtime_checkable
class LlmProvider(Protocol):
    """Просит модель ответить структурой заданной формы."""

    async def complete(
        self,
        *,
        instruction: str,
        request: str,
        schema: Mapping[str, Any],
        timeout_s: float,
    ) -> LlmCompletion:
        """Возвращает разобранный ответ модели.

        Форма ответа задаётся схемой: свободный текст пришлось бы разбирать
        догадками, а догадка в обосновании утверждения недопустима.
        """
        ...
