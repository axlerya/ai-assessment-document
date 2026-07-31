"""Отклонение сообщений, которые не удалось разобрать.

Декодирование типизированного параметра происходит до обработчика, поэтому
ошибку разбора видит только middleware. При ручном подтверждении такое
сообщение иначе не подтвердит никто, и оно провисит непринятым до закрытия
канала — то есть до принудительного возврата всей пачки.

Здесь именно `reject`, а не публикация копии: тела мы не понимаем,
идентификатора события в нём может не быть, обогащать нечем. База в этой ветке
не трогается вовсе.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, override

from faststream import BaseMiddleware
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

DECODE_FAILURES = (
    ValidationError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    ValueError,
)


class DecodeGuardMiddleware(BaseMiddleware):
    """Нечитаемое тело — неисправимая ошибка: отклонение и брокерский путь в DLQ."""

    @override
    async def consume_scope(
        self,
        call_next: Callable[[Any], Awaitable[Any]],
        msg: Any,
    ) -> Any:
        """Пропускает сообщение к обработчику, отклоняя неразобранное."""
        try:
            return await call_next(msg)
        except DECODE_FAILURES:
            if self.msg is not None:  # pragma: no branch — сообщение здесь всегда есть
                await self.msg.reject()
            return None
