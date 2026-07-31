"""Подключение к брокеру и проверка согласованности его настроек.

Предвыборка жёстко равна единице. Обработка упирается в процессорный пул, и
при большей предвыборке второе сообщение всё равно ждёт освобождения того же
пула — но ждёт оно в состоянии «доставлено», где тикает `x-consumer-timeout`.
Пропускная способность масштабируется числом инстансов, а не предвыборкой.
"""

from __future__ import annotations

from typing import Final

from faststream.rabbit import Channel, RabbitBroker

PREFETCH_COUNT: Final[int] = 1


def build_broker(url: str, *, graceful_timeout_s: float = 30.0) -> RabbitBroker:
    """Создаёт брокер с единичной предвыборкой."""
    return RabbitBroker(
        url,
        default_channel=Channel(prefetch_count=PREFETCH_COUNT),
        graceful_timeout=graceful_timeout_s,
    )
