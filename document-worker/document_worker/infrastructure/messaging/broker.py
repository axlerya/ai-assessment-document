"""Подключение к брокеру и проверка согласованности его настроек.

Предвыборка жёстко равна единице. Обработка упирается в процессорный пул, и
при большей предвыборке второе сообщение всё равно ждёт освобождения того же
пула — но ждёт оно в состоянии «доставлено», где тикает `x-consumer-timeout`.
Пропускная способность масштабируется числом инстансов, а не предвыборкой.
"""

from __future__ import annotations

from typing import Final

from faststream.rabbit import Channel, RabbitBroker

from document_worker.application.errors import SchemaMisconfiguredError

PREFETCH_COUNT: Final[int] = 1
MILLISECONDS_IN_SECOND: Final[int] = 1000
# Доля таймаута потребителя, за которой очередь доставленных сообщений
# перестаёт помещаться в отведённое брокером время.
SAFE_TIMEOUT_SHARE: Final[float] = 0.5


def build_broker(url: str, *, graceful_timeout_s: float = 30.0) -> RabbitBroker:
    """Создаёт брокер с единичной предвыборкой."""
    return RabbitBroker(
        url,
        default_channel=Channel(prefetch_count=PREFETCH_COUNT),
        graceful_timeout=graceful_timeout_s,
    )


def ensure_delivery_settings_are_consistent(
    *,
    prefetch_count: int,
    consumer_timeout_ms: int,
    document_timeout_s: float,
    claim_lease_s: int,
) -> None:
    """Проверяет, что таймауты доставки и обработки не противоречат друг другу.

    Raises:
        SchemaMisconfiguredError: Настройки гарантируют потерю или бесполезную
            параллельную работу.
    """
    consumer_timeout_s = consumer_timeout_ms / MILLISECONDS_IN_SECOND
    if document_timeout_s >= consumer_timeout_s:
        raise SchemaMisconfiguredError(
            "таймаут потребителя обязан быть строго больше таймаута документа",
            context={
                "consumer_timeout_s": consumer_timeout_s,
                "document_timeout_s": document_timeout_s,
            },
        )
    if (
        prefetch_count > PREFETCH_COUNT
        and document_timeout_s >= SAFE_TIMEOUT_SHARE * consumer_timeout_s
    ):
        # Сообщения со второго по N-е протухнут в доставленных раньше, чем
        # воркер до них дойдёт.
        raise SchemaMisconfiguredError(
            "предвыборка больше единицы несовместима с таким таймаутом документа",
            context={
                "prefetch_count": prefetch_count,
                "document_timeout_s": document_timeout_s,
                "consumer_timeout_s": consumer_timeout_s,
            },
        )
    if claim_lease_s <= document_timeout_s:
        # Лиз, протухающий в середине штатной обработки, превращает каждую
        # повторную доставку в возобновление параллельно живому воркеру.
        raise SchemaMisconfiguredError(
            "лиз захвата обязан быть строго больше таймаута документа",
            context={
                "claim_lease_s": claim_lease_s,
                "document_timeout_s": document_timeout_s,
            },
        )
