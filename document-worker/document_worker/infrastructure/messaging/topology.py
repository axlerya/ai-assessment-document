"""Топология RabbitMQ: обменники, очереди и лестница отложенных повторов.

Аргументы очередей здесь не косметика. `x-expires` на ступени повтора удалил
бы её вместе с отложенными сообщениями, и те не дедлеттерятся, а исчезают.
`x-max-length` на рабочей очереди при `reject-publish` превратил бы лавину
повторов в отказ публикации, а при `drop-head` — в тихую потерю команд.
`reject-publish` на DLQ остановил бы обработку живых документов: dead-letter
перестал бы приниматься, и сообщение вечно возвращалось бы в рабочую очередь.

Топология описывается здесь целиком, потому что это устройство брокера, а не
протокол: подписчик получает свою очередь из композиционного корня.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from faststream.rabbit import ExchangeType, QueueType, RabbitExchange, RabbitQueue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from faststream.rabbit.schemas.queue import QuorumQueueArgs

COMMANDS_EXCHANGE: Final[str] = "documents.commands"
EVENTS_EXCHANGE: Final[str] = "documents.events"
RETRY_EXCHANGE: Final[str] = "documents.retry"
DLX_EXCHANGE: Final[str] = "documents.dlx"
UNROUTED_EXCHANGE: Final[str] = "documents.unrouted"

RK_PROCESS_REQUESTED: Final[str] = "document.process.requested"
RK_RETRY_BASE: Final[str] = "document.process.retry"
RK_DLQ: Final[str] = "document.process.dlq"
RK_EVENTS_ALL: Final[str] = "document.#"

PROCESS_REQUESTED_QUEUE: Final[str] = "document.process.requested.q"
DLQ_QUEUE: Final[str] = "document.process.dlq.q"
UNROUTED_QUEUE: Final[str] = "document.unrouted.q"
AUDIT_QUEUE: Final[str] = "document.events.audit.q"

# Суммарное окно восстановления — 42,5 минуты.
RETRY_LADDER: Final[tuple[tuple[str, int], ...]] = (
    ("5s", 5_000),
    ("30s", 30_000),
    ("2m", 120_000),
    ("10m", 600_000),
    ("30m", 1_800_000),
)
RETRY_MAX_ATTEMPTS: Final[int] = len(RETRY_LADDER)

DEFAULT_DELIVERY_LIMIT: Final[int] = 20
DLQ_TTL_MS: Final[int] = 14 * 24 * 60 * 60 * 1_000
DLQ_MAX_LENGTH: Final[int] = 100_000
UNROUTED_TTL_MS: Final[int] = 7 * 24 * 60 * 60 * 1_000
AUDIT_MAX_LENGTH: Final[int] = 10_000


def retry_level_for_attempt(attempt: int) -> str:
    """Ступень задержки по номеру попытки.

    Raises:
        ValueError: Номер попытки вне лестницы.
    """
    if not 1 <= attempt <= RETRY_MAX_ATTEMPTS:
        raise ValueError(f"попытка {attempt} вне лестницы повторов")
    return RETRY_LADDER[attempt - 1][0]


def retry_queue_name(level: str) -> str:
    """Имя очереди ступени задержки."""
    return f"{RK_RETRY_BASE}.{level}.q"


@dataclass(frozen=True, slots=True)
class Binding:
    """Связь очереди с обменником по ключу маршрутизации."""

    queue: str
    exchange: str
    routing_key: str


@dataclass(frozen=True, slots=True)
class Topology:
    """Полное описание того, что сервис объявляет в брокере."""

    exchanges: tuple[RabbitExchange, ...]
    queues: tuple[RabbitQueue, ...]
    bindings: tuple[Binding, ...]
    commands: RabbitExchange
    events: RabbitExchange
    retry: RabbitExchange
    dead_letter_exchange: RabbitExchange
    process_requested: RabbitQueue
    retry_queues: Mapping[str, RabbitQueue]
    dead_letter: RabbitQueue

    @property
    def levels(self) -> tuple[str, ...]:
        """Ступени задержки в порядке возрастания."""
        return tuple(self.retry_queues)


def build_topology(
    *,
    consumer_timeout_ms: int,
    retry_ladder: Sequence[tuple[str, int]] = RETRY_LADDER,
    delivery_limit: int = DEFAULT_DELIVERY_LIMIT,
    declare_audit_queue: bool = False,
) -> Topology:
    """Собирает топологию под заданные таймауты и лестницу задержек."""
    exchanges = _exchanges()
    process_requested = _process_requested_queue(
        consumer_timeout_ms=consumer_timeout_ms, delivery_limit=delivery_limit
    )
    retry_queues = {
        level: _retry_queue(level, ttl_ms) for level, ttl_ms in retry_ladder
    }
    dead_letter = _dead_letter_queue()
    queues = (
        process_requested,
        *retry_queues.values(),
        dead_letter,
        _unrouted_queue(),
        *((_audit_queue(),) if declare_audit_queue else ()),
    )
    return Topology(
        exchanges=exchanges,
        queues=queues,
        bindings=_bindings(retry_ladder, declare_audit_queue=declare_audit_queue),
        commands=_named(exchanges, COMMANDS_EXCHANGE),
        events=_named(exchanges, EVENTS_EXCHANGE),
        retry=_named(exchanges, RETRY_EXCHANGE),
        dead_letter_exchange=_named(exchanges, DLX_EXCHANGE),
        process_requested=process_requested,
        retry_queues=retry_queues,
        dead_letter=dead_letter,
    )


def _named(exchanges: Sequence[RabbitExchange], name: str) -> RabbitExchange:
    return next(exchange for exchange in exchanges if exchange.name == name)


def _exchanges() -> tuple[RabbitExchange, ...]:
    # Альтернативный обменник обязателен на трёх первых: без него команда с
    # опечаткой в ключе исчезает молча, без ошибки у публикующей стороны.
    unrouted = {"alternate-exchange": UNROUTED_EXCHANGE}
    return (
        RabbitExchange(UNROUTED_EXCHANGE, type=ExchangeType.FANOUT, durable=True),
        RabbitExchange(
            COMMANDS_EXCHANGE, type=ExchangeType.TOPIC, durable=True, arguments=unrouted
        ),
        RabbitExchange(
            EVENTS_EXCHANGE, type=ExchangeType.TOPIC, durable=True, arguments=unrouted
        ),
        RabbitExchange(
            RETRY_EXCHANGE, type=ExchangeType.TOPIC, durable=True, arguments=unrouted
        ),
        # У ловушки своей ловушки нет: единственный потребитель DLX — сама DLQ.
        RabbitExchange(DLX_EXCHANGE, type=ExchangeType.TOPIC, durable=True),
    )


def _process_requested_queue(
    *, consumer_timeout_ms: int, delivery_limit: int
) -> RabbitQueue:
    # `x-consumer-timeout` как аргумент очереди появился в RabbitMQ 3.12, а
    # словарь аргументов FastStream описывает набор постарше — отсюда приведение.
    arguments = cast(
        "QuorumQueueArgs",
        {
            # Единственный автоматический предохранитель от возвратов, которых
            # прикладной счётчик попыток не видит: обрыв канала, OOM, деплой.
            "x-delivery-limit": delivery_limit,
            "x-dead-letter-exchange": DLX_EXCHANGE,
            "x-dead-letter-routing-key": RK_DLQ,
            # По умолчанию quorum-очередь дедлеттерит at-most-once и имеет
            # право потерять сообщение при недоступности DLX.
            "x-dead-letter-strategy": "at-least-once",
            # Требование брокера для at-least-once. Предела длины нет, поэтому
            # отказ публикации не срабатывает никогда.
            "x-overflow": "reject-publish",
            # Таймер идёт с момента доставки, а не с начала обработки: значение
            # брокера по умолчанию короче обработки скана.
            "x-consumer-timeout": consumer_timeout_ms,
        },
    )
    return RabbitQueue(
        PROCESS_REQUESTED_QUEUE,
        queue_type=QueueType.QUORUM,
        durable=True,
        routing_key=RK_PROCESS_REQUESTED,
        arguments=arguments,
    )


def _retry_queue(level: str, ttl_ms: int) -> RabbitQueue:
    # Classic, а не quorum: потребителей нет, предел доставок бессмыслен,
    # а per-queue TTL даёт ровно нужную семантику без head-of-line blocking.
    return RabbitQueue(
        retry_queue_name(level),
        queue_type=QueueType.CLASSIC,
        durable=True,
        routing_key=f"{RK_RETRY_BASE}.{level}",
        arguments={
            "x-message-ttl": ttl_ms,
            "x-dead-letter-exchange": COMMANDS_EXCHANGE,
            "x-dead-letter-routing-key": RK_PROCESS_REQUESTED,
        },
    )


def _dead_letter_queue() -> RabbitQueue:
    return RabbitQueue(
        DLQ_QUEUE,
        queue_type=QueueType.QUORUM,
        durable=True,
        routing_key=RK_DLQ,
        arguments={
            # Каждый терминальный отказ уже зафиксирован в PostgreSQL; DLQ —
            # инструмент разбора, а не хранилище истины.
            "x-message-ttl": DLQ_TTL_MS,
            "x-max-length": DLQ_MAX_LENGTH,
            "x-overflow": "drop-head",
        },
    )


def _unrouted_queue() -> RabbitQueue:
    return RabbitQueue(
        UNROUTED_QUEUE,
        queue_type=QueueType.CLASSIC,
        durable=True,
        arguments={"x-message-ttl": UNROUTED_TTL_MS},
    )


def _audit_queue() -> RabbitQueue:
    return RabbitQueue(
        AUDIT_QUEUE,
        queue_type=QueueType.CLASSIC,
        durable=True,
        routing_key=RK_EVENTS_ALL,
        arguments={"x-max-length": AUDIT_MAX_LENGTH, "x-overflow": "drop-head"},
    )


def _bindings(
    retry_ladder: Sequence[tuple[str, int]],
    *,
    declare_audit_queue: bool,
) -> tuple[Binding, ...]:
    first_level = retry_ladder[0][0]
    return (
        Binding(PROCESS_REQUESTED_QUEUE, COMMANDS_EXCHANGE, RK_PROCESS_REQUESTED),
        *(
            Binding(retry_queue_name(level), RETRY_EXCHANGE, f"{RK_RETRY_BASE}.{level}")
            for level, _ in retry_ladder
        ),
        # Ключ устава без суффикса работает и даёт первую ступень.
        Binding(retry_queue_name(first_level), RETRY_EXCHANGE, RK_RETRY_BASE),
        Binding(DLQ_QUEUE, DLX_EXCHANGE, RK_DLQ),
        # При брокерском dead-lettering по x-delivery-limit ключ может
        # сохраниться исходным — второй binding закрывает обе версии поведения.
        Binding(DLQ_QUEUE, DLX_EXCHANGE, RK_PROCESS_REQUESTED),
        Binding(UNROUTED_QUEUE, UNROUTED_EXCHANGE, ""),
        *(
            (Binding(AUDIT_QUEUE, EVENTS_EXCHANGE, RK_EVENTS_ALL),)
            if declare_audit_queue
            else ()
        ),
    )
