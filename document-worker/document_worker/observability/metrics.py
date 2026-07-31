"""Метрики конвейера.

Собственный реестр, а не глобальный по умолчанию: глобальный собирает всё, что
успела зарегистрировать любая импортированная библиотека, и повторная
регистрация в тесте роняет процесс.

Лейблов с неограниченной кардинальностью здесь нет и быть не может.
Идентификатор документа в лейбле означает отдельный временной ряд на каждый
документ — хранилище метрик умирает на первой же тысяче, а сама метрика
перестаёт отвечать на вопросы, ради которых заводилась.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from collections.abc import Collection

NAMESPACE: Final[str] = "docworker"
CONTENT_TYPE: Final[str] = "text/plain; version=0.0.4; charset=utf-8"

# Лейблы, которые не имеют права появиться ни в одной серии.
FORBIDDEN_LABELS: Final[frozenset[str]] = frozenset(
    {"document_id", "correlation_id", "event_id", "object_key", "page_id", "job_id"}
)

# Страница обрабатывается от долей секунды (текстовый слой) до десятков секунд
# (распознавание), поэтому шкала логарифмическая.
PAGE_DURATION_BUCKETS: Final[tuple[float, ...]] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


class Metrics:
    """Реестр метрик сервиса."""

    __slots__ = (
        "documents_processed",
        "errors",
        "messages_consumed",
        "messages_dlq",
        "outbox_lag_seconds",
        "outbox_pending",
        "page_duration_seconds",
        "pages_processed",
        "registry",
    )

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Заводит все серии в собственном реестре."""
        self.registry = registry or CollectorRegistry()
        self.documents_processed = Counter(
            "documents_processed_total",
            "Документы, дошедшие до терминального статуса",
            labelnames=("status",),
            namespace=NAMESPACE,
            registry=self.registry,
        )
        self.pages_processed = Counter(
            "pages_processed_total",
            "Страницы по способу извлечения и итогу",
            labelnames=("extraction_method", "status"),
            namespace=NAMESPACE,
            registry=self.registry,
        )
        self.page_duration_seconds = Histogram(
            "page_processing_duration_seconds",
            "Длительность обработки одной страницы",
            labelnames=("extraction_method",),
            buckets=PAGE_DURATION_BUCKETS,
            namespace=NAMESPACE,
            registry=self.registry,
        )
        self.messages_consumed = Counter(
            "messages_consumed_total",
            "Сообщения по исходу обработки",
            labelnames=("outcome",),
            namespace=NAMESPACE,
            registry=self.registry,
        )
        self.messages_dlq = Counter(
            "messages_dlq_total",
            "Сообщения, ушедшие в очередь разбора",
            labelnames=("error_code",),
            namespace=NAMESPACE,
            registry=self.registry,
        )
        self.errors = Counter(
            "errors_total",
            "Ошибки по типу и стадии",
            labelnames=("error_type", "error_code"),
            namespace=NAMESPACE,
            registry=self.registry,
        )
        # Отставание публикации — тихий отказ: события копятся, документы
        # обработаны, а потребители о них не знают.
        self.outbox_pending = Gauge(
            "outbox_pending",
            "Неопубликованные события",
            namespace=NAMESPACE,
            registry=self.registry,
        )
        self.outbox_lag_seconds = Gauge(
            "outbox_lag_seconds",
            "Возраст самого старого неопубликованного события",
            namespace=NAMESPACE,
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Отдаёт метрики в формате Prometheus."""
        return generate_latest(self.registry)

    def label_names(self) -> frozenset[str]:
        """Все лейблы, объявленные в реестре."""
        names: set[str] = set()
        for collector in self.registry._collector_to_names:  # noqa: SLF001 — иначе имена лейблов не достать
            names |= set(getattr(collector, "_labelnames", ()))
        return frozenset(names)


def ensure_labels_are_bounded(labels: Collection[str]) -> None:
    """Проверяет, что среди лейблов нет неограниченных по кардинальности.

    Raises:
        ValueError: Лейбл заводит отдельный ряд на каждый документ.
    """
    forbidden = sorted(set(labels) & FORBIDDEN_LABELS)
    if forbidden:
        raise ValueError(
            f"лейблы неограниченной кардинальности: {', '.join(forbidden)}"
        )
