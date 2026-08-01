"""Конфигурация обработки.

Только frozen dataclass'ы. Переменные окружения слой не читает: конфиг
собирается в композиционном корне. Пороги качества сюда не попадают — они
живут полями доменных политик.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from document_worker.domain.chunking.policy import DEFAULT_CHUNKING_POLICY
from document_worker.domain.constants import (
    MAX_FILE_SIZE_BYTES,
    MAX_PAGES,
    SUPPORTED_MIME_TYPES,
)
from document_worker.domain.errors import InvariantViolation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from document_worker.domain.chunking.policy import ChunkingPolicy
    from document_worker.domain.value_objects.versioning import PipelineVersion


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise InvariantViolation(
            f"{name} должен быть положительным",
            context={name: value},
        )


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Пределы приёма исходного файла."""

    supported_mime_types: frozenset[str] = SUPPORTED_MIME_TYPES
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES
    max_pages: int = MAX_PAGES
    download_timeout_s: float = 120.0
    temp_quota_bytes: int = 4 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        """Не даёт поднять лимиты выше доменных."""
        if self.max_pages > MAX_PAGES:
            raise InvariantViolation(
                f"предел страниц выше доменного {MAX_PAGES}",
                context={"max_pages": self.max_pages},
            )
        if self.max_file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise InvariantViolation(
                f"предел размера выше доменного {MAX_FILE_SIZE_BYTES}",
                context={"max_file_size_bytes": self.max_file_size_bytes},
            )
        _require_positive(self.download_timeout_s, "download_timeout_s")


@dataclass(frozen=True, slots=True)
class OcrConfig:
    """Параметры распознавания. Движок-специфика уходит в options."""

    languages: tuple[str, ...] = ("eslav",)
    options: Mapping[str, str] = field(default_factory=dict)
    dpi_primary: int = 300
    dpi_retry: int = 400
    dpi_degraded: int = 200
    page_timeout_s: float = 60.0
    max_page_attempts: int = 2
    # Выше этой высоты строки рост разрешения распознаванию уже ничего не даёт:
    # распознаватель масштабирует каждый кроп строки к своей высоте.
    target_line_height_px: int = 32
    # Операционное решение «повторить ли на большем DPI»: на статус страницы
    # не влияет, поэтому живёт здесь, а не в политике.
    retry_below_confidence: float = 0.60

    def __post_init__(self) -> None:
        """Проверяет таймаут и число попыток."""
        _require_positive(self.page_timeout_s, "page_timeout_s")
        _require_positive(self.max_page_attempts, "max_page_attempts")


@dataclass(frozen=True, slots=True)
class TransactionConfig:
    """Таймауты транзакций, чтобы ни одна не залипла."""

    claim_ms: int = 5_000
    release_ms: int = 3_000
    page_ms: int = 5_000
    chunks_ms: int = 15_000
    terminal_ms: int = 10_000

    def __post_init__(self) -> None:
        """Требует положительных таймаутов."""
        for name in ("claim_ms", "release_ms", "page_ms", "chunks_ms", "terminal_ms"):
            _require_positive(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class OutboxConfig:
    """Параметры публикации накопленных событий."""

    batch_size: int = 100
    poll_interval_s: float = 0.5
    lease_seconds: int = 30
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 300.0

    def backoff_for(self, attempts: int) -> timedelta:
        """Отсрочка следующей попытки публикации.

        Растёт вдвое с каждой неудачей и упирается в потолок: без него
        недоступный сутки брокер отодвинул бы событие на годы.
        """
        seconds = self.backoff_base_s * 2 ** max(attempts, 0)
        return timedelta(seconds=min(seconds, self.backoff_cap_s))

    def __post_init__(self) -> None:
        """Проверяет размеры пачки, лиза и границы backoff."""
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.poll_interval_s, "poll_interval_s")
        _require_positive(self.lease_seconds, "lease_seconds")
        _require_positive(self.backoff_base_s, "backoff_base_s")
        if self.backoff_cap_s < self.backoff_base_s:
            raise InvariantViolation(
                "потолок backoff ниже его основания",
                context={"base": self.backoff_base_s, "cap": self.backoff_cap_s},
            )


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Конфигурация обработки документа целиком."""

    pipeline_version: PipelineVersion
    consumer_name: str
    document_timeout_s: float
    claim_lease_s: int = 900
    source: SourceConfig = field(default_factory=SourceConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    chunking: ChunkingPolicy = DEFAULT_CHUNKING_POLICY
    tx: TransactionConfig = field(default_factory=TransactionConfig)
    outbox: OutboxConfig = field(default_factory=OutboxConfig)

    def __post_init__(self) -> None:
        """Проверяет таймаут документа и параметры лиза."""
        _require_positive(self.document_timeout_s, "document_timeout_s")
        _require_positive(self.claim_lease_s, "claim_lease_s")
        if not self.consumer_name:
            raise InvariantViolation("имя потребителя пустое")
