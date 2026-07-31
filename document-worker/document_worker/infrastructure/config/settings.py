"""Настройки сервиса: единственное место, где читается окружение.

Секции вложенные, разделитель — двойное подчёркивание. Лишняя переменная
отвергается: опечатка в имени иначе тихо превращается в значение по умолчанию,
и расхождение вскрывается уже в бою.

Каталог секретов подключается только если существует. Жёстко заданный
`/run/secrets` ронял бы конструктор на машине разработчика, где сервис пишут,
но не запускают.

Пароли обёрнуты в `SecretStr`: настройки печатаются в логах старта, и
незакрытое значение уезжает туда навсегда.
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path
from typing import Annotated, Self, cast

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from document_worker.application.config import (
    OutboxConfig,
    ProcessingConfig,
    SourceConfig,
    TransactionConfig,
)
from document_worker.domain.chunking.policy import (
    CHUNKING_VERSION,
    DEFAULT_CHUNKING_POLICY,
    ChunkingPolicy,
)
from document_worker.domain.constants import MAX_FILE_SIZE_BYTES, MAX_PAGES
from document_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    PipelineVersion,
)

SECRETS_DIR = Path("/run/secrets")
DEFAULT_TEMP_SUBDIR = "document-worker"

Port = Annotated[int, Field(ge=1, le=65_535)]
Positive = Annotated[int, Field(gt=0)]
PositiveSeconds = Annotated[float, Field(gt=0)]


class Section(BaseSettings):
    """Общая часть секций: неизменяемая и без лишних значений."""

    model_config = SettingsConfigDict(extra="forbid", frozen=True)


class DatabaseSettings(Section):
    """PostgreSQL: подключение, пул и таймауты транзакций."""

    dsn: SecretStr
    pool_size: Positive = 10
    max_overflow: int = Field(default=0, ge=0)
    pool_timeout_s: PositiveSeconds = 10.0
    claim_timeout_ms: Positive = 5_000
    release_timeout_ms: Positive = 3_000
    page_timeout_ms: Positive = 5_000
    chunks_timeout_ms: Positive = 15_000
    terminal_timeout_ms: Positive = 10_000


class RabbitSettings(Section):
    """RabbitMQ: подключение и параметры доставки."""

    url: SecretStr
    consumer_timeout_ms: Positive = 7_200_000
    graceful_timeout_s: PositiveSeconds = 30.0
    publish_timeout_s: PositiveSeconds = 5.0
    delivery_limit: Positive = 20
    declare_audit_queue: bool = False


class S3Settings(Section):
    """Хранилище исходных файлов."""

    endpoint_url: str
    access_key: SecretStr
    secret_key: SecretStr
    region: str = "us-east-1"
    default_bucket: str = "documents"
    download_timeout_s: PositiveSeconds = 120.0


class ProcessingSettings(Section):
    """Пределы обработки одного документа."""

    document_timeout_s: PositiveSeconds = 3_600.0
    max_concurrency: Positive = 1
    max_file_size_bytes: Positive = MAX_FILE_SIZE_BYTES
    max_pages: Positive = MAX_PAGES
    max_render_pixels: Positive = 40_000_000
    cpu_workers: Positive = 1
    temp_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / DEFAULT_TEMP_SUBDIR
    )
    temp_quota_bytes: Positive = 4 * 1024 * 1024 * 1024
    pipeline_version: str = "1.0.0"


class ChunkingSettings(Section):
    """Версия чанкования и бюджет токенов чанка."""

    version: str = str(CHUNKING_VERSION)
    encoding: str = DEFAULT_CHUNKING_POLICY.encoding
    target_tokens: Positive = DEFAULT_CHUNKING_POLICY.target_tokens
    max_tokens: Positive = DEFAULT_CHUNKING_POLICY.max_tokens
    min_tokens: Positive = DEFAULT_CHUNKING_POLICY.min_tokens
    overlap_tokens: int = Field(default=DEFAULT_CHUNKING_POLICY.overlap_tokens, ge=0)

    def policy(self) -> ChunkingPolicy:
        """Собирает политику и требует, чтобы её версия знала эти параметры.

        Raises:
            InvalidChunkingPolicy: Бюджет противоречив либо изменён без
                инкремента версии — чанки разных границ иначе попали бы в один
                namespace chunking_version.
        """
        policy = ChunkingPolicy(
            version=ChunkingVersion.parse(self.version),
            encoding=self.encoding,
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        policy.ensure_registered()
        return policy


class MessagingSettings(Section):
    """Идемпотентность доставки и имя потребителя."""

    claim_lease_s: Positive = 5_400
    consumer_name: str = Field(default_factory=socket.gethostname)


class OutboxSettings(Section):
    """Фоновая публикация накопленных событий."""

    batch_size: Positive = 100
    poll_interval_s: PositiveSeconds = 0.5
    lease_seconds: Positive = 30
    backoff_base_s: PositiveSeconds = 1.0
    backoff_cap_s: PositiveSeconds = 300.0


class AppSettings(BaseSettings):
    """Все настройки сервиса."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        # Каталог секретов существует только в развёрнутом окружении.
        secrets_dir=str(SECRETS_DIR) if SECRETS_DIR.is_dir() else None,
    )

    database: DatabaseSettings
    rabbit: RabbitSettings
    s3: S3Settings
    processing: ProcessingSettings = ProcessingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    messaging: MessagingSettings = MessagingSettings()
    outbox: OutboxSettings = OutboxSettings()

    @model_validator(mode="after")
    def timeouts_do_not_contradict_each_other(self) -> Self:
        """Сверяет пределы, которые обязаны согласовываться между собой.

        Raises:
            ValueError: Лиз короче обработки или пул меньше конкурентности.
        """
        if self.messaging.claim_lease_s <= self.processing.document_timeout_s:
            raise ValueError("лиз захвата обязан быть строго больше таймаута документа")
        consumer_timeout_s = self.rabbit.consumer_timeout_ms / 1000
        if self.processing.document_timeout_s >= consumer_timeout_s:
            raise ValueError(
                "таймаут потребителя обязан быть строго больше таймаута документа"
            )
        if self.database.pool_size < self.processing.max_concurrency:
            # Соединений меньше, чем документов в работе, значит ожидание в
            # пуле на каждой странице.
            raise ValueError("соединений в пуле меньше, чем документов в работе")
        return self

    def processing_config(self) -> ProcessingConfig:
        """Собирает прикладную конфигурацию обработки."""
        major, minor, patch = (
            int(part) for part in self.processing.pipeline_version.split(".")
        )
        return ProcessingConfig(
            pipeline_version=PipelineVersion(major, minor, patch),
            consumer_name=self.messaging.consumer_name,
            document_timeout_s=self.processing.document_timeout_s,
            claim_lease_s=self.messaging.claim_lease_s,
            chunking=self.chunking.policy(),
            source=SourceConfig(
                max_file_size_bytes=self.processing.max_file_size_bytes,
                max_pages=self.processing.max_pages,
                download_timeout_s=self.s3.download_timeout_s,
                temp_quota_bytes=self.processing.temp_quota_bytes,
            ),
            tx=TransactionConfig(
                claim_ms=self.database.claim_timeout_ms,
                release_ms=self.database.release_timeout_ms,
                page_ms=self.database.page_timeout_ms,
                chunks_ms=self.database.chunks_timeout_ms,
                terminal_ms=self.database.terminal_timeout_ms,
            ),
            outbox=OutboxConfig(
                batch_size=self.outbox.batch_size,
                poll_interval_s=self.outbox.poll_interval_s,
                lease_seconds=self.outbox.lease_seconds,
                backoff_base_s=self.outbox.backoff_base_s,
                backoff_cap_s=self.outbox.backoff_cap_s,
            ),
        )


def env_variable_names() -> tuple[str, ...]:
    """Имена всех переменных окружения, которые читает сервис."""
    names: list[str] = []
    for section, field in AppSettings.model_fields.items():
        # Все поля настроек — секции; иных полей у них нет по построению.
        model = cast("type[Section]", field.annotation)
        names.extend(
            f"{section.upper()}__{name.upper()}" for name in model.model_fields
        )
    return tuple(sorted(names))
