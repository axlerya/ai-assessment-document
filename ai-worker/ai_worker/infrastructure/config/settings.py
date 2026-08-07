"""Настройки сервиса: единственное место, где читается окружение.

Секции вложенные, разделитель — двойное подчёркивание. Лишняя переменная
отвергается: опечатка в имени иначе тихо превращается в значение по умолчанию,
и расхождение вскрывается уже в бою.

Ключи и пароли обёрнуты в `SecretStr`: настройки печатаются в логах старта, и
незакрытое значение уезжает туда навсегда.

Проверки, связывающие секции между собой, живут здесь же. Каждая из них ловит
отказ, который иначе проявился бы под нагрузкой и выглядел бы случайным.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Annotated, Self, cast

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_worker.application.config import (
    ContextConfig,
    EmbeddingConfig,
    LlmConfig,
    OutboxConfig,
    ProcessingConfig,
    RerankConfig,
    RetrievalConfig,
)
from ai_worker.domain.constants import DENSE_DIMENSIONS, SPARSE_TOP_K
from ai_worker.domain.embedding.policy import EmbeddingPolicy
from ai_worker.domain.errors import InvalidEmbeddingPolicy
from ai_worker.domain.value_objects.versioning import EmbeddingVersion, PromptVersion

SECRETS_DIR = Path("/run/secrets")

Positive = Annotated[int, Field(gt=0)]
PositiveSeconds = Annotated[float, Field(gt=0)]
Share = Annotated[float, Field(ge=0.0, le=1.0)]


class Section(BaseSettings):
    """Общая часть секций: неизменяемая и без лишних значений."""

    model_config = SettingsConfigDict(extra="forbid", frozen=True)


class DatabaseSettings(Section):
    """PostgreSQL: подключение и пул."""

    dsn: SecretStr
    pool_size: Positive = 10
    max_overflow: int = Field(default=0, ge=0)
    pool_timeout_s: PositiveSeconds = 10.0
    statement_timeout_ms: Positive = 30_000


class RabbitSettings(Section):
    """RabbitMQ: подключение и параметры доставки."""

    url: SecretStr
    consumer_timeout_ms: Positive = 1_800_000
    graceful_timeout_s: PositiveSeconds = 30.0
    publish_timeout_s: PositiveSeconds = 5.0
    delivery_limit: Positive = 20
    declare_audit_queue: bool = False


class EmbeddingSettings(Section):
    """Модель представлений и её пределы.

    Всё, что меняет сам вектор, сверяется с реестром версий: правка здесь без
    инкремента версии отвергается на старте (ADR-0004).
    """

    version: str = "1.0.0"
    model_name: str = "BAAI/bge-m3"
    model_dir: Path = Path(".models")
    dimensions: Positive = DENSE_DIMENSIONS
    sparse_top_k: Positive = SPARSE_TOP_K
    normalize: bool = True
    # bge-m3 обучена без служебных префиксов, в отличие от семейства e5.
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: Positive = 8
    max_input_tokens: Positive = 1024
    timeout_s: PositiveSeconds = 120.0

    def policy(self) -> EmbeddingPolicy:
        """Собирает политику версий из настроек."""
        return EmbeddingPolicy(
            version=EmbeddingVersion.parse(self.version),
            model_name=self.model_name,
            dimensions=self.dimensions,
            normalize=self.normalize,
            max_input_tokens=self.max_input_tokens,
            sparse_top_k=self.sparse_top_k,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )


class RerankSettings(Section):
    """Кросс-энкодер и его пределы."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n: Positive = 50
    batch_size: Positive = 8
    timeout_s: PositiveSeconds = 120.0


class RetrievalSettings(Section):
    """Профиль поиска."""

    profile: str = "hybrid-rrf-v1"
    top_k_dense: Positive = 50
    top_k_sparse: Positive = 50
    rrf_k: Positive = 60
    ef_search: Positive = 100


class ContextSettings(Section):
    """Бюджет контекста и порог цитируемости."""

    token_budget: Positive = 8_000
    max_chunks: Positive = 20
    min_citable_confidence: Share = 0.60


class LlmSettings(Section):
    """Провайдер генерации."""

    base_url: str = "https://api.deepinfra.com/v1/openai"
    model: str
    api_key: SecretStr
    timeout_s: PositiveSeconds = 120.0
    max_output_tokens: Positive = 4_000


class PromptSettings(Section):
    """Версия промпта. Правка текста без инкремента ломает сравнимость оценок."""

    version: str = "1.0.0"


class ProcessingSettings(Section):
    """Пределы обработки одного сообщения."""

    message_timeout_s: PositiveSeconds = 900.0
    cpu_workers: Positive = 1


class MessagingSettings(Section):
    """Идемпотентность доставки и имя потребителя."""

    claim_lease_s: Positive = 1_800
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
    llm: LlmSettings
    embedding: EmbeddingSettings = EmbeddingSettings()
    rerank: RerankSettings = RerankSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    context: ContextSettings = ContextSettings()
    prompt: PromptSettings = PromptSettings()
    processing: ProcessingSettings = ProcessingSettings()
    messaging: MessagingSettings = MessagingSettings()
    outbox: OutboxSettings = OutboxSettings()

    @model_validator(mode="after")
    def limits_do_not_contradict_each_other(self) -> Self:
        """Сверяет пределы, которые обязаны согласовываться между собой.

        Raises:
            ValueError: Один из пределов делает другой недостижимым.
        """
        self._check_timeouts()
        self._check_embedding_policy()
        self._check_pipeline_widths()
        return self

    def _check_timeouts(self) -> None:
        consumer_timeout_s = self.rabbit.consumer_timeout_ms / 1000
        if self.processing.message_timeout_s >= consumer_timeout_s:
            # Таймер потребителя идёт с момента доставки, а не с начала
            # обработки: при более коротком значении брокер закроет канал, и
            # сообщения вернутся в очередь мимо прикладного счётчика попыток.
            raise ValueError(
                "таймаут потребителя обязан быть строго больше таймаута сообщения"
            )
        if self.messaging.claim_lease_s <= self.processing.message_timeout_s:
            raise ValueError("лиз захвата обязан быть строго больше таймаута сообщения")

    def _check_embedding_policy(self) -> None:
        # Пределы вектора и его сверка с реестром версий — одно и то же
        # решение, принятое в домене. Повторять его здесь значило бы завести
        # вторую формулировку, которая однажды разойдётся с первой.
        try:
            self.embedding.policy().ensure_registered()
        except InvalidEmbeddingPolicy as error:
            raise ValueError(error.message) from error

    def _check_pipeline_widths(self) -> None:
        if self.rerank.top_n < self.context.max_chunks:
            # В контекст попадает только переранжированное: просить больше
            # фрагментов, чем прошло реранкинг, значит просить несуществующее.
            raise ValueError(
                "реранкер обязан обрабатывать не меньше фрагментов, чем берёт контекст"
            )

    def processing_config(self) -> ProcessingConfig:
        """Собирает конфигурацию для прикладного слоя."""
        return ProcessingConfig(
            consumer_name=self.messaging.consumer_name,
            message_timeout_s=self.processing.message_timeout_s,
            claim_lease_s=self.messaging.claim_lease_s,
            embedding=EmbeddingConfig(
                policy=self.embedding.policy(),
                batch_size=self.embedding.batch_size,
                timeout_s=self.embedding.timeout_s,
            ),
            retrieval=RetrievalConfig(
                profile=self.retrieval.profile,
                top_k_dense=self.retrieval.top_k_dense,
                top_k_sparse=self.retrieval.top_k_sparse,
                rrf_k=self.retrieval.rrf_k,
                ef_search=self.retrieval.ef_search,
            ),
            rerank=RerankConfig(
                model_name=self.rerank.model_name,
                top_n=self.rerank.top_n,
                batch_size=self.rerank.batch_size,
                timeout_s=self.rerank.timeout_s,
            ),
            context=ContextConfig(
                token_budget=self.context.token_budget,
                max_chunks=self.context.max_chunks,
                min_citable_confidence=self.context.min_citable_confidence,
            ),
            llm=LlmConfig(
                model=self.llm.model,
                prompt_version=PromptVersion.parse(self.prompt.version),
                timeout_s=self.llm.timeout_s,
                max_output_tokens=self.llm.max_output_tokens,
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
