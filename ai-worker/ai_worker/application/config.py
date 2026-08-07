"""Конфигурация прикладного слоя.

Сценарий не читает окружение: он получает готовые значения, уже прошедшие
проверки. Иначе каждая новая настройка требовала бы своей проверки в каждом
месте использования, и однажды одна из них была бы забыта.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_worker.domain.embedding.policy import EmbeddingPolicy
    from ai_worker.domain.value_objects.versioning import PromptVersion


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Чем строятся представления и в каких пределах.

    Всё, что влияет на сам вектор, живёт в политике: держать те же параметры
    ещё и здесь значило бы завести второй источник истины для геометрии.
    Остаются пределы прогона — они на вектор не влияют.
    """

    policy: EmbeddingPolicy
    batch_size: int
    timeout_s: float


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Профиль поиска. Имя профиля уезжает в черновик и в историю."""

    profile: str
    top_k_dense: int
    top_k_sparse: int
    rrf_k: int
    ef_search: int


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """Пределы переранжирования."""

    model_name: str
    top_n: int
    batch_size: int
    timeout_s: float


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Бюджет контекста и порог цитируемости фрагмента."""

    token_budget: int
    max_chunks: int
    min_citable_confidence: float


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Куда и с какой моделью обращаться за черновиком."""

    model: str
    prompt_version: PromptVersion
    timeout_s: float
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class OutboxConfig:
    """Фоновая публикация накопленных событий."""

    batch_size: int
    poll_interval_s: float
    lease_seconds: int
    backoff_base_s: float
    backoff_cap_s: float


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Всё, что нужно сценариям сервиса."""

    consumer_name: str
    message_timeout_s: float
    claim_lease_s: int
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    context: ContextConfig
    llm: LlmConfig
    outbox: OutboxConfig
