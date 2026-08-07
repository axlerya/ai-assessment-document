"""Настройки: единственное место, где читается окружение."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from ai_worker.domain.constants import DENSE_DIMENSIONS, SPARSE_TOP_K
from ai_worker.infrastructure.config.settings import AppSettings, env_variable_names

pytestmark = pytest.mark.unit

MINIMAL: dict[str, Any] = {
    "database": {
        "dsn": "postgresql+asyncpg://ai_worker:secret@localhost:5432/documents"
    },
    "rabbit": {"url": "amqp://ai_worker:secret@localhost:5672/documents"},
    "llm": {"api_key": "secret", "model": "deepseek-ai/DeepSeek-V4-Flash"},
}


def _settings(**overrides: Any) -> AppSettings:
    return AppSettings.model_validate(MINIMAL | overrides)


def test_minimal_configuration_is_enough_to_start() -> None:
    assert _settings().embedding.model_name


def test_settings_reject_unknown_variable() -> None:
    # Опечатка в имени иначе тихо превращается в значение по умолчанию, и
    # расхождение вскрывается уже в бою.
    with pytest.raises(ValidationError):
        _settings(embedding={"versionn": "1.0.0"})


def test_secrets_are_not_printed() -> None:
    # Настройки печатаются в логах старта: незакрытое значение уезжает туда
    # навсегда.
    printed = repr(_settings())

    assert "secret" not in printed


def test_message_timeout_must_be_below_consumer_timeout() -> None:
    # Таймер потребителя идёт с момента доставки: если он короче обработки,
    # брокер закрывает канал, и сообщения возвращаются мимо счётчика попыток.
    with pytest.raises(ValidationError, match="потребител"):
        _settings(
            processing={"message_timeout_s": 3600},
            rabbit={"url": MINIMAL["rabbit"]["url"], "consumer_timeout_ms": 60_000},
        )


def test_claim_lease_must_outlive_the_message() -> None:
    # Иначе захват протухнет посреди работы, и второй воркер начнёт ту же.
    with pytest.raises(ValidationError, match="захват"):
        _settings(
            processing={"message_timeout_s": 900},
            messaging={"claim_lease_s": 300},
        )


def test_embedding_dimensions_must_match_column_width() -> None:
    # Колонка объявлена миграцией: вектор другой ширины не сохранится, и
    # узнать об этом на старте дешевле, чем на первом документе.
    with pytest.raises(ValidationError, match="ширин"):
        _settings(embedding={"dimensions": 768})


def test_sparse_limit_cannot_exceed_the_index_limit() -> None:
    # HNSW в pgvector отказывается строиться при большем числе весов.
    with pytest.raises(ValidationError, match="разреженн"):
        _settings(embedding={"sparse_top_k": SPARSE_TOP_K + 1})


def test_rerank_pool_cannot_be_smaller_than_the_context() -> None:
    # В контекст попадает только переранжированное: просить больше фрагментов,
    # чем прошло реранкинг, значит просить несуществующее.
    with pytest.raises(ValidationError, match="реранк"):
        _settings(rerank={"top_n": 5}, context={"max_chunks": 20})


def test_defaults_match_the_declared_stack() -> None:
    settings = _settings()

    assert settings.embedding.model_name == "BAAI/bge-m3"
    assert settings.embedding.dimensions == DENSE_DIMENSIONS
    assert settings.rerank.model_name == "BAAI/bge-reranker-v2-m3"
    assert settings.llm.base_url == "https://api.deepinfra.com/v1/openai"


def test_settings_expose_every_documented_variable() -> None:
    # Справочник настроек собирается отсюда: переменная, о которой знает
    # только код, не попадёт ни в `.env.example`, ни в документацию.
    names = env_variable_names()

    assert "DATABASE__DSN" in names
    assert "EMBEDDING__MODEL_NAME" in names
    assert "LLM__API_KEY" in names
    assert names == tuple(sorted(names))


def test_every_section_is_frozen() -> None:
    # Настройка, изменённая в рантайме, разъезжается с тем, что записано в
    # логе старта.
    settings = _settings()

    with pytest.raises(ValidationError):
        settings.embedding.dimensions = 512  # type: ignore[misc]


def test_processing_config_is_assembled_for_the_application_layer() -> None:
    # Прикладной слой не читает окружение: он получает готовую конфигурацию,
    # уже прошедшую проверки.
    config = _settings().processing_config()

    assert config.embedding.version.major >= 1
    assert config.retrieval.rrf_k > 0
    assert config.context.token_budget > 0
