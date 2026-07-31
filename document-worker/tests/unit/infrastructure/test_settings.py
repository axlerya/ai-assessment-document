"""Конфигурация приложения.

Настройки собираются на любой платформе без файла окружения: сервис
разрабатывается на Windows, а работает в Linux, и падение конструктора из-за
жёстко зашитого `/run/secrets` останавливало бы разработку.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from document_worker.infrastructure.config.settings import AppSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

REQUIRED_ENV = {
    "DATABASE__DSN": "postgresql+asyncpg://worker:secret@postgres:5432/documents",
    "RABBIT__URL": "amqp://worker:secret@rabbitmq:5672/documents",
    "S3__ENDPOINT_URL": "http://minio:9000",
    "S3__ACCESS_KEY": "minio-user",
    "S3__SECRET_KEY": "minio-password",
}
ENV_EXAMPLE = Path(__file__).resolve().parents[3] / ".env.example"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Окружение с обязательными переменными и без посторонних."""
    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_settings_construct_on_current_platform(env: pytest.MonkeyPatch) -> None:
    del env

    settings = AppSettings()

    assert settings.database.dsn
    assert settings.processing.temp_dir.is_absolute()


def test_unknown_variable_is_rejected(env: pytest.MonkeyPatch) -> None:
    # Опечатка в имени переменной иначе тихо превращается в значение по
    # умолчанию, и расхождение вскрывается в бою.
    env.setenv("PROCESSING__DOCUMENT_TIMEOUTS", "3600")

    with pytest.raises(ValidationError):
        AppSettings()


def test_settings_are_frozen(env: pytest.MonkeyPatch) -> None:
    del env
    settings = AppSettings()

    with pytest.raises(ValidationError):
        settings.database.pool_size = 20  # type: ignore[misc]


def test_secrets_are_not_rendered(env: pytest.MonkeyPatch) -> None:
    # Настройки печатаются в логах старта, и пароль в них уезжает навсегда.
    del env
    settings = AppSettings()

    rendered = repr(settings)

    assert "secret" not in rendered
    assert "minio-password" not in rendered


def test_dsn_secret_is_readable_where_needed(env: pytest.MonkeyPatch) -> None:
    del env
    settings = AppSettings()

    assert settings.database.dsn.get_secret_value().startswith("postgresql+asyncpg://")


def test_timeouts_are_validated_together(env: pytest.MonkeyPatch) -> None:
    # Лиз захвата короче таймаута документа делает каждую повторную доставку
    # возобновлением параллельно живому воркеру.
    env.setenv("MESSAGING__CLAIM_LEASE_S", "60")
    env.setenv("PROCESSING__DOCUMENT_TIMEOUT_S", "3600")

    with pytest.raises(ValidationError):
        AppSettings()


def test_pool_fits_the_concurrency(env: pytest.MonkeyPatch) -> None:
    # Соединений меньше, чем одновременно обрабатываемых документов, — это
    # гарантированное ожидание в пуле на каждой странице.
    env.setenv("DATABASE__POOL_SIZE", "1")
    env.setenv("PROCESSING__MAX_CONCURRENCY", "8")

    with pytest.raises(ValidationError):
        AppSettings()


def test_processing_config_is_derived_from_settings(env: pytest.MonkeyPatch) -> None:
    del env
    settings = AppSettings()

    config = settings.processing_config()

    assert config.consumer_name
    assert config.document_timeout_s == settings.processing.document_timeout_s


def test_env_example_covers_every_variable(env: pytest.MonkeyPatch) -> None:
    # Пример окружения — единственная документация настроек до этапа доков;
    # разъехавшись с кодом, он вводит в заблуждение вернее, чем его отсутствие.
    del env
    documented = {
        match.group(1)
        for match in re.finditer(
            r"^#?\s*([A-Z][A-Z0-9_]*__[A-Z0-9_]+)=",
            ENV_EXAMPLE.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    }

    assert _variable_names(AppSettings()) <= documented


def _variable_names(settings: AppSettings) -> set[str]:
    names: set[str] = set()
    for section, value in settings:
        if not hasattr(value, "model_fields"):
            continue
        names |= {
            f"{section.upper()}__{field.upper()}" for field in type(value).model_fields
        }
    return names
