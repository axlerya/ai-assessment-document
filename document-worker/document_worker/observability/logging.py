"""Структурные логи с обязательным редактированием.

Сервис обрабатывает юридические документы, и утечка в логи необратима: их
хранилище живёт годами, доступ к нему шире, чем к базе, и вырезать оттуда
одну строку задним числом нельзя.

Поэтому редактирование работает по чёрному списку ключей и по длине значения
сразу, а не «когда понадобится». Отдельная забота — сообщения библиотек
разбора PDF: pikepdf и pdfplumber кладут в текст исключения куски содержимого,
и пересказ такого исключения в лог равен публикации документа.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import structlog

if TYPE_CHECKING:
    from collections.abc import MutableMapping

REDACTED: Final[str] = "[вырезано]"
MAX_VALUE_LENGTH: Final[int] = 512
TRUNCATED_SUFFIX: Final[str] = "…[усечено]"

# Ключи, значение которых не попадает в лог ни при каких условиях.
FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "text",
        "content",
        "page_text",
        "raw_text",
        "body",
        "payload",
        "password",
        "secret",
        "token",
        "access_key",
        "secret_key",
        "dsn",
        "url",
        "filename",
        "file_name",
        "original_filename",
    }
)

# Библиотеки разбора PDF цитируют содержимое в тексте исключения.
UNSAFE_EXCEPTION_SOURCES: Final[frozenset[str]] = frozenset(
    {"pikepdf", "pdfplumber", "pdfminer", "pypdfium2"}
)
UNSAFE_EXCEPTION_MESSAGE: Final[str] = "ошибка разбора документа"

PRODUCTION_ENVIRONMENTS: Final[frozenset[str]] = frozenset({"ci", "production"})


class UnsafeLogLevelError(ValueError):
    """Уровень логирования недопустим в этом окружении."""


def redact(value: object, *, key: str | None = None) -> object:
    """Приводит значение к виду, пригодному для лога."""
    if key is not None and key.lower() in FORBIDDEN_KEYS:
        return REDACTED
    if isinstance(value, str):
        return (
            value
            if len(value) <= MAX_VALUE_LENGTH
            else value[:MAX_VALUE_LENGTH] + TRUNCATED_SUFFIX
        )
    if isinstance(value, Mapping):
        return {name: redact(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    return value


def redacting_processor(
    logger: object,
    name: str,
    event: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Вырезает запрещённые поля и усекает длинные значения."""
    del logger, name
    return {key: redact(value, key=key) for key, value in event.items()}


def safe_exception_message(error: BaseException) -> str:
    """Текст исключения, пригодный для лога.

    Сообщение библиотеки разбора заменяется целиком: в нём цитируется
    содержимое документа, а класс исключения причину описывает достаточно.
    """
    module = type(error).__module__.split(".")[0]
    if module in UNSAFE_EXCEPTION_SOURCES:
        return f"{UNSAFE_EXCEPTION_MESSAGE}: {type(error).__name__}"
    text = str(error)
    return (
        text
        if len(text) <= MAX_VALUE_LENGTH
        else text[:MAX_VALUE_LENGTH] + TRUNCATED_SUFFIX
    )


def ensure_level_is_allowed(level: str, *, environment: str) -> None:
    """Не даёт включить отладочный уровень там, где логи хранятся.

    Raises:
        UnsafeLogLevelError: Отладочный уровень в окружении, где логи хранятся.
    """
    if environment.lower() in PRODUCTION_ENVIRONMENTS and level.upper() == "DEBUG":
        raise UnsafeLogLevelError(
            f"уровень DEBUG недопустим в окружении {environment}: "
            "отладочные записи цитируют содержимое документов"
        )


def configure_logging(*, level: str = "INFO", environment: str = "local") -> None:
    """Настраивает структурные логи процесса."""
    ensure_level_is_allowed(level, environment=environment)
    logging.basicConfig(
        format="%(message)s", level=getattr(logging, level.upper()), force=True
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redacting_processor,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        cache_logger_on_first_use=True,
    )
