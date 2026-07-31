"""Редактирование логов.

Утечка в логи необратима: их хранилище живёт годами, доступ к нему шире, чем
к базе, и вырезать оттуда одну строку задним числом нельзя.
"""

from __future__ import annotations

import pytest

from document_worker.observability.logging import (
    MAX_VALUE_LENGTH,
    REDACTED,
    UnsafeLogLevelError,
    ensure_level_is_allowed,
    redact,
    redacting_processor,
    safe_exception_message,
)

pytestmark = pytest.mark.unit

PAGE_TEXT = "Настоящим договором стороны устанавливают, что поставка"


@pytest.mark.parametrize(
    "key", ["text", "content", "page_text", "password", "dsn", "filename"]
)
def test_forbidden_key_is_never_rendered(key: str) -> None:
    assert redact(PAGE_TEXT, key=key) == REDACTED


def test_long_value_is_truncated() -> None:
    # Гигантская строка в логе — это не только утечка, но и счёт за хранилище.
    long = "я" * (MAX_VALUE_LENGTH * 3)

    rendered = redact(long, key="message")

    assert isinstance(rendered, str)
    assert len(rendered) < len(long)
    assert rendered.startswith("я" * 10)


def test_nested_values_are_redacted_too() -> None:
    # Содержимое чаще всего приезжает вложенным в контекст ошибки.
    event = {"context": {"page_text": PAGE_TEXT, "page_number": 3}}

    rendered = redacting_processor(None, "info", event)

    assert rendered["context"]["page_text"] == REDACTED
    assert rendered["context"]["page_number"] == 3


def test_values_in_lists_are_redacted() -> None:
    event = {"pages": [{"content": PAGE_TEXT}, {"content": PAGE_TEXT}]}

    rendered = redacting_processor(None, "info", event)

    assert [page["content"] for page in rendered["pages"]] == [REDACTED, REDACTED]


def test_safe_fields_survive() -> None:
    event = {"document_id": "abc", "pages_total": 12}

    assert redacting_processor(None, "info", event) == event


def test_pdf_library_message_is_replaced() -> None:
    # pikepdf и pdfplumber цитируют содержимое в тексте исключения, и пересказ
    # такого исключения в лог равен публикации документа.
    error = _pdf_error(f"cannot parse object near {PAGE_TEXT}")

    rendered = safe_exception_message(error)

    assert PAGE_TEXT not in rendered
    assert "PdfError" in rendered


def test_our_own_message_survives() -> None:
    assert safe_exception_message(ValueError("страниц больше предела")) == (
        "страниц больше предела"
    )


def test_long_message_of_our_own_is_truncated() -> None:
    assert len(safe_exception_message(ValueError("я" * 5_000))) < 5_000


def test_debug_is_rejected_where_logs_are_kept() -> None:
    with pytest.raises(UnsafeLogLevelError):
        ensure_level_is_allowed("DEBUG", environment="production")


@pytest.mark.parametrize(
    ("level", "environment"),
    [("INFO", "production"), ("DEBUG", "local"), ("WARNING", "ci")],
)
def test_allowed_combinations_pass(level: str, environment: str) -> None:
    ensure_level_is_allowed(level, environment=environment)


def _pdf_error(message: str) -> Exception:
    import pikepdf  # noqa: PLC0415 — нужен именно класс библиотеки

    return pikepdf.PdfError(message)
