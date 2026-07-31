"""AMQP-заголовки доставки и чтение номера попытки.

Счётчик попыток — собственный заголовок, а не `x-death`. Тот агрегирует счёт
по паре «очередь, причина», при лестнице из пяти очередей даёт пять записей с
единицей, при обычной публикации брокером не создаётся вовсе и не входит в
AMQP 0-9-1: его структура отличается между версиями брокера.
"""

from __future__ import annotations

from typing import Any, Final

HEADER_ATTEMPT: Final[str] = "x-attempt"
HEADER_RETRY_LEVEL: Final[str] = "x-retry-level"
HEADER_FIRST_FAILED_AT: Final[str] = "x-first-failed-at"
HEADER_ERROR_TYPE: Final[str] = "x-error-type"
HEADER_ERROR_CODE: Final[str] = "x-error-code"
HEADER_ERROR_MESSAGE: Final[str] = "x-error-message"
HEADER_ATTEMPTS_EXHAUSTED: Final[str] = "x-attempts-exhausted"

# Заголовки брокера про возвраты помимо нашего повтора: `x-death`,
# `x-first-death-*`, `x-last-death-*`. В копию не переносятся — иначе они
# уехали бы в неё обычными заголовками и врали бы про число возвратов.
DEATH_MARKER: Final[str] = "death"

MAX_ERROR_MESSAGE_LENGTH: Final[int] = 512


def current_attempt(headers: dict[str, Any]) -> int:
    """Сколько попыток уже израсходовано. Первая доставка даёт ноль."""
    raw = headers.get(HEADER_ATTEMPT)
    if isinstance(raw, bool) or not isinstance(raw, int | str):
        return 0
    try:
        attempt = int(raw)
    except ValueError:
        return 0
    return max(attempt, 0)


def without_death_headers(headers: dict[str, Any]) -> dict[str, Any]:
    """Копия заголовков без следов брокерских возвратов."""
    return {name: value for name, value in headers.items() if DEATH_MARKER not in name}
