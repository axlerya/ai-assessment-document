"""Перевод отказов драйвера в прикладную классификацию.

Без перевода подписчик видит `DBAPIError` и не может решить: повторить,
подтвердить или отправить в разбор. Разница между «база моргнула» и «строка
противоречит схеме» тут стоит либо потерянного документа, либо вечного цикла
повторов.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    InterfaceError,
    OperationalError,
)

from ai_worker.application.errors import (
    ApplicationError,
    PermanentError,
    TransientError,
)
from ai_worker.infrastructure.persistence.errors import translate

pytestmark = pytest.mark.unit


class _Pg(Exception):
    """Подделка исключения драйвера: у него есть только код состояния."""

    def __init__(self, sqlstate: str) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


def _integrity(sqlstate: str) -> IntegrityError:
    return IntegrityError("INSERT ...", {}, _Pg(sqlstate))


def _operational(sqlstate: str) -> OperationalError:
    return OperationalError("SELECT ...", {}, _Pg(sqlstate))


@pytest.mark.parametrize(
    "sqlstate",
    [
        "23514",  # нарушение CHECK
        "23503",  # нарушение внешнего ключа
        "23502",  # NOT NULL
    ],
)
def test_broken_invariant_is_permanent(sqlstate: str) -> None:
    # Строка противоречит схеме: повтор даст ровно то же самое.
    translated = translate(_integrity(sqlstate))

    assert isinstance(translated, PermanentError)


def test_unique_violation_is_translated_to_duplicate_not_to_internal_error() -> None:
    # Дубль — ожидаемый исход повторной доставки, а не внутренняя ошибка: по
    # нему принимается решение «уже сделано», а не «всё сломалось».
    translated = translate(_integrity("23505"))

    assert translated.code == "duplicate_record"
    assert isinstance(translated, PermanentError)


@pytest.mark.parametrize(
    "sqlstate",
    [
        "40001",  # сбой сериализации
        "40P01",  # взаимоблокировка
        "57P01",  # администратор завершил соединение
        "53300",  # соединений слишком много
    ],
)
def test_recoverable_database_failures_are_transient(sqlstate: str) -> None:
    translated = translate(_operational(sqlstate))

    assert isinstance(translated, TransientError)


def test_lost_connection_is_transient() -> None:
    translated = translate(
        InterfaceError("connection was closed", {}, OSError("closed"))
    )

    assert isinstance(translated, TransientError)


def test_unknown_database_failure_is_transient_not_silent() -> None:
    # Неизвестный код — повод повторить с ограничением попыток, а не считать
    # документ обработанным.
    translated = translate(DBAPIError("SELECT ...", {}, _Pg("XX000")))

    assert isinstance(translated, TransientError)


def test_translation_keeps_the_original_cause() -> None:
    original = _integrity("23514")

    translated = translate(original)

    assert translated.__cause__ is original


def test_translation_records_the_sqlstate() -> None:
    # Без кода состояния разбор инцидента упирается в текст сообщения драйвера.
    translated = translate(_integrity("23514"))

    assert translated.context["sqlstate"] == "23514"


def test_application_error_passes_through_untouched() -> None:
    # Ошибка, уже прошедшая классификацию, второй раз не переводится: иначе
    # `PermanentError` мог бы стать `TransientError` и уйти в вечный повтор.
    original = PermanentError("уже классифицирована")

    assert translate(original) is original


def test_translated_error_is_an_application_error() -> None:
    assert isinstance(translate(_operational("40001")), ApplicationError)
