"""Классификация прикладных ошибок: она решает ack, retry и DLQ."""

from __future__ import annotations

import pytest

from ai_worker.application import errors

pytestmark = pytest.mark.unit


def _application_error_classes() -> list[type[errors.ApplicationError]]:
    return [
        value
        for value in vars(errors).values()
        if isinstance(value, type) and issubclass(value, errors.ApplicationError)
    ]


def test_every_error_is_either_transient_or_permanent_or_chunk_level() -> None:
    # Ошибка без класса означает сообщение, судьбу которого подписчик решить не
    # может: ни подтвердить, ни повторить, ни отправить в разбор.
    unclassified = [
        error_class.__name__
        for error_class in _application_error_classes()
        if error_class is not errors.ApplicationError
        and not issubclass(
            error_class,
            errors.TransientError | errors.PermanentError | errors.ChunkLevelError,
        )
    ]

    assert not unclassified, f"ошибки вне классификации: {unclassified}"


def test_transient_and_permanent_errors_are_disjoint() -> None:
    # Пересечение означало бы, что одна ошибка одновременно требует повтора и
    # запрещает его.
    both = [
        error_class.__name__
        for error_class in _application_error_classes()
        if issubclass(error_class, errors.TransientError)
        and issubclass(error_class, errors.PermanentError)
    ]

    assert not both


def test_chunk_level_error_is_not_permanent() -> None:
    # Сбой одного чанка не обязан валить документ: он даёт частичный индекс, а
    # не отказ.
    assert not issubclass(errors.ChunkLevelError, errors.PermanentError)
    assert not issubclass(errors.ChunkLevelError, errors.TransientError)


def test_every_error_carries_a_code() -> None:
    codes = [error_class.code for error_class in _application_error_classes()]

    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    assert not duplicates, f"коды ошибок повторяются: {duplicates}"


def test_error_keeps_message_and_context() -> None:
    error = errors.StorageUnavailable("база недоступна", context={"attempt": 2})

    assert error.message == "база недоступна"
    assert error.context == {"attempt": 2}


def test_error_renders_itself_for_the_log() -> None:
    # Это представление уезжает в заголовки DLQ: без кода и данных копию
    # сообщения в очереди разбора нечем объяснить.
    error = errors.DuplicateRecord("уже есть", context={"event_id": "abc"})

    assert error.to_dict() == {
        "code": "duplicate_record",
        "message": "уже есть",
        "context": {"event_id": "abc"},
    }


def test_retry_hint_lives_only_on_transient_errors() -> None:
    # Подсказка о задержке имеет смысл ровно там, где повтор разрешён.
    assert errors.StorageUnavailable("x").retry_after_s is not None
    assert not hasattr(errors.InvariantRejectedByStorage("x"), "retry_after_s")
