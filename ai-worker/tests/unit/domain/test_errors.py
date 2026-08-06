"""Доменные ошибки: одна иерархия, без признака повторяемости."""

from __future__ import annotations

import pytest

from ai_worker.domain import errors

pytestmark = pytest.mark.unit

# Домен не знает про доставку сообщений: решение о retry и DLQ принимает
# application. Признак повторяемости здесь означал бы второй источник этого
# решения, и они разошлись бы на первой же нестандартной ошибке.
RETRYABILITY_NAMES = (
    "retryable",
    "is_retryable",
    "retry",
    "retry_after",
    "retry_after_s",
    "transient",
    "is_transient",
    "permanent",
    "is_permanent",
)


def _domain_error_classes() -> list[type[errors.DomainError]]:
    return [
        value
        for value in vars(errors).values()
        if isinstance(value, type) and issubclass(value, errors.DomainError)
    ]


def test_every_error_descends_from_the_single_base() -> None:
    classes = _domain_error_classes()

    assert classes, "в модуле нет ни одной доменной ошибки"
    for error_class in classes:
        assert issubclass(error_class, Exception)


def test_domain_errors_do_not_declare_retryability() -> None:
    offenders = [
        f"{error_class.__name__}.{name}"
        for error_class in _domain_error_classes()
        for name in RETRYABILITY_NAMES
        if hasattr(error_class, name)
    ]

    assert not offenders, "домен не вправе знать про повторяемость: " + ", ".join(
        offenders
    )


def test_every_error_has_its_own_code() -> None:
    classes = _domain_error_classes()

    codes = [error_class.code for error_class in classes]
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    assert not duplicates, (
        f"код ошибки обязан отличать её от остальных, повторяются: {duplicates}"
    )


def test_error_keeps_message_and_context() -> None:
    error = errors.InvariantViolation("нарушен инвариант", context={"page": 3})

    assert error.message == "нарушен инвариант"
    assert error.context == {"page": 3}


def test_context_is_a_copy_and_cannot_be_changed_through_the_error() -> None:
    source = {"page": 3}

    error = errors.InvariantViolation("нарушен инвариант", context=source)
    source["page"] = 99

    assert error.context == {"page": 3}


def test_error_renders_itself_for_the_log() -> None:
    error = errors.InvalidVector("размерность не та", context={"actual": 512})

    assert error.to_dict() == {
        "code": "invalid_vector",
        "message": "размерность не та",
        "context": {"actual": 512},
    }


@pytest.mark.parametrize(
    "error_class",
    [
        errors.InvalidIdentifier,
        errors.InvalidVersion,
        errors.InvalidVector,
        errors.InvalidScore,
        errors.InvalidTextSpan,
    ],
)
def test_value_object_errors_share_one_ancestor(
    error_class: type[errors.DomainError],
) -> None:
    # Один предок позволяет ловить «объект нельзя построить» одним except,
    # не перечисляя все ошибки значений поимённо.
    assert issubclass(error_class, errors.InvalidValueObject)
