"""Тесты доменной иерархии ошибок."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from document_worker import domain
from document_worker.domain.errors import (
    ChecksumMismatch,
    DocumentRejected,
    DocumentTooLarge,
    DomainError,
    InvariantViolation,
    UnsupportedDocumentFormat,
)

pytestmark = pytest.mark.unit

# Признак повторяемости — забота application/errors.py.
RETRYABILITY_ATTRIBUTES = (
    "retryable",
    "is_retryable",
    "recoverable",
    "is_recoverable",
    "transient",
    "is_transient",
)


def _import_all_domain_modules() -> None:
    """Импортирует весь пакет domain, иначе __subclasses__ не увидит потомков."""
    for module in pkgutil.walk_packages(domain.__path__, f"{domain.__name__}."):
        importlib.import_module(module.name)


def _all_subclasses(root: type[DomainError]) -> set[type[DomainError]]:
    found: set[type[DomainError]] = set()
    for subclass in root.__subclasses__():
        found.add(subclass)
        found |= _all_subclasses(subclass)
    return found


def test_all_domain_errors_expose_unique_codes() -> None:
    _import_all_domain_modules()
    errors = _all_subclasses(DomainError)

    assert errors, "иерархия доменных ошибок пуста"

    codes: dict[str, str] = {}
    duplicates: list[str] = []
    for error in sorted(errors, key=lambda cls: cls.__name__):
        code = error.code
        assert isinstance(code, str), f"{error.__name__}: код обязан быть строкой"
        assert code, f"{error.__name__}: пустой код"
        if code in codes:
            duplicates.append(f"{code}: {codes[code]} и {error.__name__}")
        codes[code] = error.__name__

    assert not duplicates, f"код ошибки обязан быть уникальным: {duplicates}"


def test_domain_errors_do_not_declare_retryability() -> None:
    _import_all_domain_modules()
    errors: set[type[DomainError]] = {DomainError, *_all_subclasses(DomainError)}

    offenders = [
        f"{error.__name__}.{attribute}"
        for error in errors
        for attribute in RETRYABILITY_ATTRIBUTES
        if hasattr(error, attribute)
    ]

    assert not offenders, (
        f"признак повторяемости объявляется в application, а не в домене: {offenders}"
    )


def test_unsupported_format_carries_mime_type_and_supported_list() -> None:
    error = UnsupportedDocumentFormat("application/zip", supported=["application/pdf"])

    assert error.context["mime_type"] == "application/zip"
    assert error.context["supported"] == ("application/pdf",)


def test_to_dict_contains_code_message_and_context() -> None:
    error = UnsupportedDocumentFormat("application/zip", supported=["application/pdf"])

    payload = error.to_dict()

    assert payload["code"] == "unsupported_format"
    assert payload["message"]
    assert payload["context"] == dict(error.context)


def test_document_too_large_carries_actual_size_and_limit() -> None:
    error = DocumentTooLarge(actual_bytes=209_715_200, limit_bytes=104_857_600)

    assert error.context == {
        "actual_bytes": 209_715_200,
        "limit_bytes": 104_857_600,
    }


def test_checksum_mismatch_carries_both_sums() -> None:
    error = ChecksumMismatch(expected="a" * 64, actual="b" * 64)

    assert error.context == {"expected": "a" * 64, "actual": "b" * 64}


def test_document_rejected_and_invariant_violation_are_disjoint_branches() -> None:
    assert not issubclass(DocumentRejected, InvariantViolation)
    assert not issubclass(InvariantViolation, DocumentRejected)


def test_domain_error_context_is_empty_mapping_by_default() -> None:
    error = DomainError("что-то пошло не так")

    assert error.context == {}
    assert str(error) == "что-то пошло не так"
