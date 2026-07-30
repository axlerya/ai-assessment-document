"""Тесты классификации ошибок обработки."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from document_worker import application
from document_worker.application.errors import (
    ApplicationError,
    DocumentNotFoundError,
    DomainInvariantViolationError,
    PageLevelError,
    PermanentError,
    TransientError,
    translate_domain_error,
)
from document_worker.domain.errors import DomainError, UnsupportedDocumentFormat

pytestmark = pytest.mark.unit


def _import_all_application_modules() -> None:
    for module in pkgutil.walk_packages(
        application.__path__, f"{application.__name__}."
    ):
        importlib.import_module(module.name)


def _all_subclasses(root: type[ApplicationError]) -> set[type[ApplicationError]]:
    found: set[type[ApplicationError]] = set()
    for subclass in root.__subclasses__():
        found.add(subclass)
        found |= _all_subclasses(subclass)
    return found


def test_transient_and_permanent_errors_are_disjoint() -> None:
    _import_all_application_modules()

    overlap = _all_subclasses(TransientError) & _all_subclasses(PermanentError)

    assert not overlap


def test_page_level_error_does_not_inherit_permanent_error() -> None:
    assert not issubclass(PageLevelError, PermanentError)
    assert not issubclass(PageLevelError, TransientError)


def test_every_application_error_declares_unique_code() -> None:
    _import_all_application_modules()
    errors = _all_subclasses(ApplicationError)

    assert errors, "иерархия прикладных ошибок пуста"
    codes = [error.code for error in errors]
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    assert not duplicates, f"код ошибки обязан быть уникальным: {duplicates}"


def test_error_classification_covers_all_declared_codes() -> None:
    _import_all_application_modules()
    classified = (
        _all_subclasses(TransientError)
        | _all_subclasses(PermanentError)
        | _all_subclasses(PageLevelError)
        | {TransientError, PermanentError, PageLevelError}
    )

    unclassified = _all_subclasses(ApplicationError) - classified

    assert not unclassified, f"ошибка вне трёх ветвей: {unclassified}"


def test_domain_error_maps_to_permanent_error() -> None:
    translated = translate_domain_error(DomainError("что-то не так"))

    assert isinstance(translated, DomainInvariantViolationError)
    assert isinstance(translated, PermanentError)


def test_unsupported_format_maps_to_its_own_permanent_error() -> None:
    translated = translate_domain_error(
        UnsupportedDocumentFormat("application/zip", supported=["application/pdf"])
    )

    assert isinstance(translated, PermanentError)
    assert translated.code == "unsupported_media_type"


def test_translation_keeps_original_as_cause() -> None:
    original = DomainError("что-то не так")

    translated = translate_domain_error(original)

    assert translated.__cause__ is original


def test_document_not_found_is_transient_with_retry_delay() -> None:
    # Сообщение может обогнать коммит сервиса приёма файлов, поэтому отсутствие
    # документа это гонка, а не приговор.
    error = DocumentNotFoundError("документа ещё нет")

    assert isinstance(error, TransientError)
    assert error.retry_after_s == 5.0


def test_error_exposes_message_and_dict_form() -> None:
    error = DocumentNotFoundError("документа ещё нет", context={"document_id": "d-1"})

    payload = error.to_dict()

    assert error.message == "документа ещё нет"
    assert payload["code"] == "document_not_found"
    assert payload["context"] == {"document_id": "d-1"}


def test_page_level_error_carries_page_number() -> None:
    error = PageLevelError("страница не прочитана", page_number=14)

    assert error.page_number == 14
    assert error.context["page_number"] == 14
