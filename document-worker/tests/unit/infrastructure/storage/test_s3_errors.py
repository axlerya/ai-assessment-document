"""Трансляция ошибок S3 во временные и неисправимые.

От ветви зависит судьба сообщения: временная уходит в retry, неисправимая в DLQ.
Ошибку хранилища здесь не повторяют вечно и не хоронят по ошибке.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from document_worker.application.errors import (
    PermanentError,
    SourceObjectNotFoundError,
    StorageAccessDeniedError,
    StorageThrottledError,
    StorageUnavailableError,
    TransientError,
)
from document_worker.infrastructure.storage.errors import translate_storage_error

pytestmark = pytest.mark.unit

ENDPOINT = "http://localhost:9000"


def _client_error(code: str, status: int) -> ClientError:
    response: Any = {
        "Error": {"Code": code, "Message": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, "HeadObject")


@pytest.mark.parametrize(
    "code",
    ["NoSuchKey", "NoSuchBucket", "404"],
)
def test_absence_becomes_source_object_not_found(code: str) -> None:
    translated = translate_storage_error(_client_error(code, 404))

    assert isinstance(translated, SourceObjectNotFoundError)
    assert isinstance(translated, PermanentError)


@pytest.mark.parametrize(
    "code",
    ["AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "403"],
)
def test_authorization_failures_become_access_denied(code: str) -> None:
    translated = translate_storage_error(_client_error(code, 403))

    assert isinstance(translated, StorageAccessDeniedError)
    assert isinstance(translated, PermanentError)


@pytest.mark.parametrize("code", ["SlowDown", "RequestTimeout", "429"])
def test_throttling_becomes_transient(code: str) -> None:
    translated = translate_storage_error(_client_error(code, 503))

    assert isinstance(translated, StorageThrottledError)
    assert isinstance(translated, TransientError)


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_become_storage_unavailable(status: int) -> None:
    translated = translate_storage_error(_client_error("InternalError", status))

    assert isinstance(translated, StorageUnavailableError)
    assert isinstance(translated, TransientError)


def test_unreachable_endpoint_is_transient() -> None:
    translated = translate_storage_error(EndpointConnectionError(endpoint_url=ENDPOINT))

    assert isinstance(translated, StorageUnavailableError)


def test_read_timeout_is_transient() -> None:
    translated = translate_storage_error(ReadTimeoutError(endpoint_url=ENDPOINT))

    assert isinstance(translated, StorageUnavailableError)


def test_closed_connection_is_transient() -> None:
    translated = translate_storage_error(ConnectionClosedError(endpoint_url=ENDPOINT))

    assert isinstance(translated, StorageUnavailableError)


def test_unknown_client_error_is_permanent() -> None:
    # Повторять непонятное без предела запрещено.
    translated = translate_storage_error(_client_error("InvalidRequest", 400))

    assert isinstance(translated, PermanentError)
    assert not isinstance(translated, TransientError)


def test_translated_error_keeps_the_original_as_cause() -> None:
    original = _client_error("SlowDown", 503)

    translated = translate_storage_error(original)

    assert translated.__cause__ is original


def test_translated_error_carries_the_code_in_context() -> None:
    translated = translate_storage_error(_client_error("SlowDown", 503))

    assert translated.context["code"] == "SlowDown"
