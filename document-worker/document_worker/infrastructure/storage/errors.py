"""Ошибки S3 → прикладные.

От ветви зависит судьба сообщения: временная уходит в retry, неисправимая в DLQ.
Неизвестный код считается неисправимым — повторять непонятное без предела
запрещено уставом.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from botocore.exceptions import BotoCoreError, ClientError

from document_worker.application.errors import (
    PermanentError,
    SourceObjectNotFoundError,
    StorageAccessDeniedError,
    StorageThrottledError,
    StorageUnavailableError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from document_worker.application.errors import ApplicationError

_NOT_FOUND: Final[frozenset[str]] = frozenset({"NoSuchKey", "NoSuchBucket", "404"})
_ACCESS_DENIED: Final[frozenset[str]] = frozenset(
    {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "403"}
)
_THROTTLED: Final[frozenset[str]] = frozenset(
    {"SlowDown", "RequestTimeout", "RequestTimeTooSkewed", "429"}
)
_SERVER_ERROR: Final[int] = 500

_BY_CODE: Final[Mapping[frozenset[str], type[ApplicationError]]] = {
    _NOT_FOUND: SourceObjectNotFoundError,
    _ACCESS_DENIED: StorageAccessDeniedError,
    _THROTTLED: StorageThrottledError,
}


def translate_storage_error(error: Exception) -> ApplicationError:
    """Переводит ошибку S3 в прикладную, сохраняя исходную как причину."""
    if isinstance(error, ClientError):
        translated = _from_client_error(error)
    elif isinstance(error, BotoCoreError):
        # Сеть до хранилища не дошла: адрес, таймаут, оборванное соединение.
        translated = StorageUnavailableError(
            str(error), context={"code": type(error).__name__}
        )
    else:  # pragma: no cover — сюда попадает только неожиданный тип
        translated = PermanentError(str(error), context={"code": type(error).__name__})
    translated.__cause__ = error
    return translated


def _from_client_error(error: ClientError) -> ApplicationError:
    response = error.response
    code = str(response.get("Error", {}).get("Code", ""))
    status = int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
    context: dict[str, object] = {"code": code, "status": status}

    for codes, failure in _BY_CODE.items():
        if code in codes:
            return failure(str(error), context=context)
    if status >= _SERVER_ERROR:
        return StorageUnavailableError(str(error), context=context)
    return PermanentError(str(error), context=context)
