"""Подписчик обработки документа. Бизнес-логики здесь нет.

Обработчик делает ровно четыре вещи: читает номер попытки, переводит сообщение
в команду, вызывает прикладной сценарий и превращает его исход в действие с
сообщением. Фиксацию отказа выполняет `ProcessDocument` внутри своей
транзакции — из подписчика она не вызывается ни в одной ветке: двойная
фиксация и есть та ошибка, из-за которой обработанный документ помечался
отказом.

Каждая ветка обязана завершиться подтверждением, отклонением или сознательным
пробросом: неподтверждённое сообщение висит до закрытия канала.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from faststream import AckPolicy
from faststream.rabbit import RabbitExchange, RabbitQueue, RabbitRouter
from faststream.rabbit.annotations import RabbitMessage

from document_worker.application.dto.results import ProcessDocumentResult
from document_worker.application.errors import PermanentError, TransientError
from document_worker.domain.value_objects.enums import DocumentStatus
from document_worker.presentation.messaging.contracts.commands import (
    ProcessDocumentRequestedV1,
)
from document_worker.presentation.messaging.headers import (
    HEADER_ATTEMPT,
    HEADER_ATTEMPTS_EXHAUSTED,
    HEADER_ERROR_CODE,
    HEADER_ERROR_MESSAGE,
    HEADER_ERROR_TYPE,
    HEADER_FIRST_FAILED_AT,
    HEADER_RETRY_LEVEL,
    MAX_ERROR_MESSAGE_LENGTH,
    current_attempt,
    without_death_headers,
)
from document_worker.presentation.messaging.mappers import to_process_document_command
from document_worker.presentation.messaging.middlewares.decode_guard import (
    DecodeGuardMiddleware,
)
from document_worker.presentation.messaging.ports import (
    DocumentProcessor,
    MessageRetrier,
)

SUBSCRIBER_TITLE = "document.process.requested"


def build_process_document_router(  # noqa: PLR0913 — подписчик собирается из всех этих частей
    *,
    queue: RabbitQueue,
    exchange: RabbitExchange,
    processor: DocumentProcessor,
    retrier: MessageRetrier,
    default_bucket: str,
    max_retries: int,
) -> RabbitRouter:
    """Собирает роутер с единственным подписчиком обработки документа.

    Попыток обработки на одну больше, чем ступеней задержки: первая идёт без
    ожидания, каждая следующая — после своей ступени.
    """
    max_attempts = max_retries + 1
    router = RabbitRouter(middlewares=(DecodeGuardMiddleware,))

    @router.subscriber(
        queue,
        exchange,
        ack_policy=AckPolicy.MANUAL,
        no_reply=True,
        title=SUBSCRIBER_TITLE,
    )
    async def handle(
        payload: ProcessDocumentRequestedV1,
        message: RabbitMessage,
    ) -> None:
        await dispatch(
            payload,
            message,
            processor=processor,
            retrier=retrier,
            default_bucket=default_bucket,
            max_attempts=max_attempts,
            max_retries=max_retries,
        )

    return router


async def dispatch(  # noqa: PLR0913 — исход зависит от всех этих величин
    payload: ProcessDocumentRequestedV1,
    message: RabbitMessage,
    *,
    processor: DocumentProcessor,
    retrier: MessageRetrier,
    default_bucket: str,
    max_attempts: int,
    max_retries: int,
) -> None:
    """Проводит сообщение через прикладной сценарий и финализирует его."""
    spent = current_attempt(dict(message.headers))
    command = to_process_document_command(
        payload,
        default_bucket=default_bucket,
        attempt=spent + 1,
        max_attempts=max_attempts,
    )
    try:
        result = await processor.execute(command)
    except asyncio.CancelledError:
        # Остановка процесса: ни подтверждения, ни отклонения — сообщение
        # вернёт брокер по разрыву канала.
        raise
    except PermanentError as error:
        await _to_dlq(retrier, message, error=error, spent=spent)
    except TransientError as error:
        await _retry_or_dlq(
            retrier, message, error=error, spent=spent, max_retries=max_retries
        )
    except Exception as error:  # noqa: BLE001 — неизвестное считаем повторяемым
        await _retry_or_dlq(
            retrier, message, error=error, spent=spent, max_retries=max_retries
        )
    else:
        await _finalize(retrier, message, result)


async def _finalize(
    retrier: MessageRetrier,
    message: RabbitMessage,
    result: ProcessDocumentResult,
) -> None:
    if result.status is DocumentStatus.FAILED:
        # Отказ уже зафиксирован транзакцией: копия ложится в разбор, а
        # оригинал подтверждается — повторять нечего.
        headers = without_death_headers(dict(message.headers))
        headers[HEADER_ERROR_CODE] = result.failure_code or "unknown_error"
        await retrier.send_to_dlq(message.body, headers)
    await message.ack()


async def _retry_or_dlq(
    retrier: MessageRetrier,
    message: RabbitMessage,
    *,
    error: BaseException,
    spent: int,
    max_retries: int,
) -> None:
    if spent >= max_retries:
        await _to_dlq(retrier, message, error=error, spent=spent)
        return
    following = spent + 1
    await retrier.schedule(
        message.body,
        _retry_headers(message, error=error, attempt=following),
        attempt=following,
    )
    # Подтверждение только после подтверждённой публикации копии: иначе работа
    # исчезает вместе с сообщением.
    await message.ack()


async def _to_dlq(
    retrier: MessageRetrier,
    message: RabbitMessage,
    *,
    error: BaseException,
    spent: int,
) -> None:
    headers = _error_headers(message, error=error)
    headers[HEADER_ATTEMPT] = spent
    headers[HEADER_ATTEMPTS_EXHAUSTED] = True
    await retrier.send_to_dlq(message.body, headers)
    await message.ack()


def _retry_headers(
    message: RabbitMessage,
    *,
    error: BaseException,
    attempt: int,
) -> dict[str, Any]:
    headers = _error_headers(message, error=error)
    headers[HEADER_ATTEMPT] = attempt
    headers[HEADER_RETRY_LEVEL] = attempt
    return headers


def _error_headers(message: RabbitMessage, *, error: BaseException) -> dict[str, Any]:
    headers = without_death_headers(dict(message.headers))
    headers.setdefault(HEADER_FIRST_FAILED_AT, datetime.now(UTC).isoformat())
    headers[HEADER_ERROR_TYPE] = type(error).__name__
    headers[HEADER_ERROR_CODE] = getattr(type(error), "code", "unknown_error")
    headers[HEADER_ERROR_MESSAGE] = str(error)[:MAX_ERROR_MESSAGE_LENGTH]
    return headers
