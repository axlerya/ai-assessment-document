"""Ошибки обработки: три ветви, определяющие ack, retry и DLQ.

Домен своей повторяемости не знает, поэтому классификация живёт здесь.
Исключения библиотек транслируются в эти типы адаптерами инфраструктуры;
нетранслированное исключение не подменяется и долетает до presentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from document_worker.domain.errors import (
    ChecksumMismatch,
    CorruptedDocument,
    DocumentTooLarge,
    EmptyDocument,
    EncryptedDocument,
    UnsupportedDocumentFormat,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from document_worker.domain.errors import DomainError


class ApplicationError(Exception):
    """Базовая ошибка обработки."""

    code: ClassVar[str] = "application_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт ошибку с данными для лога и заголовков DLQ."""
        super().__init__(message)
        self._context: dict[str, object] = dict(context or {})

    @property
    def message(self) -> str:
        """Текст ошибки."""
        return str(self)

    @property
    def context(self) -> Mapping[str, object]:
        """Данные ошибки."""
        return dict(self._context)

    def to_dict(self) -> dict[str, object]:
        """Представление для структурного лога и заголовков сообщения."""
        return {
            "code": type(self).code,
            "message": self.message,
            "context": dict(self._context),
        }


class TransientError(ApplicationError):
    """Повтор имеет смысл. Документ в failed не переводится."""

    code: ClassVar[str] = "transient_error"
    retry_after_s: ClassVar[float | None] = None


class PermanentError(ApplicationError):
    """Повтор бессмысленен: фиксируем отказ, событие и DLQ."""

    code: ClassVar[str] = "permanent_error"


class PageLevelError(ApplicationError):
    """Сбой одной страницы. Документ продолжает обработку.

    От PermanentError не наследуется намеренно: иначе обработчик документа
    завалил бы весь документ из-за одной нечитаемой страницы.
    """

    code: ClassVar[str] = "page_level_error"

    def __init__(
        self,
        message: str,
        *,
        page_number: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт ошибку страницы."""
        merged: dict[str, object] = {"page_number": page_number}
        merged.update(context or {})
        super().__init__(message, context=merged)
        self.page_number = page_number


class DatabaseUnavailableError(TransientError):
    """PostgreSQL недоступен."""

    code: ClassVar[str] = "database_unavailable"
    retry_after_s: ClassVar[float | None] = 5.0


class DatabaseTimeoutError(TransientError):
    """Запрос к PostgreSQL прерван по таймауту."""

    code: ClassVar[str] = "database_timeout"
    retry_after_s: ClassVar[float | None] = 10.0


class SerializationConflictError(TransientError):
    """Конфликт сериализации или взаимоблокировка."""

    code: ClassVar[str] = "serialization_conflict"
    retry_after_s: ClassVar[float | None] = 1.0


class ConcurrentProcessingError(TransientError):
    """Документ уже обрабатывается другим воркером с живым лизом."""

    code: ClassVar[str] = "concurrent_processing"
    retry_after_s: ClassVar[float | None] = 30.0


class DocumentNotFoundError(TransientError):
    """Строки документа ещё нет: сообщение обогнало коммит продюсера."""

    code: ClassVar[str] = "document_not_found"
    retry_after_s: ClassVar[float | None] = 5.0


class StorageUnavailableError(TransientError):
    """Хранилище объектов недоступно."""

    code: ClassVar[str] = "storage_unavailable"
    retry_after_s: ClassVar[float | None] = 10.0


class StorageThrottledError(TransientError):
    """Хранилище просит снизить темп."""

    code: ClassVar[str] = "storage_throttled"
    retry_after_s: ClassVar[float | None] = 10.0


class OcrEngineUnavailableError(TransientError):
    """Движок распознавания не поднялся."""

    code: ClassVar[str] = "ocr_engine_unavailable"
    retry_after_s: ClassVar[float | None] = 60.0


class BrokerUnavailableError(TransientError):
    """Брокер сообщений недоступен или не подтвердил публикацию."""

    code: ClassVar[str] = "broker_unavailable"
    retry_after_s: ClassVar[float | None] = 5.0


class TempStorageExhaustedError(TransientError):
    """Кончилось место во временном каталоге."""

    code: ClassVar[str] = "temp_storage_exhausted"
    retry_after_s: ClassVar[float | None] = 30.0


class CpuPoolBrokenError(TransientError):
    """Пул процессов сломан и пересоздаётся."""

    code: ClassVar[str] = "cpu_pool_broken"
    retry_after_s: ClassVar[float | None] = 5.0


class ProcessingDeadlineExceededError(TransientError):
    """Обработка документа не уложилась в отведённое время."""

    code: ClassVar[str] = "processing_deadline_exceeded"
    retry_after_s: ClassVar[float | None] = 60.0


class OcrModelsUnavailableError(PermanentError):
    """Модели распознавания отсутствуют или подменены.

    Отказ при старте, а не ошибка сообщения: ошибка конфигурации, выданная за
    временную, отправляет на бесконечный повтор всю очередь.
    """

    code: ClassVar[str] = "ocr_models_unavailable"


class UnsupportedMediaTypeError(PermanentError):
    """Формат файла не поддерживается."""

    code: ClassVar[str] = "unsupported_media_type"


class DocumentTooLargeError(PermanentError):
    """Размер документа выше допустимого."""

    code: ClassVar[str] = "document_too_large"


class PageLimitExceededError(PermanentError):
    """Число страниц выше допустимого."""

    code: ClassVar[str] = "page_limit_exceeded"


class CorruptedDocumentError(PermanentError):
    """Файл повреждён и не читается."""

    code: ClassVar[str] = "corrupted_document"


class EncryptedDocumentError(PermanentError):
    """Документ зашифрован."""

    code: ClassVar[str] = "encrypted_document"


class SourceObjectNotFoundError(PermanentError):
    """Объекта нет в хранилище."""

    code: ClassVar[str] = "source_object_not_found"


class ChecksumMismatchError(PermanentError):
    """Контрольная сумма скачанного файла не совпала."""

    code: ClassVar[str] = "checksum_mismatch"


class StorageAccessDeniedError(PermanentError):
    """Нет прав на объект хранилища."""

    code: ClassVar[str] = "storage_access_denied"


class InvalidCommandError(PermanentError):
    """Команда не проходит валидацию или указывает на чужой объект."""

    code: ClassVar[str] = "invalid_command"


class DomainInvariantViolationError(PermanentError):
    """Нарушен доменный инвариант."""

    code: ClassVar[str] = "domain_invariant_violation"


class ChunkPersistenceMismatchError(PermanentError):
    """Вставлено не столько чанков, сколько подготовлено."""

    code: ClassVar[str] = "chunk_persistence_mismatch"


class SchemaMisconfiguredError(PermanentError):
    """Схема БД, бакет или очередь настроены не так, как ожидает сервис."""

    code: ClassVar[str] = "schema_misconfigured"


class DuplicateRecordError(PermanentError):
    """Нарушена уникальность.

    Ожидаемые дубли гасит ON CONFLICT, поэтому долетевший сюда конфликт
    означает нарушенный инвариант, а не повторную доставку.
    """

    code: ClassVar[str] = "duplicate_record"

    def __init__(
        self,
        message: str,
        *,
        constraint: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт ошибку с именем нарушенного ограничения."""
        merged: dict[str, object] = {"constraint": constraint}
        merged.update(context or {})
        super().__init__(message, context=merged)
        self.constraint = constraint


class PageRenderError(PageLevelError):
    """Страницу не удалось отрендерить."""

    code: ClassVar[str] = "page_render_failed"


class PageOcrTimeoutError(PageLevelError):
    """Распознавание страницы не уложилось в таймаут."""

    code: ClassVar[str] = "page_ocr_timeout"


class CorruptedPageImageError(PageLevelError):
    """Изображение страницы не читается."""

    code: ClassVar[str] = "corrupted_page_image"


class PageOutOfMemoryError(PageLevelError):
    """Не хватило памяти на обработку страницы."""

    code: ClassVar[str] = "page_out_of_memory"


class OcrEngineError(PageLevelError):
    """Движок распознавания не смог выполнить вывод на этой странице."""

    code: ClassVar[str] = "ocr_engine_failed"


_DOMAIN_TO_APPLICATION: dict[type[DomainError], type[PermanentError]] = {
    UnsupportedDocumentFormat: UnsupportedMediaTypeError,
    DocumentTooLarge: DocumentTooLargeError,
    CorruptedDocument: CorruptedDocumentError,
    EmptyDocument: CorruptedDocumentError,
    EncryptedDocument: EncryptedDocumentError,
    ChecksumMismatch: ChecksumMismatchError,
}


def translate_domain_error(error: DomainError) -> PermanentError:
    """Отображает доменную ошибку в неисправимую прикладную.

    Домен не знает про доставку сообщений, поэтому решение о повторе
    принимается здесь: нарушенный инвариант повтором не лечится.
    """
    target = _DOMAIN_TO_APPLICATION.get(type(error), DomainInvariantViolationError)
    translated = target(error.message, context=error.context)
    translated.__cause__ = error
    return translated
