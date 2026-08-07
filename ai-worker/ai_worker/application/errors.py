"""Прикладные ошибки.

Именно эта классификация решает судьбу сообщения: подтвердить, повторить или
отправить в разбор. Домен её не знает — он про доставку не осведомлён, — а
инфраструктура в неё переводит отказы своих библиотек.

Три класса, а не два. `ChunkLevelError` не наследуется от `PermanentError`
намеренно: сбой на одном чанке не обязан валить весь документ, он даёт
частичный индекс. Смешав его с неисправимой ошибкой, мы теряли бы весь
документ из-за одного нечитаемого фрагмента.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping


class ApplicationError(Exception):
    """Базовая прикладная ошибка. `code` уезжает в событие и заголовки DLQ."""

    code: ClassVar[str] = "application_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт ошибку с данными, без которых разбор невозможен."""
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
    """Отказ, который пройдёт сам: сообщение уходит на повтор."""

    code: ClassVar[str] = "transient_error"
    retry_after_s: ClassVar[float] = 5.0


class PermanentError(ApplicationError):
    """Отказ, который повтор не исправит: сообщение уходит в разбор."""

    code: ClassVar[str] = "permanent_error"


class ChunkLevelError(ApplicationError):
    """Сбой на одном чанке.

    Не наследуется ни от временных, ни от неисправимых: документ продолжает
    индексироваться, а чанк попадает в счётчик пропущенных.
    """

    code: ClassVar[str] = "chunk_level_error"


class StorageUnavailable(TransientError):
    """База недоступна, разорвала соединение или отказала по нагрузке."""

    code: ClassVar[str] = "storage_unavailable"


class StorageConflict(TransientError):
    """Взаимоблокировка или сбой сериализации: повтор транзакции уместен."""

    code: ClassVar[str] = "storage_conflict"

    retry_after_s: ClassVar[float] = 0.5


class InvariantRejectedByStorage(PermanentError):
    """Строка противоречит схеме: нарушен CHECK, внешний ключ или NOT NULL."""

    code: ClassVar[str] = "invariant_rejected_by_storage"


class EmbeddingModelMissing(PermanentError):
    """Файлов модели нет или они подменены.

    Это ошибка конфигурации, а не сбой: повтор сообщения её не исправит, а
    выданная за временную она отправила бы на бесконечный повтор всю очередь.
    """

    code: ClassVar[str] = "embedding_model_missing"


class EmbeddingBackendUnavailable(TransientError):
    """Прогон модели не состоялся: таймаут или упавший рабочий процесс."""

    code: ClassVar[str] = "embedding_backend_unavailable"


class DuplicateRecord(PermanentError):
    """Строка с таким ключом уже есть.

    Это ожидаемый исход повторной доставки, а не поломка: вызывающий решает,
    что работа уже сделана, и не заводит вторую.
    """

    code: ClassVar[str] = "duplicate_record"
