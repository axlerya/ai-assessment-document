"""Доменные ошибки.

Признака повторяемости здесь нет: домен не знает про доставку сообщений.
Разделение на временные и неисправимые живёт в `application/errors.py`, и
второй источник этого решения разошёлся бы с первым на первой же нестандартной
ошибке.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping


class DomainError(Exception):
    """Базовая доменная ошибка. `code` попадает в failure_code и в заголовки DLQ."""

    code: ClassVar[str] = "domain_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт ошибку с данными, без которых разбор в DLQ невозможен."""
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


class InvariantViolation(DomainError):
    """Нарушен инвариант доменной модели."""

    code: ClassVar[str] = "invariant_violation"


class InvalidValueObject(InvariantViolation):
    """Value object нельзя построить из переданных значений."""

    code: ClassVar[str] = "invalid_value_object"


class InvalidIdentifier(InvalidValueObject):
    """Идентификатор не UUID или является nil-UUID."""

    code: ClassVar[str] = "invalid_identifier"


class InvalidVersion(InvalidValueObject):
    """Версия не в формате семвера или вне допустимых границ."""

    code: ClassVar[str] = "invalid_version"


class InvalidEmbeddingVersion(InvalidVersion):
    """Версия эмбеддингов некорректна."""

    code: ClassVar[str] = "invalid_embedding_version"


class InvalidPromptVersion(InvalidVersion):
    """Версия промпта некорректна."""

    code: ClassVar[str] = "invalid_prompt_version"


class InvalidChunkingVersion(InvalidVersion):
    """Версия чанкования, пришедшая от document-worker, некорректна."""

    code: ClassVar[str] = "invalid_chunking_version"


class InvalidPipelineVersion(InvalidVersion):
    """Версия обработки, пришедшая от document-worker, некорректна."""

    code: ClassVar[str] = "invalid_pipeline_version"


class InvalidVector(InvalidValueObject):
    """Вектор не той ширины, пуст или содержит значения, не являющиеся числами."""

    code: ClassVar[str] = "invalid_vector"


class InvalidEmbeddingPolicy(InvalidValueObject):
    """Параметры эмбеддингов непригодны или разошлись со своей версией."""

    code: ClassVar[str] = "invalid_embedding_policy"


class InvalidScore(InvalidValueObject):
    """Оценка не число либо вне своего диапазона."""

    code: ClassVar[str] = "invalid_score"


class InvalidTextSpan(InvalidValueObject):
    """Диапазон цитаты перевёрнут, пуст или выходит за пределы текста."""

    code: ClassVar[str] = "invalid_text_span"


class InvalidStatusTransition(InvariantViolation):
    """Переход статуса запрещён таблицей переходов."""

    code: ClassVar[str] = "invalid_status_transition"


class FabricatedQuote(InvariantViolation):
    """Цитата не является срезом текста своего чанка."""

    code: ClassVar[str] = "fabricated_quote"
