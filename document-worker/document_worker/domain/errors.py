"""Доменные ошибки.

Признака повторяемости здесь нет: домен не знает про доставку сообщений.
Разделение на временные и неисправимые живёт в `application/errors.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class DomainError(Exception):
    """Базовая доменная ошибка. `code` попадает в failure_code и заголовки DLQ."""

    code: ClassVar[str] = "domain_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт ошибку с данными, без которых сообщение в DLQ бесполезно."""
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


class InvalidCorrelationId(InvalidValueObject):
    """Correlation id не проходит по длине или набору символов."""

    code: ClassVar[str] = "invalid_correlation_id"


class InvalidPageNumber(InvalidValueObject):
    """Номер страницы вне допустимого диапазона."""

    code: ClassVar[str] = "invalid_page_number"


class InvalidObjectRef(InvalidValueObject):
    """Ссылка на объект хранилища некорректна или небезопасна."""

    code: ClassVar[str] = "invalid_object_ref"


class InvalidChecksum(InvalidValueObject):
    """Контрольная сумма не соответствует своему алгоритму."""

    code: ClassVar[str] = "invalid_checksum"


class InvalidMimeType(InvalidValueObject):
    """Строка не является корректным MIME-типом."""

    code: ClassVar[str] = "invalid_mime_type"


class InvalidFileSize(InvalidValueObject):
    """Размер файла отрицателен или превышает предел."""

    code: ClassVar[str] = "invalid_file_size"


class InvalidConfidence(InvalidValueObject):
    """Уверенность распознавания вне диапазона 0..1 либо не число."""

    code: ClassVar[str] = "invalid_confidence"


class InvalidTextSpan(InvalidValueObject):
    """Текстовый диапазон перевёрнут или выходит за пределы текста."""

    code: ClassVar[str] = "invalid_text_span"


class InvalidIllegibleSpan(InvalidValueObject):
    """Диапазон неразборчивости противоречит своей причине."""

    code: ClassVar[str] = "invalid_illegible_span"


class InvalidBoundingBox(InvalidValueObject):
    """Прямоугольник вырожден или не нормализован."""

    code: ClassVar[str] = "invalid_bounding_box"


class InvalidRecognizedWord(InvalidValueObject):
    """Распознанное слово противоречит своему диапазону."""

    code: ClassVar[str] = "invalid_recognized_word"


class InvalidPipelineVersion(InvalidValueObject):
    """Версия пайплайна не в формате семвера или вне границ."""

    code: ClassVar[str] = "invalid_pipeline_version"


class InvalidChunkingVersion(InvalidValueObject):
    """Версия чанкования не в формате семвера или вне границ."""

    code: ClassVar[str] = "invalid_chunking_version"


class InvalidChunkingPolicy(InvariantViolation):
    """Бюджет токенов противоречив или не соответствует объявленной версии."""

    code: ClassVar[str] = "invalid_chunking_policy"


class InvalidStatusTransition(InvariantViolation):
    """Переход статуса запрещён таблицей переходов."""

    code: ClassVar[str] = "invalid_status_transition"


class IncompletePageSet(InvariantViolation):
    """Набор страниц документа неполон или содержит пропуски."""

    code: ClassVar[str] = "incomplete_page_set"


class ChunkSpanMismatch(InvariantViolation):
    """Текст чанка не совпадает со срезом текста своей страницы."""

    code: ClassVar[str] = "chunk_span_mismatch"


class OffsetMapBroken(InvariantViolation):
    """Карта смещений не проецирует диапазон в исходный текст."""

    code: ClassVar[str] = "offset_map_broken"


class FabricatedTextDetected(InvariantViolation):
    """Сохранённый фрагмент не совпадает со срезом канонического текста."""

    code: ClassVar[str] = "fabricated_text"


class DocumentRejected(DomainError):
    """Документ не может быть обработан по свойствам самого файла."""

    code: ClassVar[str] = "document_rejected"


class UnsupportedDocumentFormat(DocumentRejected):
    """Формат файла не входит в список поддерживаемых."""

    code: ClassVar[str] = "unsupported_format"

    def __init__(self, mime_type: str, *, supported: Iterable[str]) -> None:
        """Создаёт ошибку неподдерживаемого формата."""
        supported_types = tuple(supported)
        super().__init__(
            f"формат {mime_type} не поддерживается, ожидается один из "
            f"{', '.join(supported_types)}",
            context={"mime_type": mime_type, "supported": supported_types},
        )


class CorruptedDocument(DocumentRejected):
    """Файл повреждён и не читается разбором PDF."""

    code: ClassVar[str] = "corrupted_document"


class DocumentTooLarge(DocumentRejected):
    """Размер документа превышает допустимый предел."""

    code: ClassVar[str] = "document_too_large"

    def __init__(self, *, actual_bytes: int, limit_bytes: int) -> None:
        """Создаёт ошибку превышения размера."""
        super().__init__(
            f"размер документа {actual_bytes} байт превышает предел {limit_bytes} байт",
            context={"actual_bytes": actual_bytes, "limit_bytes": limit_bytes},
        )


class EmptyDocument(DocumentRejected):
    """Документ пуст: нулевой размер или ноль страниц."""

    code: ClassVar[str] = "empty_document"


class EncryptedDocument(DocumentRejected):
    """Документ зашифрован и не открывается без пароля."""

    code: ClassVar[str] = "encrypted_document"


class ChecksumMismatch(DocumentRejected):
    """Контрольная сумма скачанного файла не совпала с заявленной."""

    code: ClassVar[str] = "checksum_mismatch"

    def __init__(self, *, expected: str, actual: str) -> None:
        """Создаёт ошибку расхождения контрольных сумм."""
        super().__init__(
            f"контрольная сумма не совпала: ожидалась {expected}, получена {actual}",
            context={"expected": expected, "actual": actual},
        )
