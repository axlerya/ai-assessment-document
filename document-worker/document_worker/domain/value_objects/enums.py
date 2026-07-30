"""Словари домена: статусы, способы извлечения, причины неразборчивости.

Строковые значения — контракт со схемой БД, там это varchar + CHECK.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final

from document_worker.domain.errors import InvalidStatusTransition

if TYPE_CHECKING:
    from collections.abc import Mapping


class DocumentStatus(StrEnum):
    """Статус обработки документа."""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    PARTIALLY_PROCESSED = "partially_processed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Дошёл ли документ до окончательного состояния."""
        return self in _TERMINAL_DOCUMENT_STATUSES

    @property
    def is_successful(self) -> bool:
        """Получен ли пригодный результат; partially_processed тоже успех."""
        return self in _SUCCESSFUL_DOCUMENT_STATUSES

    def can_transition_to(self, target: DocumentStatus) -> bool:
        """Разрешён ли переход в целевой статус."""
        return target in _DOCUMENT_TRANSITIONS[self]

    def ensure_can_transition_to(self, target: DocumentStatus) -> None:
        """Проверяет переход.

        Raises:
            InvalidStatusTransition: Переход запрещён таблицей переходов.
        """
        if not self.can_transition_to(target):
            raise InvalidStatusTransition(
                f"переход {self.value} -> {target.value} запрещён",
                context={"from": self.value, "to": target.value},
            )


class PageStatus(StrEnum):
    """Итог чтения одной страницы."""

    EXTRACTED = "extracted"
    PARTIALLY_ILLEGIBLE = "partially_illegible"
    ILLEGIBLE = "illegible"
    FAILED = "failed"

    @property
    def is_usable(self) -> bool:
        """Годится ли текст страницы для чанкования и retrieval."""
        return self in _USABLE_PAGE_STATUSES

    @property
    def is_fully_read(self) -> bool:
        """Прочитана ли страница целиком, без неразборчивых диапазонов."""
        return self is PageStatus.EXTRACTED


class PageFailureReason(StrEnum):
    """Техническая причина, по которой страница не прочитана."""

    RENDER_FAILED = "render_failed"
    OCR_FAILED = "ocr_failed"
    TEXT_EXTRACTION_FAILED = "text_extraction_failed"
    PAGE_CORRUPTED = "page_corrupted"
    TIMEOUT = "timeout"


class ExtractionMethod(StrEnum):
    """Способ, которым получен текст страницы. Синонима `mixed` нет."""

    TEXT_LAYER = "text_layer"
    OCR = "ocr"
    HYBRID = "hybrid"
    NONE = "none"

    @property
    def is_ocr_based(self) -> bool:
        """Участвовало ли распознавание, то есть есть ли у страницы confidence."""
        return self in _OCR_BASED_METHODS

    @property
    def yields_text(self) -> bool:
        """Даёт ли способ текст вообще."""
        return self is not ExtractionMethod.NONE


class CompletionOutcome(StrEnum):
    """Итог терминального перехода. Дубль это не ошибка, а штатный повтор."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"


class JobStatus(StrEnum):
    """Статус прогона обработки документа."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IllegibleReason(StrEnum):
    """Причина, по которой фрагмент документа помечен неразборчивым.

    Только свойства самого документа. Сбой обработки страницу не размечает,
    а отменяет целиком — его причина живёт в PageFailureReason.
    """

    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    NO_TEXT_RECOGNIZED = "no_text_recognized"
    IMAGE_TOO_NOISY = "image_too_noisy"
    HANDWRITING = "handwriting"
    GLYPH_MAPPING_FAILED = "glyph_mapping_failed"


class ProcessingStage(StrEnum):
    """Стадия обработки, попадает в failure_stage."""

    DOWNLOAD = "download"
    VALIDATION = "validation"
    INSPECTION = "inspection"
    TEXT_EXTRACTION = "text_extraction"
    RENDERING = "rendering"
    OCR = "ocr"
    NORMALIZATION = "normalization"
    CHUNKING = "chunking"
    PERSISTENCE = "persistence"
    PUBLISHING = "publishing"


# Константы вынесены из тел классов: присваивание внутри StrEnum сделало бы их
# members словаря.

_TERMINAL_DOCUMENT_STATUSES: Final[frozenset[DocumentStatus]] = frozenset(
    {
        DocumentStatus.PROCESSED,
        DocumentStatus.PARTIALLY_PROCESSED,
        DocumentStatus.FAILED,
    }
)

_SUCCESSFUL_DOCUMENT_STATUSES: Final[frozenset[DocumentStatus]] = frozenset(
    {DocumentStatus.PROCESSED, DocumentStatus.PARTIALLY_PROCESSED}
)

_USABLE_PAGE_STATUSES: Final[frozenset[PageStatus]] = frozenset(
    {PageStatus.EXTRACTED, PageStatus.PARTIALLY_ILLEGIBLE}
)

_OCR_BASED_METHODS: Final[frozenset[ExtractionMethod]] = frozenset(
    {ExtractionMethod.OCR, ExtractionMethod.HYBRID}
)

# Возврат из терминального статуса в processing разрешён только повторной
# обработкой более новой версией пайплайна; версию проверяет сущность Document.
_DOCUMENT_TRANSITIONS: Final[Mapping[DocumentStatus, frozenset[DocumentStatus]]] = {
    DocumentStatus.PENDING: frozenset(
        {DocumentStatus.PROCESSING, DocumentStatus.FAILED}
    ),
    DocumentStatus.PROCESSING: frozenset(
        {
            DocumentStatus.PROCESSED,
            DocumentStatus.PARTIALLY_PROCESSED,
            DocumentStatus.FAILED,
        }
    ),
    DocumentStatus.PROCESSED: frozenset({DocumentStatus.PROCESSING}),
    DocumentStatus.PARTIALLY_PROCESSED: frozenset({DocumentStatus.PROCESSING}),
    DocumentStatus.FAILED: frozenset({DocumentStatus.PROCESSING}),
}
