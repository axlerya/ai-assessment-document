"""Словари домена.

Значения хранятся в базе как есть и уезжают в сообщения, поэтому перечисления
строковые: обычный `Enum` требовал бы `.value` в каждой точке сериализации, и
первая же забытая точка положила бы в колонку `IndexStatus.INDEXED`.

Наборы значений совпадают с CHECK-ограничениями схемы. Расхождение здесь
означает строку, которую база откажется принять уже после обработки.
"""

from __future__ import annotations

from enum import StrEnum


class IndexStatus(StrEnum):
    """Состояние индексации документа в конкретной версии эмбеддингов."""

    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"

    @classmethod
    def terminal(cls) -> frozenset[IndexStatus]:
        """Состояния, из которых обработка уже не продолжается."""
        return frozenset({cls.INDEXED, cls.FAILED})


class SourceStatus(StrEnum):
    """Исход обработки документа, при котором его есть смысл индексировать.

    Документ со статусом `failed` сюда не попадает: пригодного текста у него
    нет, и индексировать нечего.
    """

    PROCESSED = "processed"
    PARTIALLY_PROCESSED = "partially_processed"


class DraftStatus(StrEnum):
    """Исход подготовки черновика.

    `insufficient_evidence` — не отказ, а результат: подтверждений не нашлось,
    и сервис говорит об этом прямо, вместо того чтобы додумать факты.
    """

    GENERATED = "generated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class DraftType(StrEnum):
    """Вид черновика."""

    CASE_FACT_SUMMARY = "case_fact_summary"


class ClaimSection(StrEnum):
    """Разделы сводки фактов по делу в порядке их следования в черновике.

    Последний раздел обязателен всегда: это единственное место, где сервис
    сообщает о недостатке данных.
    """

    PARTIES = "parties"
    DOCUMENTS = "documents"
    DATES = "dates"
    AMOUNTS = "amounts"
    OPEN_QUESTIONS = "open_questions"


class RejectCode(StrEnum):
    """Почему утверждение не попало в тело черновика.

    Отклонённые утверждения хранятся: без кода причины разбор превращается в
    чтение логов, а вместе с ним теряется и сам факт попытки додумать.
    """

    NO_CITATION = "no_citation"
    CHUNK_NOT_IN_CONTEXT = "chunk_not_in_context"
    QUOTE_NOT_FOUND = "quote_not_found"
    UNRELIABLE_EVIDENCE_ONLY = "unreliable_evidence_only"


class ExtractionMethod(StrEnum):
    """Как был получен текст чанка. Копируется из document-worker.

    Значения `none` здесь нет: у страницы, с которой ничего не извлечено, не
    бывает чанков, а значит нечего и индексировать.
    """

    TEXT_LAYER = "text_layer"
    OCR = "ocr"
    HYBRID = "hybrid"

    @property
    def is_recognized(self) -> bool:
        """Участвовало ли распознавание — то есть есть ли у чанка уверенность."""
        return self is not ExtractionMethod.TEXT_LAYER
