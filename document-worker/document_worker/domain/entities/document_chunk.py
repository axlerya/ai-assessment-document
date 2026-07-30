"""Чанк документа.

Нумерация внутри страницы, а не сквозная по документу: сквозной индекс при
частичном повторе перестраивался бы и молча конфликтовал с уже вставленными
чанками. Порядок по документу даёт пара (номер страницы, порядковый номер).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

from document_worker.domain.constants import (
    MAX_CHUNK_OVERLAP_CHARS,
    MAX_CHUNK_TOKENS,
)
from document_worker.domain.errors import ChunkSpanMismatch, InvariantViolation
from document_worker.domain.value_objects.enums import ExtractionMethod
from document_worker.domain.value_objects.storage import Checksum

if TYPE_CHECKING:
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.value_objects.confidence import OcrConfidence
    from document_worker.domain.value_objects.identifiers import (
        ChunkId,
        DocumentId,
        PageId,
    )
    from document_worker.domain.value_objects.paging import PageNumber
    from document_worker.domain.value_objects.text import TextSpan
    from document_worker.domain.value_objects.versioning import ChunkingVersion

_MIN_TOKEN_COUNT = 1


@dataclass(frozen=True, slots=True, eq=False)
class DocumentChunk:
    """Фрагмент текста страницы, сохраняющий связь с источником."""

    id: ChunkId
    document_id: DocumentId
    page_id: PageId
    page_number: PageNumber
    ordinal: int
    content: str
    span: TextSpan
    method: ExtractionMethod
    avg_confidence: OcrConfidence | None
    illegible_span_count: int
    chunking_version: ChunkingVersion
    checksum: Checksum
    token_count: int
    heading_path: tuple[str, ...] = ()
    overlap_prefix_chars: int = 0

    def __post_init__(self) -> None:
        """Сверяет текст со своим диапазоном, контрольной суммой и лимитами."""
        if not self.content.strip():
            raise InvariantViolation(
                "пустой чанк не попадает в индекс",
                context={"page": int(self.page_number), "ordinal": self.ordinal},
            )
        if len(self.content) != self.span.length:
            raise ChunkSpanMismatch(
                "текст чанка перестал быть точным срезом текста страницы",
                context={"content": len(self.content), "span": self.span.length},
            )
        if self.checksum != Checksum.sha256_of(self.content.encode("utf-8")):
            raise InvariantViolation(
                "контрольная сумма не соответствует тексту чанка",
                context={"checksum": self.checksum.value},
            )
        self._validate_numbers()
        self._validate_method()

    def _validate_numbers(self) -> None:
        if self.ordinal < 0:
            raise InvariantViolation(
                "порядковый номер чанка отрицателен",
                context={"ordinal": self.ordinal},
            )
        if not _MIN_TOKEN_COUNT <= self.token_count <= MAX_CHUNK_TOKENS:
            raise InvariantViolation(
                f"число токенов вне {_MIN_TOKEN_COUNT}..{MAX_CHUNK_TOKENS}",
                context={"token_count": self.token_count},
            )
        if self.illegible_span_count < 0:
            raise InvariantViolation(
                "счётчик неразборчивых фрагментов отрицателен",
                context={"illegible_span_count": self.illegible_span_count},
            )
        if not 0 <= self.overlap_prefix_chars <= MAX_CHUNK_OVERLAP_CHARS:
            raise InvariantViolation(
                f"перекрытие вне 0..{MAX_CHUNK_OVERLAP_CHARS}",
                context={"overlap_prefix_chars": self.overlap_prefix_chars},
            )
        if self.overlap_prefix_chars >= len(self.content):
            raise InvariantViolation(
                "перекрытие не короче самого чанка",
                context={
                    "overlap_prefix_chars": self.overlap_prefix_chars,
                    "content": len(self.content),
                },
            )

    def _validate_method(self) -> None:
        if self.method is ExtractionMethod.NONE:
            raise InvariantViolation(
                "чанк не строится по странице без текста",
                context={"page": int(self.page_number)},
            )
        text_layer = self.method is ExtractionMethod.TEXT_LAYER
        if text_layer and self.avg_confidence is not None:
            raise InvariantViolation(
                "у текстового слоя уверенности не существует",
                context={"method": self.method.value},
            )
        if not text_layer and self.avg_confidence is None:
            raise InvariantViolation(
                f"способ {self.method.value} обязан нести уверенность",
                context={"method": self.method.value},
            )

    @classmethod
    def from_page_slice(  # noqa: PLR0913 — чанк описывается всеми этими значениями
        cls,
        *,
        chunk_id: ChunkId,
        page: DocumentPage,
        ordinal: int,
        span: TextSpan,
        avg_confidence: OcrConfidence | None,
        token_count: int,
        chunking_version: ChunkingVersion,
        heading_path: tuple[str, ...] = (),
        overlap_prefix_chars: int = 0,
    ) -> Self:
        """Строит чанк срезом текста страницы.

        Текст берётся срезом, поэтому расхождение текста и смещений
        непредставимо по построению.
        """
        content = span.slice_of(page.text.content)
        return cls(
            id=chunk_id,
            document_id=page.document_id,
            page_id=page.id,
            page_number=page.number,
            ordinal=ordinal,
            content=content,
            span=span,
            method=page.method,
            avg_confidence=avg_confidence,
            illegible_span_count=sum(
                1 for illegible in page.illegible_spans if illegible.span.overlaps(span)
            ),
            chunking_version=chunking_version,
            checksum=Checksum.sha256_of(content.encode("utf-8")),
            token_count=token_count,
            heading_path=heading_path,
            overlap_prefix_chars=overlap_prefix_chars,
        )

    @property
    def has_illegible(self) -> bool:
        """Попадают ли в чанк неразборчивые фрагменты."""
        return self.illegible_span_count > 0

    @property
    def char_count(self) -> int:
        """Длина текста чанка."""
        return len(self.content)

    def citation(self) -> str:
        """Якорь для проверки источника."""
        return f"p.{int(self.page_number)} [{self.span.start}:{self.span.end}]"

    def follows(self, other: DocumentChunk) -> bool:
        """Идёт ли этот чанк сразу за другим на той же странице."""
        return self.page_id == other.page_id and self.ordinal == other.ordinal + 1

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentChunk):
            return NotImplemented
        return self.id == other.id

    @override
    def __hash__(self) -> int:
        return hash(self.id)
