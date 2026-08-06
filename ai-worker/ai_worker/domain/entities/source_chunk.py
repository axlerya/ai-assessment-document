"""Чанк документа в том виде, в каком его отдаёт document-worker.

Три части, потому что у них разные потребители. `ChunkRef` — координаты, по
которым восстанавливается цитата, и они нужны и эмбеддингу, и ссылке в
черновике. `ChunkQuality` — признаки надёжности, посчитанные при обработке; они
решают, можно ли опереть на этот фрагмент утверждение. `SourceChunk` — то и
другое вместе с текстом.

Признаки надёжности копируются, а не берутся джойном: retrieval обязан отдавать
их вместе с попаданием, а джойн к чужим таблицам на каждый поиск сделал бы
границу между сервисами фиктивной.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_worker.domain.errors import InvariantViolation
from ai_worker.domain.value_objects.enums import ExtractionMethod
from ai_worker.domain.value_objects.hashing import ContentHash

if TYPE_CHECKING:
    from ai_worker.domain.value_objects.identifiers import (
        ChunkId,
        DocumentId,
        PageId,
    )
    from ai_worker.domain.value_objects.scores import Ratio
    from ai_worker.domain.value_objects.versioning import ChunkingVersion

MIN_PAGE_NUMBER = 1


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """Координаты чанка: по ним проверяется цитата."""

    chunk_id: ChunkId
    document_id: DocumentId
    page_id: PageId
    page_number: int

    def __post_init__(self) -> None:
        """Требует осмысленный номер страницы.

        Raises:
            InvariantViolation: Нумерация страниц начинается с единицы, и ноль
                сдвинул бы адресацию всех цитат документа.
        """
        if self.page_number < MIN_PAGE_NUMBER:
            raise InvariantViolation(
                f"номер страницы меньше {MIN_PAGE_NUMBER}",
                context={"page_number": self.page_number},
            )


@dataclass(frozen=True, slots=True)
class ChunkQuality:
    """Насколько можно доверять тексту чанка."""

    extraction_method: ExtractionMethod
    avg_confidence: Ratio | None
    illegible_span_count: int

    def __post_init__(self) -> None:
        """Сверяет уверенность со способом извлечения.

        Raises:
            InvariantViolation: У текстового слоя уверенность есть, у
                распознавания её нет, либо счётчик неразборчивых мест
                отрицателен.
        """
        text_layer = self.extraction_method is ExtractionMethod.TEXT_LAYER
        if text_layer and self.avg_confidence is not None:
            # Единица от текстового слоя и единица от распознавания — величины
            # разной природы: смешав их, мы навсегда портим любой агрегат.
            raise InvariantViolation(
                "у текстового слоя уверенности не существует",
                context={"method": self.extraction_method.value},
            )
        if not text_layer and self.avg_confidence is None:
            raise InvariantViolation(
                f"способ {self.extraction_method.value} обязан нести уверенность",
                context={"method": self.extraction_method.value},
            )
        if self.illegible_span_count < 0:
            raise InvariantViolation(
                "счётчик неразборчивых фрагментов отрицателен",
                context={"illegible_span_count": self.illegible_span_count},
            )

    @property
    def has_illegible(self) -> bool:
        """Попадают ли в чанк неразборчивые фрагменты."""
        return self.illegible_span_count > 0


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """Фрагмент текста документа вместе с координатами и признаками качества."""

    ref: ChunkRef
    quality: ChunkQuality
    text: str
    token_count: int
    chunking_version: ChunkingVersion
    heading_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Требует непустой текст и осмысленное число токенов.

        Raises:
            InvariantViolation: Текст пробельный либо число токенов не
                положительно — такой чанк нечего ни индексировать, ни цитировать.
        """
        if not self.text.strip():
            raise InvariantViolation(
                "пустой чанк не попадает в индекс",
                context={"chunk_id": str(self.ref.chunk_id)},
            )
        if self.token_count < 1:
            raise InvariantViolation(
                "число токенов чанка не положительно",
                context={"token_count": self.token_count},
            )

    @property
    def content_hash(self) -> ContentHash:
        """Контрольная сумма текста: по ней пропускается повторный эмбеддинг."""
        return ContentHash.sha256_of(self.text)
