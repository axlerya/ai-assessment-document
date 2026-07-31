"""Общее для тестов чанкера: счётчик токенов без ввода-вывода и страницы.

Счётчик подменяется, доменная логика — нет: tiktoken в unit-слое не участвует,
иначе тест границ чанков зависел бы от чужого BPE-словаря.

Модуль обычный, а не conftest: помощники импортируются по имени, а фикстур
pytest здесь нет.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from document_worker.domain.chunking.policy import (
    CHUNKING_VERSION,
    DEFAULT_CHUNKING_POLICY,
    ChunkingPolicy,
)
from document_worker.domain.entities.document_page import DocumentPage
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    IllegibleReason,
    PageFailureReason,
    PageStatus,
)
from document_worker.domain.value_objects.identifiers import DocumentId, PageId
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.recognized_text import RecognizedText
from document_worker.domain.value_objects.storage import ObjectRef
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan
from document_worker.domain.value_objects.versioning import PipelineVersion

if TYPE_CHECKING:
    from document_worker.domain.value_objects.versioning import ChunkingVersion

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
PIPELINE_VERSION = PipelineVersion(1, 0, 0)
DOCUMENT_ID = DocumentId(uuid.UUID("3f2a1c0e-0b1d-4a2e-8c3f-5d6e7a8b9c01"))
CHARS_PER_TOKEN = 4


class FakeTokenCounter:
    """Токен на каждые четыре символа: детерминированно и монотонно по длине."""

    def __init__(self) -> None:
        """Заводит счётчик вызовов — по нему проверяется двоичный поиск."""
        self.calls = 0

    def count(self, text: str) -> int:
        """Число токенов в тексте."""
        self.calls += 1
        return math.ceil(len(text) / CHARS_PER_TOKEN)


def default_policy(  # noqa: PLR0913 — политика описывается всеми своими полями
    *,
    version: ChunkingVersion = CHUNKING_VERSION,
    encoding: str = DEFAULT_CHUNKING_POLICY.encoding,
    target_tokens: int = DEFAULT_CHUNKING_POLICY.target_tokens,
    max_tokens: int = DEFAULT_CHUNKING_POLICY.max_tokens,
    min_tokens: int = DEFAULT_CHUNKING_POLICY.min_tokens,
    overlap_tokens: int = DEFAULT_CHUNKING_POLICY.overlap_tokens,
) -> ChunkingPolicy:
    """Политика с промышленными значениями, если тест не задал своих."""
    return ChunkingPolicy(
        version=version,
        encoding=encoding,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
    )


def text_layer_page(content: str, *, number: int = 1) -> DocumentPage:
    """Страница текстового слоя: уверенности нет, диапазонов нет."""
    return DocumentPage.from_text_layer(
        page_id=PageId(uuid.uuid4()),
        document_id=DOCUMENT_ID,
        number=PageNumber(number),
        pipeline_version=PIPELINE_VERSION,
        content=content,
        now=NOW,
    )


def failed_page(*, number: int = 1) -> DocumentPage:
    """Страница, которую не удалось прочитать: способа нет, текста нет."""
    return DocumentPage.failed(
        page_id=PageId(uuid.uuid4()),
        document_id=DOCUMENT_ID,
        number=PageNumber(number),
        pipeline_version=PIPELINE_VERSION,
        reason=PageFailureReason.RENDER_FAILED,
        message="страница не отрендерилась",
        now=NOW,
        recoverable=True,
    )


def ocr_page(  # noqa: PLR0913 — страница описывается всеми этими значениями
    content: str,
    *,
    number: int = 1,
    confidence: float = 0.9,
    illegible: tuple[tuple[int, int, float], ...] = (),
    status: PageStatus | None = None,
    method: ExtractionMethod = ExtractionMethod.OCR,
) -> DocumentPage:
    """Распознанная страница с заданными неразборчивыми диапазонами."""
    spans = tuple(
        IllegibleSpan(
            span=TextSpan(start, end),
            confidence=OcrConfidence(span_confidence),
            reason=IllegibleReason.LOW_OCR_CONFIDENCE,
            raw_text=content[start:end],
        )
        for start, end, span_confidence in illegible
    )
    if status is None:
        status = PageStatus.PARTIALLY_ILLEGIBLE if spans else PageStatus.EXTRACTED
    return DocumentPage(
        id=PageId(uuid.uuid4()),
        document_id=DOCUMENT_ID,
        number=PageNumber(number),
        pipeline_version=PIPELINE_VERSION,
        status=status,
        text=RecognizedText(
            content=content,
            method=method,
            confidence=OcrConfidence(confidence),
            illegible_spans=spans,
        ),
        created_at=NOW,
        image_ref=ObjectRef(bucket="renders", key=f"{uuid.uuid4().hex}/{number}.png"),
        render_dpi=300,
    )
