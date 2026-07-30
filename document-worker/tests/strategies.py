"""Стратегии hypothesis для доменных сущностей.

Сущности собираются сразу согласованными: статус выбирается первым, остальные
поля достраиваются под него. Генерация «поле за полем» отсеивалась бы
инвариантами сущности почти целиком.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from hypothesis import strategies as st

from document_worker.domain.constants import (
    MAX_CHUNK_OVERLAP_CHARS,
    MAX_CHUNK_TOKENS,
    MAX_PAGES,
    MAX_RENDER_DPI,
    MIN_RENDER_DPI,
)
from document_worker.domain.entities.document import Document
from document_worker.domain.entities.document_chunk import DocumentChunk
from document_worker.domain.entities.document_page import DocumentPage
from document_worker.domain.entities.processing_job import ProcessingJob
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    DocumentStatus,
    ExtractionMethod,
    IllegibleReason,
    JobStatus,
    PageFailureReason,
    PageStatus,
    ProcessingStage,
)
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.domain.value_objects.identifiers import (
    ChunkId,
    CorrelationId,
    DocumentId,
    EventId,
    JobId,
    PageId,
)
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.quality import PageFailure
from document_worker.domain.value_objects.recognized_text import RecognizedText
from document_worker.domain.value_objects.storage import (
    Checksum,
    ChecksumAlgorithm,
    FileSize,
    MimeType,
    ObjectRef,
    SourceFile,
)
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan
from document_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    PipelineVersion,
)

if TYPE_CHECKING:
    import uuid

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

# Маркеры неразборчивости в тексте страницы запрещены, поэтому в алфавите нет
# квадратных скобок; пробельные символы исключены — они не переживают ни одной
# нормализации и к персистентности отношения не имеют.
TEXT_ALPHABET = st.characters(
    min_codepoint=0x20,
    max_codepoint=0x4FF,
    blacklist_characters="[]\x7f",
    blacklist_categories=("Cc", "Cs", "Zs", "Zl", "Zp"),
)


def timestamps() -> st.SearchStrategy[datetime]:
    """Момент времени в UTC с точностью до микросекунд."""
    return st.integers(min_value=0, max_value=10**9).map(
        lambda offset: EPOCH + timedelta(microseconds=offset)
    )


def uuids() -> st.SearchStrategy[uuid.UUID]:
    """UUID, кроме nil."""
    return st.uuids(version=4)


def correlation_ids() -> st.SearchStrategy[CorrelationId]:
    """Сквозной идентификатор запроса."""
    return uuids().map(lambda value: CorrelationId(str(value)))


def pipeline_versions() -> st.SearchStrategy[PipelineVersion]:
    """Версия пайплайна."""
    return st.builds(
        PipelineVersion,
        major=st.integers(min_value=1, max_value=999),
        minor=st.integers(min_value=0, max_value=999),
        patch=st.integers(min_value=0, max_value=999),
    )


def chunking_versions() -> st.SearchStrategy[ChunkingVersion]:
    """Версия чанкования."""
    return st.builds(
        ChunkingVersion,
        major=st.integers(min_value=1, max_value=999),
        minor=st.integers(min_value=0, max_value=999),
        patch=st.integers(min_value=0, max_value=999),
    )


def confidences() -> st.SearchStrategy[OcrConfidence]:
    """Уверенность распознавания."""
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False).map(OcrConfidence)


def object_refs() -> st.SearchStrategy[ObjectRef]:
    """Ссылка на объект хранилища."""
    return st.builds(
        ObjectRef,
        bucket=st.text("abcdefghijklmnopqrstuvwxyz0123456789-", min_size=3, max_size=20)
        .filter(lambda value: value[0].isalnum() and value[-1].isalnum())
        .filter(lambda value: ".." not in value),
        key=st.text("abcdefghijklmnopqrstuvwxyz0123456789/._-", min_size=1, max_size=60)
        .filter(lambda value: value[0].isalnum())
        .filter(lambda value: ".." not in value.split("/")),
    )


def checksums() -> st.SearchStrategy[Checksum]:
    """Контрольная сумма sha256."""
    return st.binary(min_size=0, max_size=64).map(Checksum.sha256_of)


def source_files() -> st.SearchStrategy[SourceFile]:
    """Исходный файл документа."""
    return st.builds(
        SourceFile,
        ref=object_refs(),
        mime_type=st.just(MimeType(MimeType.PDF)),
        size=st.integers(min_value=1, max_value=104_857_600).map(FileSize),
        checksum=st.none() | checksums(),
    )


@st.composite
def documents(draw: st.DrawFn) -> Document:
    """Документ, согласованный со своим статусом."""
    status = draw(st.sampled_from(DocumentStatus))
    created_at = draw(timestamps())
    started_at = created_at + timedelta(seconds=draw(st.integers(0, 3600)))
    finished_at = started_at + timedelta(seconds=draw(st.integers(0, 3600)))
    terminal = status.is_terminal
    successful = status.is_successful
    source = draw(source_files())
    return Document(
        id=DocumentId(draw(uuids())),
        source=source
        if not successful
        else SourceFile(
            ref=source.ref,
            mime_type=source.mime_type,
            size=source.size,
            checksum=draw(checksums()),
        ),
        status=status,
        pipeline_version=draw(pipeline_versions()),
        correlation_id=draw(correlation_ids()),
        created_at=created_at,
        updated_at=finished_at if terminal else started_at,
        page_count=draw(st.integers(min_value=1, max_value=MAX_PAGES))
        if successful
        else draw(st.none() | st.integers(min_value=1, max_value=MAX_PAGES)),
        processing_started_at=None if status is DocumentStatus.PENDING else started_at,
        processed_at=finished_at if terminal else None,
        failure_code=draw(st.text(TEXT_ALPHABET, min_size=1, max_size=64))
        if status is DocumentStatus.FAILED
        else None,
        failure_message=draw(st.text(TEXT_ALPHABET, min_size=0, max_size=200))
        if status is DocumentStatus.FAILED
        else None,
        failure_stage=draw(st.sampled_from(ProcessingStage))
        if status is DocumentStatus.FAILED
        else None,
    )


@st.composite
def illegible_spans(draw: st.DrawFn, content: str) -> tuple[IllegibleSpan, ...]:
    """Непересекающиеся отсортированные диапазоны внутри текста."""
    if not content:
        return ()
    # Каждому диапазону нужны две различные границы, поэтому их число ограничено
    # длиной текста, а не только желаемым.
    count = draw(st.integers(min_value=1, max_value=max(1, min(3, len(content) // 2))))
    bounds = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=len(content)),
                min_size=count * 2,
                max_size=count * 2,
                unique=True,
            )
        )
    )
    spans: list[IllegibleSpan] = []
    for index in range(count):
        start, end = bounds[index * 2], bounds[index * 2 + 1]
        span = TextSpan(start, end)
        spans.append(
            IllegibleSpan(
                span=span,
                confidence=draw(
                    st.floats(min_value=0.0, max_value=0.75, allow_nan=False).map(
                        OcrConfidence
                    )
                ),
                reason=draw(
                    st.sampled_from(
                        [
                            IllegibleReason.LOW_OCR_CONFIDENCE,
                            IllegibleReason.IMAGE_TOO_NOISY,
                            IllegibleReason.HANDWRITING,
                            IllegibleReason.GLYPH_MAPPING_FAILED,
                        ]
                    )
                ),
                raw_text=span.slice_of(content),
                bbox=draw(st.none() | bounding_boxes()),
                line_number=draw(st.none() | st.integers(min_value=1, max_value=100)),
            )
        )
    return tuple(spans)


def bounding_boxes() -> st.SearchStrategy[BoundingBox]:
    """Нормализованный прямоугольник."""
    return st.builds(
        lambda x0, y0, w, h: BoundingBox(x0, y0, min(x0 + w, 1.0), min(y0 + h, 1.0)),
        x0=st.floats(min_value=0.0, max_value=0.5),
        y0=st.floats(min_value=0.0, max_value=0.5),
        w=st.floats(min_value=0.01, max_value=0.5),
        h=st.floats(min_value=0.01, max_value=0.5),
    ).filter(lambda box: box.x1 > box.x0 and box.y1 > box.y0)


@st.composite
def pages(
    draw: st.DrawFn,
    document_id: DocumentId | None = None,
    pipeline_version: PipelineVersion | None = None,
) -> DocumentPage:
    """Страница, согласованная со своим статусом и способом извлечения."""
    status = draw(st.sampled_from(PageStatus))
    page_id = PageId(draw(uuids()))
    owner = document_id or DocumentId(draw(uuids()))
    number = PageNumber(draw(st.integers(min_value=1, max_value=MAX_PAGES)))
    version = pipeline_version or draw(pipeline_versions())
    created_at = draw(timestamps())
    if status is PageStatus.FAILED:
        return DocumentPage(
            id=page_id,
            document_id=owner,
            number=number,
            pipeline_version=version,
            status=status,
            text=RecognizedText.not_extracted(),
            created_at=created_at,
            failure=PageFailure(
                reason=draw(st.sampled_from(PageFailureReason)),
                message=draw(st.text(TEXT_ALPHABET, min_size=0, max_size=200)),
                recoverable=draw(st.booleans()),
            ),
            warnings=draw(warning_lists()),
        )

    content = draw(st.text(TEXT_ALPHABET, min_size=0, max_size=120))
    if status is PageStatus.EXTRACTED:
        method = draw(
            st.sampled_from(
                [
                    ExtractionMethod.TEXT_LAYER,
                    ExtractionMethod.OCR,
                    ExtractionMethod.HYBRID,
                ]
            )
        )
        spans: tuple[IllegibleSpan, ...] = ()
    else:
        method = draw(st.sampled_from([ExtractionMethod.OCR, ExtractionMethod.HYBRID]))
        spans = draw(illegible_spans(content))
        if not spans:
            content = draw(st.text(TEXT_ALPHABET, min_size=2, max_size=120))
            spans = draw(illegible_spans(content))

    ocr_based = method.is_ocr_based
    return DocumentPage(
        id=page_id,
        document_id=owner,
        number=number,
        pipeline_version=version,
        status=status,
        text=RecognizedText(
            content=content,
            method=method,
            confidence=draw(confidences()) if ocr_based else None,
            illegible_spans=spans,
        ),
        created_at=created_at,
        image_ref=draw(object_refs()) if ocr_based else None,
        render_dpi=draw(st.integers(min_value=MIN_RENDER_DPI, max_value=MAX_RENDER_DPI))
        if ocr_based
        else None,
        warnings=draw(warning_lists()),
    )


def warning_lists() -> st.SearchStrategy[tuple[str, ...]]:
    """Список предупреждений страницы."""
    return st.lists(
        st.text(TEXT_ALPHABET, min_size=1, max_size=40),
        max_size=4,
    ).map(tuple)


@st.composite
def chunks(draw: st.DrawFn) -> DocumentChunk:
    """Чанк, текст которого совпадает со своим диапазоном."""
    content = draw(st.text(TEXT_ALPHABET, min_size=1, max_size=200).filter(str.strip))
    start = draw(st.integers(min_value=0, max_value=10_000))
    method = draw(
        st.sampled_from(
            [ExtractionMethod.TEXT_LAYER, ExtractionMethod.OCR, ExtractionMethod.HYBRID]
        )
    )
    return DocumentChunk(
        id=ChunkId(draw(uuids())),
        document_id=DocumentId(draw(uuids())),
        page_id=PageId(draw(uuids())),
        page_number=PageNumber(draw(st.integers(min_value=1, max_value=MAX_PAGES))),
        ordinal=draw(st.integers(min_value=0, max_value=500)),
        content=content,
        span=TextSpan(start, start + len(content)),
        method=method,
        avg_confidence=None
        if method is ExtractionMethod.TEXT_LAYER
        else draw(confidences()),
        illegible_span_count=draw(st.integers(min_value=0, max_value=10)),
        chunking_version=draw(chunking_versions()),
        checksum=Checksum(
            ChecksumAlgorithm.SHA256, Checksum.sha256_of(content.encode("utf-8")).value
        ),
        token_count=draw(st.integers(min_value=1, max_value=MAX_CHUNK_TOKENS)),
        heading_path=draw(
            st.lists(st.text(TEXT_ALPHABET, min_size=1, max_size=30), max_size=3).map(
                tuple
            )
        ),
        overlap_prefix_chars=draw(
            st.integers(
                min_value=0,
                max_value=min(len(content) - 1, MAX_CHUNK_OVERLAP_CHARS),
            )
        ),
    )


@st.composite
def jobs(draw: st.DrawFn, document_id: DocumentId | None = None) -> ProcessingJob:
    """Прогон обработки, согласованный со своим статусом."""
    status = draw(st.sampled_from(JobStatus))
    scheduled_at = draw(timestamps())
    started_at = scheduled_at + timedelta(seconds=draw(st.integers(0, 3600)))
    finished_at = started_at + timedelta(seconds=draw(st.integers(0, 3600)))
    running = status is not JobStatus.QUEUED
    terminal = status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
    text_layer = draw(st.integers(min_value=0, max_value=50))
    ocr = draw(st.integers(min_value=0, max_value=50))
    hybrid = draw(st.integers(min_value=0, max_value=50))
    failed = draw(st.integers(min_value=0, max_value=50))
    total = text_layer + ocr + hybrid + failed
    return ProcessingJob(
        id=JobId(draw(uuids())),
        document_id=document_id or DocumentId(draw(uuids())),
        event_id=EventId(draw(uuids())),
        correlation_id=draw(correlation_ids()),
        pipeline_version=draw(pipeline_versions()),
        status=status,
        attempt=draw(st.integers(min_value=1, max_value=10)),
        scheduled_at=scheduled_at,
        started_at=started_at if running else None,
        finished_at=finished_at if terminal else None,
        pages_total=total
        if status is JobStatus.SUCCEEDED
        else draw(st.none() | st.just(total)),
        pages_text_layer=text_layer,
        pages_ocr=ocr,
        pages_hybrid=hybrid,
        pages_failed=failed,
        chunks_created=draw(st.integers(min_value=0, max_value=2000)),
        error_code=draw(st.text(TEXT_ALPHABET, min_size=1, max_size=64))
        if status is JobStatus.FAILED
        else None,
        error_message=draw(st.text(TEXT_ALPHABET, min_size=0, max_size=200))
        if status is JobStatus.FAILED
        else None,
        stage=draw(st.sampled_from(ProcessingStage))
        if status is JobStatus.FAILED
        else None,
    )
