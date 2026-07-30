"""Готовые доменные сущности для тестов, которым нужен один конкретный случай.

Генератор из `tests.strategies` покрывает пространство значений; здесь лежат
представители каждого способа извлечения и каждого статуса, на которых удобно
проверять поведение, а не инварианты.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

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
from document_worker.domain.value_objects.recognized_text import RecognizedText
from document_worker.domain.value_objects.storage import (
    Checksum,
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

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
PIPELINE_VERSION = PipelineVersion(1, 0, 0)
CHUNKING_VERSION = ChunkingVersion(1, 0, 0)
PAGE_TEXT = "договор поставки товаров и услуг"


def new_correlation_id() -> CorrelationId:
    """Новый сквозной идентификатор запроса."""
    return CorrelationId(str(uuid.uuid4()))


def make_document(
    *,
    status: DocumentStatus = DocumentStatus.PENDING,
    document_id: DocumentId | None = None,
    pipeline_version: PipelineVersion = PIPELINE_VERSION,
    page_count: int | None = None,
) -> Document:
    """Документ в указанном статусе, согласованный с ограничениями схемы."""
    terminal = status.is_terminal
    successful = status.is_successful
    return Document(
        id=document_id or DocumentId(uuid.uuid4()),
        source=SourceFile(
            ref=ObjectRef(bucket="documents", key=f"{uuid.uuid4().hex}/source.pdf"),
            mime_type=MimeType(MimeType.PDF),
            size=FileSize(2048),
            checksum=Checksum.sha256_of(b"source") if successful else None,
        ),
        status=status,
        pipeline_version=pipeline_version,
        correlation_id=new_correlation_id(),
        created_at=NOW,
        updated_at=NOW + timedelta(minutes=1),
        page_count=page_count
        if page_count is not None
        else (2 if successful else None),
        processing_started_at=None if status is DocumentStatus.PENDING else NOW,
        processed_at=NOW + timedelta(minutes=1) if terminal else None,
        failure_code="corrupted_document" if status is DocumentStatus.FAILED else None,
        failure_message="файл не читается" if status is DocumentStatus.FAILED else None,
        failure_stage=ProcessingStage.VALIDATION
        if status is DocumentStatus.FAILED
        else None,
    )


def version_of(document: Document) -> PipelineVersion:
    """Версия, которой обрабатывается документ; у необработанного её нет."""
    return document.pipeline_version or PIPELINE_VERSION


def make_text_layer_page(document: Document, *, number: int = 1) -> DocumentPage:
    """Страница из текстового слоя: уверенности нет, рендера нет."""
    return DocumentPage.from_text_layer(
        page_id=PageId(uuid.uuid4()),
        document_id=document.id,
        number=PageNumber(number),
        pipeline_version=version_of(document),
        content=PAGE_TEXT,
        now=NOW,
    )


def make_ocr_page(
    document: Document,
    *,
    number: int = 2,
    confidence: float = 0.9312,
) -> DocumentPage:
    """Страница, прочитанная распознаванием целиком."""
    return DocumentPage(
        id=PageId(uuid.uuid4()),
        document_id=document.id,
        number=PageNumber(number),
        pipeline_version=version_of(document),
        status=PageStatus.EXTRACTED,
        text=RecognizedText(
            content=PAGE_TEXT,
            method=ExtractionMethod.OCR,
            confidence=OcrConfidence(confidence),
        ),
        created_at=NOW,
        image_ref=ObjectRef(bucket="renders", key=f"{uuid.uuid4().hex}/{number}.png"),
        render_dpi=300,
        warnings=("low_mean_confidence",),
    )


def make_illegible_page(document: Document, *, number: int = 3) -> DocumentPage:
    """Страница с неразборчивым фрагментом: диапазон, рамка и номер строки."""
    span = TextSpan(0, 7)
    return DocumentPage(
        id=PageId(uuid.uuid4()),
        document_id=document.id,
        number=PageNumber(number),
        pipeline_version=version_of(document),
        status=PageStatus.PARTIALLY_ILLEGIBLE,
        text=RecognizedText(
            content=PAGE_TEXT,
            method=ExtractionMethod.HYBRID,
            confidence=OcrConfidence(0.5),
            illegible_spans=(
                IllegibleSpan(
                    span=span,
                    confidence=OcrConfidence(0.4),
                    reason=IllegibleReason.LOW_OCR_CONFIDENCE,
                    raw_text=span.slice_of(PAGE_TEXT),
                    bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
                    line_number=3,
                ),
            ),
        ),
        created_at=NOW,
        image_ref=ObjectRef(bucket="renders", key=f"{uuid.uuid4().hex}/{number}.png"),
        render_dpi=400,
    )


def make_failed_page(document: Document, *, number: int = 4) -> DocumentPage:
    """Страница, которую не удалось прочитать."""
    return DocumentPage.failed(
        page_id=PageId(uuid.uuid4()),
        document_id=document.id,
        number=PageNumber(number),
        pipeline_version=version_of(document),
        reason=PageFailureReason.RENDER_FAILED,
        message="страница не отрендерилась",
        now=NOW,
        recoverable=True,
    )


def make_chunk(
    page: DocumentPage,
    *,
    ordinal: int = 0,
    start: int = 0,
) -> DocumentChunk:
    """Чанк, вырезанный из текста страницы."""
    span = TextSpan(start, start + len(PAGE_TEXT))
    return DocumentChunk(
        id=ChunkId(uuid.uuid4()),
        document_id=page.document_id,
        page_id=page.id,
        page_number=page.number,
        ordinal=ordinal,
        content=PAGE_TEXT,
        span=span,
        method=page.method,
        avg_confidence=page.confidence,
        illegible_span_count=len(page.illegible_spans),
        chunking_version=CHUNKING_VERSION,
        checksum=Checksum.sha256_of(PAGE_TEXT.encode("utf-8")),
        token_count=12,
        heading_path=("Раздел 1", "Пункт 1.1"),
        overlap_prefix_chars=4,
    )


def make_job(
    document: Document,
    *,
    status: JobStatus = JobStatus.RUNNING,
    event_id: EventId | None = None,
) -> ProcessingJob:
    """Прогон обработки документа."""
    terminal = status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
    return ProcessingJob(
        id=JobId(uuid.uuid4()),
        document_id=document.id,
        event_id=event_id or EventId(uuid.uuid4()),
        correlation_id=document.correlation_id,
        pipeline_version=version_of(document),
        status=status,
        attempt=1,
        scheduled_at=NOW,
        started_at=None if status is JobStatus.QUEUED else NOW,
        finished_at=NOW + timedelta(minutes=2) if terminal else None,
        pages_total=2,
        pages_text_layer=1,
        pages_ocr=1,
        pages_hybrid=0,
        pages_failed=0,
        chunks_created=7,
    )
