"""Страница ↔ строки document_pages и document_illegible_spans."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from document_worker.domain.constants import NS_DOCWORKER
from document_worker.domain.entities.document_page import DocumentPage
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    IllegibleReason,
    PageFailureReason,
    PageStatus,
)
from document_worker.domain.value_objects.geometry import BoundingBox
from document_worker.domain.value_objects.identifiers import DocumentId, PageId
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.quality import PageFailure
from document_worker.domain.value_objects.recognized_text import RecognizedText
from document_worker.domain.value_objects.storage import ObjectRef
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan
from document_worker.domain.value_objects.versioning import PipelineVersion
from document_worker.infrastructure.persistence.models.page import (
    DocumentPageRow,
    IllegibleSpanRow,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def span_id(page_id: PageId, index: int) -> uuid.UUID:
    """Идентификатор диапазона, выводимый из страницы и порядкового номера.

    Детерминированность нужна повтору: вторая доставка обязана попасть в ту же
    строку, а не создать вторую.
    """
    return uuid.uuid5(NS_DOCWORKER, f"{page_id}|{index}")


def page_to_row(page: DocumentPage) -> DocumentPageRow:
    """Собирает строку страницы без её диапазонов."""
    failure = page.failure
    return DocumentPageRow(
        id=page.id.value,
        document_id=page.document_id.value,
        pipeline_version=str(page.pipeline_version),
        page_number=int(page.number),
        status=page.status.value,
        extraction_method=page.method.value,
        text=page.text.content,
        text_length=page.text.char_count,
        ocr_confidence=_decimal(page.confidence),
        illegible_span_count=len(page.illegible_spans),
        image_bucket=page.image_ref.bucket if page.image_ref else None,
        image_key=page.image_ref.key if page.image_ref else None,
        render_dpi=page.render_dpi,
        warnings=list(page.warnings),
        failure_reason=failure.reason.value if failure else None,
        failure_message=failure.message if failure else None,
        failure_recoverable=failure.recoverable if failure else None,
        created_at=page.created_at,
    )


def page_spans_to_rows(page: DocumentPage) -> list[IllegibleSpanRow]:
    """Собирает строки неразборчивых диапазонов страницы."""
    return [
        IllegibleSpanRow(
            id=span_id(page.id, index),
            page_id=page.id.value,
            span_index=index,
            start_offset=span.span.start,
            end_offset=span.span.end,
            reason=span.reason.value,
            confidence=Decimal(str(span.confidence.value)),
            raw_text=span.raw_text,
            line_number=span.line_number,
            bbox_x0=span.bbox.x0 if span.bbox else None,
            bbox_y0=span.bbox.y0 if span.bbox else None,
            bbox_x1=span.bbox.x1 if span.bbox else None,
            bbox_y1=span.bbox.y1 if span.bbox else None,
            created_at=page.created_at,
        )
        for index, span in enumerate(page.illegible_spans)
    ]


def page_to_domain(
    row: DocumentPageRow,
    spans: Sequence[IllegibleSpanRow],
) -> DocumentPage:
    """Восстанавливает страницу из строки и её диапазонов.

    Читаемость и отказ берутся из собственных колонок, а не выводятся из
    статуса: вывести их оттуда нельзя, статус описывает не всё.
    """
    method = ExtractionMethod(row.extraction_method)
    return DocumentPage(
        id=PageId(row.id),
        document_id=DocumentId(row.document_id),
        number=PageNumber(row.page_number),
        pipeline_version=PipelineVersion.parse(row.pipeline_version),
        status=PageStatus(row.status),
        text=RecognizedText(
            content=row.text,
            method=method,
            confidence=_confidence(row.ocr_confidence),
            illegible_spans=tuple(_span_to_domain(span) for span in spans),
        ),
        created_at=row.created_at,
        image_ref=ObjectRef(bucket=row.image_bucket, key=row.image_key)
        if row.image_bucket and row.image_key
        else None,
        render_dpi=row.render_dpi,
        failure=_failure_of(row),
        warnings=tuple(row.warnings),
    )


def _failure_of(row: DocumentPageRow) -> PageFailure | None:
    if row.failure_reason is None:
        return None
    return PageFailure(
        reason=PageFailureReason(row.failure_reason),
        message=row.failure_message or "",
        recoverable=bool(row.failure_recoverable),
    )


def _span_to_domain(row: IllegibleSpanRow) -> IllegibleSpan:
    return IllegibleSpan(
        span=TextSpan(row.start_offset, row.end_offset),
        confidence=OcrConfidence(float(row.confidence)),
        reason=IllegibleReason(row.reason),
        raw_text=row.raw_text,
        bbox=BoundingBox(row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1)
        if row.bbox_x0 is not None
        and row.bbox_y0 is not None
        and row.bbox_x1 is not None
        and row.bbox_y1 is not None
        else None,
        line_number=row.line_number,
    )


def _decimal(confidence: OcrConfidence | None) -> Decimal | None:
    return None if confidence is None else Decimal(str(confidence.value))


def _confidence(value: Decimal | None) -> OcrConfidence | None:
    return None if value is None else OcrConfidence(float(value))
