"""Доступ к страницам документа."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from document_worker.application.dto.results import PageSummaryDTO
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import ExtractionMethod, PageStatus
from document_worker.domain.value_objects.identifiers import PageId
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.infrastructure.persistence.mappers.page import (
    page_spans_to_values,
    page_to_domain,
    page_to_values,
)
from document_worker.infrastructure.persistence.models.page import (
    DocumentPageRow,
    IllegibleSpanRow,
)

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.value_objects.identifiers import DocumentId
    from document_worker.domain.value_objects.versioning import PipelineVersion

PAGE_CONSTRAINT = "uq__document_pages__document__version__number"
SPAN_CONSTRAINT = "uq__illegible_spans__page__index"


class SqlAlchemyDocumentPageRepository:
    """Страницы: идемпотентная вставка вместе с диапазонами и чтение."""

    def __init__(self, session: AsyncSession) -> None:
        """Работает в транзакции переданной сессии."""
        self._session = session

    async def add(self, page: DocumentPage) -> bool:
        """Пишет страницу вместе с её неразборчивыми диапазонами.

        False — строка уже была: повторную доставку гасит ограничение, а не
        проверка в коде.
        """
        statement = (
            pg_insert(DocumentPageRow)
            .values(**page_to_values(page))
            .on_conflict_do_nothing(constraint=PAGE_CONSTRAINT)
            .returning(DocumentPageRow.id)
        )
        inserted = (await self._session.execute(statement)).scalar_one_or_none()
        if inserted is None:
            return False

        spans = page_spans_to_values(page)
        if spans:
            await self._session.execute(
                pg_insert(IllegibleSpanRow)
                .values(spans)
                .on_conflict_do_nothing(constraint=SPAN_CONSTRAINT)
            )
        return True

    async def list_persisted_page_numbers(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> frozenset[int]:
        """Номера уже сохранённых страниц — вход для возобновления."""
        statement = select(DocumentPageRow.page_number).where(
            self._scope(document_id, pipeline_version)
        )
        return frozenset((await self._session.execute(statement)).scalars().all())

    async def list_summaries(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> tuple[PageSummaryDTO, ...]:
        """Метрики страниц без их текста."""
        illegible_chars = func.coalesce(
            func.sum(IllegibleSpanRow.end_offset - IllegibleSpanRow.start_offset), 0
        )
        statement = (
            select(
                DocumentPageRow.id,
                DocumentPageRow.page_number,
                DocumentPageRow.status,
                DocumentPageRow.extraction_method,
                DocumentPageRow.ocr_confidence,
                DocumentPageRow.text_length,
                illegible_chars.label("illegible_chars"),
            )
            .outerjoin(IllegibleSpanRow, IllegibleSpanRow.page_id == DocumentPageRow.id)
            .where(self._scope(document_id, pipeline_version))
            .group_by(DocumentPageRow.id)
            .order_by(DocumentPageRow.page_number)
        )
        return tuple(
            PageSummaryDTO(
                page_id=PageId(row.id),
                page_number=PageNumber(row.page_number),
                status=PageStatus(row.status),
                method=ExtractionMethod(row.extraction_method),
                confidence=None
                if row.ocr_confidence is None
                else OcrConfidence(float(row.ocr_confidence)),
                char_count=row.text_length,
                illegible_char_count=int(row.illegible_chars),
            )
            for row in await self._session.execute(statement)
        )

    async def load_pages(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
        *,
        statuses: frozenset[PageStatus],
    ) -> tuple[DocumentPage, ...]:
        """Читает страницы указанных статусов целиком."""
        statement = (
            select(DocumentPageRow)
            .where(
                self._scope(document_id, pipeline_version),
                DocumentPageRow.status.in_([status.value for status in statuses]),
            )
            .order_by(DocumentPageRow.page_number)
        )
        rows = (await self._session.execute(statement)).scalars().all()
        spans = await self._spans_of([row.id for row in rows])
        return tuple(page_to_domain(row, spans.get(row.id, ())) for row in rows)

    async def count(
        self,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> int:
        """Сколько страниц сохранено."""
        statement = select(func.count()).where(
            self._scope(document_id, pipeline_version)
        )
        return int((await self._session.execute(statement)).scalar_one())

    async def _spans_of(
        self,
        page_ids: list[object],
    ) -> dict[object, tuple[IllegibleSpanRow, ...]]:
        if not page_ids:
            return {}
        statement = (
            select(IllegibleSpanRow)
            .where(IllegibleSpanRow.page_id.in_(page_ids))
            .order_by(IllegibleSpanRow.page_id, IllegibleSpanRow.span_index)
        )
        grouped: dict[object, list[IllegibleSpanRow]] = {}
        for span in (await self._session.execute(statement)).scalars():
            grouped.setdefault(span.page_id, []).append(span)
        return {page_id: tuple(spans) for page_id, spans in grouped.items()}

    @staticmethod
    def _scope(
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
    ) -> ColumnElement[bool]:
        return and_(
            DocumentPageRow.document_id == document_id.value,
            DocumentPageRow.pipeline_version == str(pipeline_version),
        )
