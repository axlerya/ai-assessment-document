"""Прогон корпуса через настоящую обработку документа.

Собирается тот же `ProcessDocument`, что работает в проде, — через
`build_processing`. Подменено ровно одно: хранилище читает файлы корпуса с
диска. Всё остальное настоящее, включая базу, распознавание и сборку чанков.

Строку `documents` здесь создаёт стенд, потому что в проде её создаёт сервис
приёма файлов, которого в корпусе нет.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from document_worker.application.dto.commands import ProcessDocumentCommand
from document_worker.domain.entities.document import Document
from document_worker.domain.value_objects.enums import DocumentStatus
from document_worker.domain.value_objects.identifiers import (
    CorrelationId,
    DocumentId,
    EventId,
)
from document_worker.domain.value_objects.storage import (
    FileSize,
    MimeType,
    ObjectRef,
    SourceFile,
)
from document_worker.infrastructure.persistence.mappers.document import document_to_row

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from document_worker.bootstrap.composition import Processing

CORPUS_BUCKET = "corpus"
SOURCE_NAME = "source.pdf"


@dataclass(frozen=True, slots=True)
class PageOutcome:
    """Что сервис записал про одну страницу."""

    number: int
    text: str
    extraction_method: str
    status: str
    mean_confidence: float | None
    illegible_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ChunkOutcome:
    """Что сервис записал про один чанк."""

    page_number: int
    start_offset: int
    end_offset: int
    text: str
    token_count: int
    page_text_matches: bool


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    """Результат обработки одного документа корпуса."""

    doc_id: str
    status: str
    duration_s: float
    pages: tuple[PageOutcome, ...]
    chunks: tuple[ChunkOutcome, ...]


async def process_document(
    processing: Processing,
    *,
    doc_id: str,
) -> DocumentOutcome:
    """Проводит документ корпуса через настоящую обработку."""
    document = _pending_document(doc_id)
    await _insert(processing.engine, document)
    started = time.monotonic()
    result = await processing.process_document.execute(_command(document))
    duration = time.monotonic() - started
    return DocumentOutcome(
        doc_id=doc_id,
        status=result.status.value,
        duration_s=duration,
        pages=await _pages(processing.engine, document.id.value),
        chunks=await _chunks(processing.engine, document.id.value),
    )


def _pending_document(doc_id: str) -> Document:
    now = datetime.now(UTC)
    return Document(
        id=DocumentId(uuid.uuid4()),
        source=SourceFile(
            ref=ObjectRef(bucket=CORPUS_BUCKET, key=f"{doc_id}/{SOURCE_NAME}"),
            mime_type=MimeType(MimeType.PDF),
            # Размер объявленный: настоящий сервис тоже узнаёт его от приёмника
            # и перепроверяет по факту скачивания.
            size=FileSize(1),
            checksum=None,
        ),
        status=DocumentStatus.PENDING,
        pipeline_version=None,
        correlation_id=CorrelationId(str(uuid.uuid4())),
        created_at=now,
        updated_at=now,
    )


def _command(document: Document) -> ProcessDocumentCommand:
    return ProcessDocumentCommand(
        event_id=EventId(uuid.uuid4()),
        document_id=document.id,
        correlation_id=document.correlation_id,
        object_ref=document.source.ref,
        mime_type=document.source.mime_type,
        occurred_at=datetime.now(UTC),
    )


async def _insert(engine: AsyncEngine, document: Document) -> None:
    sessions = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(document_to_row(document))
        await session.commit()


async def _pages(
    engine: AsyncEngine, document_id: uuid.UUID
) -> tuple[PageOutcome, ...]:
    async with engine.connect() as connection:
        rows = list(
            await connection.execute(
                text(
                    "SELECT id, page_number, text, extraction_method, status,"
                    " ocr_confidence FROM document_pages WHERE document_id = :id"
                    " ORDER BY page_number"
                ),
                {"id": document_id},
            )
        )
        spans = list(
            await connection.execute(
                text(
                    "SELECT s.page_id, s.start_offset, s.end_offset"
                    " FROM document_illegible_spans s"
                    " JOIN document_pages p ON p.id = s.page_id"
                    " WHERE p.document_id = :id ORDER BY s.span_index"
                ),
                {"id": document_id},
            )
        )
    by_page: dict[uuid.UUID, list[tuple[int, int]]] = {}
    for span in spans:
        by_page.setdefault(span.page_id, []).append(
            (span.start_offset, span.end_offset)
        )
    return tuple(
        PageOutcome(
            number=row.page_number,
            text=row.text,
            extraction_method=row.extraction_method,
            status=row.status,
            mean_confidence=(
                float(row.ocr_confidence) if row.ocr_confidence is not None else None
            ),
            illegible_spans=tuple(by_page.get(row.id, ())),
        )
        for row in rows
    )


async def _chunks(
    engine: AsyncEngine,
    document_id: uuid.UUID,
) -> tuple[ChunkOutcome, ...]:
    async with engine.connect() as connection:
        rows = list(
            await connection.execute(
                text(
                    "SELECT c.page_number, c.start_offset, c.end_offset, c.text,"
                    " c.token_count,"
                    " c.text = substring(p.text from c.start_offset + 1"
                    " for c.end_offset - c.start_offset) AS matches"
                    " FROM document_chunks c JOIN document_pages p ON p.id = c.page_id"
                    " WHERE c.document_id = :id ORDER BY c.chunk_index"
                ),
                {"id": document_id},
            )
        )
    return tuple(
        ChunkOutcome(
            page_number=row.page_number,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            text=row.text,
            token_count=row.token_count,
            page_text_matches=bool(row.matches),
        )
        for row in rows
    )
