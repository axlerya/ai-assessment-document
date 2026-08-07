"""Наполнение таблиц document-worker для тестов, которым нужен корпус.

Живёт рядом с фабриками, а не в тесте read-model: чужие таблицы нужны и
сценарию индексации, а вторая копия этих вставок разошлась бы с первой.

Единственное место вне read-model, где упоминаются имена чужих таблиц, —
поэтому тест-страж границы отбирает файлы только внутри пакета сервиса.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncConnection

CHUNK_TEXTS: tuple[str, ...] = (
    "Договор поставки № 12/АБ от 3 марта 2024 года заключён сторонами.",
    "Покупатель обязуется оплатить товар в течение десяти банковских дней.",
    "Общая стоимость поставки составляет 1 250 000 рублей без учёта НДС.",
)

_DOCUMENT = text(
    "INSERT INTO documents (id, bucket, object_key, declared_mime_type,"
    " declared_size_bytes, status, pipeline_version, page_count, checksum,"
    " size_bytes, correlation_id, processing_finished_at)"
    " VALUES (:id, 'documents', :key, 'application/pdf', 1024, :status,"
    " :pipeline_version, 1, :checksum, 1024, :correlation_id, now())"
)

_PAGE = text(
    "INSERT INTO document_pages (id, document_id, pipeline_version, page_number,"
    " status, extraction_method, text, text_length)"
    " VALUES (:id, :document_id, :pipeline_version, 1, 'extracted', 'text_layer',"
    " :text, :length)"
)

_CHUNK = text(
    "INSERT INTO document_chunks (id, document_id, page_id, page_number,"
    " chunking_version, chunk_index, start_offset, end_offset, text, token_count,"
    " extraction_method, avg_ocr_confidence, illegible_span_count, heading_path,"
    " content_hash)"
    " VALUES (:id, :document_id, :page_id, 1, :chunking_version, :chunk_index,"
    " :start, :end, :text, :token_count, :method, :confidence, 0,"
    " CAST('[]' AS jsonb), :content_hash)"
)


async def seed_document(
    connection: AsyncConnection,
    *,
    document_id: uuid.UUID,
    texts: Sequence[str] = CHUNK_TEXTS,
    status: str = "processed",
    chunking_version: str = "1.0.0",
    pipeline_version: str = "1.0.0",
) -> tuple[uuid.UUID, ...]:
    """Кладёт документ со страницей и чанками, возвращает их идентификаторы."""
    await connection.execute(
        _DOCUMENT,
        {
            "id": document_id,
            "key": f"documents/{document_id}/source.pdf",
            "status": status,
            "pipeline_version": pipeline_version,
            "checksum": "a" * 64,
            "correlation_id": str(uuid.uuid4()),
        },
    )
    page_id = uuid.uuid4()
    joined = "\n".join(texts)
    await connection.execute(
        _PAGE,
        {
            "id": page_id,
            "document_id": document_id,
            "pipeline_version": pipeline_version,
            "text": joined,
            "length": len(joined),
        },
    )
    return await add_chunks(
        connection,
        document_id=document_id,
        page_id=page_id,
        texts=texts,
        chunking_version=chunking_version,
    )


async def add_chunks(
    connection: AsyncConnection,
    *,
    document_id: uuid.UUID,
    page_id: uuid.UUID,
    texts: Sequence[str],
    chunking_version: str = "1.0.0",
) -> tuple[uuid.UUID, ...]:
    """Добавляет чанки указанной версии чанкования."""
    created: list[uuid.UUID] = []
    offset = 0
    for index, chunk_text in enumerate(texts):
        chunk_id = uuid.uuid4()
        await connection.execute(
            _CHUNK,
            {
                "id": chunk_id,
                "document_id": document_id,
                "page_id": page_id,
                "chunking_version": chunking_version,
                "chunk_index": index,
                "start": offset,
                "end": offset + len(chunk_text),
                "text": chunk_text,
                "token_count": max(1, len(chunk_text) // 4),
                "method": "text_layer",
                "confidence": None,
                "content_hash": f"{index:064d}",
            },
        )
        created.append(chunk_id)
        offset += len(chunk_text) + 1
    return tuple(created)


async def page_of(connection: AsyncConnection, document_id: uuid.UUID) -> uuid.UUID:
    """Идентификатор единственной страницы документа."""
    row = (
        await connection.execute(
            text("SELECT id FROM document_pages WHERE document_id = :id"),
            {"id": document_id},
        )
    ).one()
    return uuid.UUID(str(row.id))
