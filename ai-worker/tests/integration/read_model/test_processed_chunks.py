"""Чтение результатов document-worker: единственная точка доступа к ним."""

from __future__ import annotations

import ast
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

from ai_worker.application.errors import PermanentError
from ai_worker.domain.value_objects.enums import ExtractionMethod, SourceStatus
from ai_worker.domain.value_objects.identifiers import ChunkId, DocumentId
from ai_worker.domain.value_objects.versioning import ChunkingVersion
from ai_worker.infrastructure.persistence.read_model.processed_chunks import (
    SqlAlchemyProcessedChunkReader,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration

SERVICE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = SERVICE_ROOT / "ai_worker"
READ_MODEL = PACKAGE_ROOT / "infrastructure" / "persistence" / "read_model"

# Имена чужих таблиц. Упоминание любого из них вне read-model означает, что
# граница между сервисами перестала быть проверяемой.
FOREIGN_TABLES = ("documents", "document_pages", "document_chunks")

CHUNK_TEXT = "Договор поставки № 12/АБ от 3 марта 2024 года заключён сторонами."


async def _insert_document(
    connection: AsyncConnection,
    *,
    document_id: uuid.UUID,
    status: str = "processed",
) -> None:
    await connection.execute(
        text(
            "INSERT INTO documents (id, bucket, object_key, declared_mime_type,"
            " declared_size_bytes, status, pipeline_version, page_count, checksum,"
            " size_bytes, processing_finished_at)"
            " VALUES (:id, 'documents', :key, 'application/pdf', 1024, :status,"
            " '1.0.0', 3, :checksum, 1024, now())"
        ),
        {
            "id": document_id,
            "key": f"documents/{document_id}/source.pdf",
            "status": status,
            "checksum": "a" * 64,
        },
    )


async def _insert_page(
    connection: AsyncConnection,
    *,
    document_id: uuid.UUID,
    page_id: uuid.UUID,
    number: int,
) -> None:
    await connection.execute(
        text(
            "INSERT INTO document_pages (id, document_id, pipeline_version,"
            " page_number, status, extraction_method, text, text_length)"
            " VALUES (:id, :document_id, '1.0.0', :number, 'extracted',"
            " 'text_layer', :text, :length)"
        ),
        {
            "id": page_id,
            "document_id": document_id,
            "number": number,
            "text": CHUNK_TEXT,
            "length": len(CHUNK_TEXT),
        },
    )


async def _insert_chunk(
    connection: AsyncConnection,
    *,
    document_id: uuid.UUID,
    page_id: uuid.UUID,
    page_number: int,
    chunk_index: int,
    chunking_version: str = "1.0.0",
    method: str = "text_layer",
    confidence: float | None = None,
    end: int | None = None,
) -> uuid.UUID:
    chunk_id = uuid.uuid4()
    stop = end if end is not None else len(CHUNK_TEXT)
    await connection.execute(
        text(
            "INSERT INTO document_chunks (id, document_id, page_id, page_number,"
            " chunking_version, chunk_index, start_offset, end_offset, text,"
            " token_count, extraction_method, avg_ocr_confidence,"
            " illegible_span_count, heading_path, content_hash)"
            " VALUES (:id, :document_id, :page_id, :page_number, :version, :index,"
            " 0, :end, :text, 42, :method, :confidence, 0,"
            " CAST(:heading AS jsonb), :hash)"
        ),
        {
            "id": chunk_id,
            "document_id": document_id,
            "page_id": page_id,
            "page_number": page_number,
            "version": chunking_version,
            "index": chunk_index,
            "end": stop,
            "text": CHUNK_TEXT[:stop],
            "method": method,
            "confidence": confidence,
            "heading": '["Предмет договора"]',
            "hash": "b" * 64,
        },
    )
    return chunk_id


async def _prepared(connection: AsyncConnection, **kwargs: Any) -> uuid.UUID:
    document_id = uuid.uuid4()
    page_id = uuid.uuid4()
    await _insert_document(connection, document_id=document_id, **kwargs)
    await _insert_page(connection, document_id=document_id, page_id=page_id, number=1)
    await _insert_chunk(
        connection,
        document_id=document_id,
        page_id=page_id,
        page_number=1,
        chunk_index=0,
    )
    return document_id


def _reader(connection: AsyncConnection) -> SqlAlchemyProcessedChunkReader:
    return SqlAlchemyProcessedChunkReader(connection)


async def test_document_summary_is_read_back(
    foreign_connection: AsyncConnection,
) -> None:
    document_id = await _prepared(foreign_connection)

    summary = await _reader(foreign_connection).document_summary(
        DocumentId(document_id)
    )

    assert summary is not None
    assert summary.status is SourceStatus.PROCESSED
    assert summary.page_count == 3


async def test_missing_document_reads_as_absent(
    foreign_connection: AsyncConnection,
) -> None:
    summary = await _reader(foreign_connection).document_summary(DocumentId.generate())

    assert summary is None


async def test_failed_document_is_not_offered_for_indexing(
    foreign_connection: AsyncConnection,
) -> None:
    # У документа со статусом `failed` пригодного текста нет: индексировать
    # нечего, и сводка обязана сказать об этом, а не притвориться успехом.
    document_id = uuid.uuid4()
    await _insert_document(foreign_connection, document_id=document_id, status="failed")

    with pytest.raises(PermanentError):
        await _reader(foreign_connection).document_summary(DocumentId(document_id))


async def test_single_chunking_version_is_selected(
    foreign_connection: AsyncConnection,
) -> None:
    document_id = await _prepared(foreign_connection)

    versions = await _reader(foreign_connection).chunking_versions(
        DocumentId(document_id)
    )

    assert versions == (ChunkingVersion(1, 0, 0),)


async def test_all_chunking_versions_are_reported(
    foreign_connection: AsyncConnection,
) -> None:
    # Версии сосуществуют как разные namespace: выбор наибольшей — дело
    # вызывающего, а прятать остальные значило бы скрывать от него факт.
    document_id = uuid.uuid4()
    page_id = uuid.uuid4()
    await _insert_document(foreign_connection, document_id=document_id)
    await _insert_page(
        foreign_connection, document_id=document_id, page_id=page_id, number=1
    )
    await _insert_chunk(
        foreign_connection,
        document_id=document_id,
        page_id=page_id,
        page_number=1,
        chunk_index=0,
        chunking_version="1.0.0",
    )
    await _insert_chunk(
        foreign_connection,
        document_id=document_id,
        page_id=page_id,
        page_number=1,
        chunk_index=0,
        chunking_version="2.0.0",
    )

    versions = await _reader(foreign_connection).chunking_versions(
        DocumentId(document_id)
    )

    assert versions == (ChunkingVersion(1, 0, 0), ChunkingVersion(2, 0, 0))


async def test_document_without_chunks_raises_permanent_error(
    foreign_connection: AsyncConnection,
) -> None:
    # Документ объявлен обработанным, а чанков нет: повтор этого не исправит.
    document_id = uuid.uuid4()
    await _insert_document(foreign_connection, document_id=document_id)

    with pytest.raises(PermanentError):
        await _reader(foreign_connection).chunking_versions(DocumentId(document_id))


async def test_chunks_arrive_in_page_and_index_order(
    foreign_connection: AsyncConnection,
) -> None:
    # Порядок — часть смысла: контекст собирается по нему, и перестановка
    # меняет то, что увидит модель.
    document_id = uuid.uuid4()
    first_page, second_page = uuid.uuid4(), uuid.uuid4()
    await _insert_document(foreign_connection, document_id=document_id)
    await _insert_page(
        foreign_connection, document_id=document_id, page_id=second_page, number=2
    )
    await _insert_page(
        foreign_connection, document_id=document_id, page_id=first_page, number=1
    )
    await _insert_chunk(
        foreign_connection,
        document_id=document_id,
        page_id=second_page,
        page_number=2,
        chunk_index=2,
    )
    await _insert_chunk(
        foreign_connection,
        document_id=document_id,
        page_id=first_page,
        page_number=1,
        chunk_index=1,
        end=20,
    )
    await _insert_chunk(
        foreign_connection,
        document_id=document_id,
        page_id=first_page,
        page_number=1,
        chunk_index=0,
        end=10,
    )

    seen = [
        (chunk.ref.page_number, chunk.span_start)
        async for chunk in _reader(foreign_connection).chunks(
            DocumentId(document_id), chunking_version=ChunkingVersion(1, 0, 0)
        )
    ]

    assert seen == [(1, 0), (1, 0), (2, 0)]


async def test_chunk_carries_everything_needed_to_index_and_cite(
    foreign_connection: AsyncConnection,
) -> None:
    document_id = await _prepared(foreign_connection)

    chunks = [
        chunk
        async for chunk in _reader(foreign_connection).chunks(
            DocumentId(document_id), chunking_version=ChunkingVersion(1, 0, 0)
        )
    ]

    chunk = chunks[0]
    assert chunk.text == CHUNK_TEXT
    assert chunk.token_count == 42
    assert chunk.quality.extraction_method is ExtractionMethod.TEXT_LAYER
    assert chunk.quality.avg_confidence is None
    assert chunk.heading_path == ("Предмет договора",)
    assert chunk.ref.document_id == DocumentId(document_id)


async def test_recognized_chunk_keeps_its_confidence(
    foreign_connection: AsyncConnection,
) -> None:
    # Без неё потребитель не отличит надёжный фрагмент от распознанного мусора.
    document_id = uuid.uuid4()
    page_id = uuid.uuid4()
    await _insert_document(foreign_connection, document_id=document_id)
    await _insert_page(
        foreign_connection, document_id=document_id, page_id=page_id, number=1
    )
    await _insert_chunk(
        foreign_connection,
        document_id=document_id,
        page_id=page_id,
        page_number=1,
        chunk_index=0,
        method="ocr",
        confidence=0.874,
    )

    chunks = [
        chunk
        async for chunk in _reader(foreign_connection).chunks(
            DocumentId(document_id), chunking_version=ChunkingVersion(1, 0, 0)
        )
    ]

    assert chunks[0].quality.avg_confidence is not None
    assert chunks[0].quality.avg_confidence.value == pytest.approx(0.874)


async def test_chunks_of_another_version_are_not_returned(
    foreign_connection: AsyncConnection,
) -> None:
    document_id = await _prepared(foreign_connection)

    chunks = [
        chunk
        async for chunk in _reader(foreign_connection).chunks(
            DocumentId(document_id), chunking_version=ChunkingVersion(2, 0, 0)
        )
    ]

    assert chunks == []


async def test_chunk_texts_are_read_for_verification(
    foreign_connection: AsyncConnection,
) -> None:
    # Цитата сверяется с живым текстом чанка, а не с тем, что вернула модель:
    # это и есть проверяемость источника.
    document_id = uuid.uuid4()
    page_id = uuid.uuid4()
    await _insert_document(foreign_connection, document_id=document_id)
    await _insert_page(
        foreign_connection, document_id=document_id, page_id=page_id, number=1
    )
    chunk_id = await _insert_chunk(
        foreign_connection,
        document_id=document_id,
        page_id=page_id,
        page_number=1,
        chunk_index=0,
    )

    texts = await _reader(foreign_connection).chunk_texts([ChunkId(chunk_id)])

    assert texts == {ChunkId(chunk_id): CHUNK_TEXT}


async def test_unknown_chunk_has_no_text(
    foreign_connection: AsyncConnection,
) -> None:
    # Исчезнувший чанк не должен ломать проверку: его отсутствие само по себе
    # означает, что цитата не подтверждается.
    texts = await _reader(foreign_connection).chunk_texts([ChunkId.generate()])

    assert texts == {}


async def test_asking_for_no_texts_touches_the_database_at_all() -> None:
    reader = SqlAlchemyProcessedChunkReader(connection=None)  # type: ignore[arg-type]

    assert await reader.chunk_texts([]) == {}


def test_no_module_outside_the_read_model_touches_document_tables() -> None:
    # Граница ADR-0001 держится машинно: второе место обращения к чужим
    # таблицам сразу видно, а не всплывает на ревью через полгода.
    offenders: list[str] = []
    for module in sorted(PACKAGE_ROOT.rglob("*.py")):
        if READ_MODEL in module.parents:
            continue
        source = module.read_text(encoding="utf-8")
        offenders.extend(
            f"{module.relative_to(SERVICE_ROOT).as_posix()}: {table}"
            for table in FOREIGN_TABLES
            if _mentions_table(source, table)
        )

    assert not offenders, "чужие таблицы упоминаются вне read-model: " + ", ".join(
        offenders
    )


def _mentions_table(source: str, table: str) -> bool:
    """Ищет имя чужой таблицы в строковых литералах модуля."""
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and table in node.value
        for node in ast.walk(ast.parse(source))
    )
