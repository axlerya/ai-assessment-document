"""Тесты ограничений схемы на живом PostgreSQL. Только голый SQL, без ORM."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration

PIPELINE_VERSION = "1.0.0"
CHUNKING_VERSION = "1.0.0"
CHECKSUM = "a" * 64


async def _insert(connection: AsyncConnection, table: str, **values: Any) -> None:
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    statement = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"  # noqa: S608 — имена колонок собираются тестом, не вводом
    await connection.execute(text(statement), values)


async def _document(connection: AsyncConnection, **overrides: Any) -> uuid.UUID:
    document_id = overrides.pop("id", uuid.uuid4())
    values: dict[str, Any] = {
        "id": document_id,
        "bucket": "documents",
        "object_key": f"a/{document_id}.pdf",
        "declared_mime_type": "application/pdf",
        "declared_size_bytes": 1024,
    }
    values.update(overrides)
    await _insert(connection, "documents", **values)
    return document_id


async def _page(
    connection: AsyncConnection,
    document_id: uuid.UUID,
    **overrides: Any,
) -> uuid.UUID:
    page_id = overrides.pop("id", uuid.uuid4())
    text_value = overrides.pop("text", "договор аренды")
    values: dict[str, Any] = {
        "id": page_id,
        "document_id": document_id,
        "pipeline_version": PIPELINE_VERSION,
        "page_number": 1,
        "status": "extracted",
        "extraction_method": "text_layer",
        "text": text_value,
        "text_length": len(text_value),
    }
    values.update(overrides)
    await _insert(connection, "document_pages", **values)
    return page_id


async def _chunk(
    connection: AsyncConnection,
    document_id: uuid.UUID,
    page_id: uuid.UUID,
    **overrides: Any,
) -> None:
    content = overrides.pop("text", "договор")
    values: dict[str, Any] = {
        "id": uuid.uuid4(),
        "document_id": document_id,
        "page_id": page_id,
        "page_number": 1,
        "chunking_version": CHUNKING_VERSION,
        "chunk_index": 0,
        "start_offset": 0,
        "end_offset": len(content),
        "text": content,
        "token_count": 2,
        "extraction_method": "text_layer",
        "content_hash": CHECKSUM,
    }
    values.update(overrides)
    await _insert(connection, "document_chunks", **values)


async def test_duplicate_page_number_for_same_pipeline_version_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    await _page(connection, document_id)

    with pytest.raises(IntegrityError):
        await _page(connection, document_id)


async def test_same_page_number_for_other_pipeline_version_is_allowed(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    await _page(connection, document_id)

    await _page(connection, document_id, pipeline_version="2.0.0")


async def test_text_layer_page_with_confidence_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)

    with pytest.raises(IntegrityError):
        await _page(connection, document_id, ocr_confidence=1.0)


async def test_ocr_page_without_confidence_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)

    with pytest.raises(IntegrityError):
        await _page(
            connection,
            document_id,
            extraction_method="ocr",
            image_bucket="pages",
            image_key="a/1.png",
            render_dpi=300,
        )


async def test_two_chunks_with_zero_start_offset_on_different_pages_are_allowed(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    first_page = await _page(connection, document_id)
    second_page = await _page(connection, document_id, page_number=2)

    await _chunk(connection, document_id, first_page, chunk_index=0)
    await _chunk(connection, document_id, second_page, page_number=2, chunk_index=1)


async def test_chunk_text_length_mismatch_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    page_id = await _page(connection, document_id)

    with pytest.raises(IntegrityError):
        await _chunk(connection, document_id, page_id, end_offset=99)


async def test_chunk_of_another_document_page_is_rejected(
    connection: AsyncConnection,
) -> None:
    first_document = await _document(connection)
    second_document = await _document(connection)
    page_id = await _page(connection, first_document)

    with pytest.raises(IntegrityError):
        await _chunk(connection, second_document, page_id)


async def test_second_running_job_for_document_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    common = {
        "document_id": document_id,
        "status": "running",
        "trigger_event_id": uuid.uuid4(),
        "started_at": text("now()"),
    }
    await _insert(
        connection,
        "processing_jobs",
        id=uuid.uuid4(),
        pipeline_version="1.0.0",
        **{key: value for key, value in common.items() if key != "started_at"},
        started_at=None,
    )

    with pytest.raises(IntegrityError):
        await _insert(
            connection,
            "processing_jobs",
            id=uuid.uuid4(),
            pipeline_version="2.0.0",
            **{key: value for key, value in common.items() if key != "started_at"},
            started_at=None,
        )


async def test_duplicate_outbox_event_id_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    event_id = uuid.uuid4()
    payload = f'{{"event_id": "{event_id}", "document_id": "{document_id}"}}'

    await _insert(
        connection,
        "outbox_events",
        event_id=event_id,
        aggregate_id=document_id,
        event_type="document.processed",
        routing_key="document.processed",
        payload=payload,
    )

    with pytest.raises(IntegrityError):
        await _insert(
            connection,
            "outbox_events",
            event_id=event_id,
            aggregate_id=document_id,
            event_type="document.processed",
            routing_key="document.processed",
            payload=payload,
        )


async def test_outbox_event_type_outside_whitelist_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    event_id = uuid.uuid4()

    with pytest.raises(IntegrityError):
        await _insert(
            connection,
            "outbox_events",
            event_id=event_id,
            aggregate_id=document_id,
            event_type="document.invented",
            routing_key="document.invented",
            payload=f'{{"event_id": "{event_id}", "document_id": "{document_id}"}}',
        )


async def test_outbox_payload_must_carry_its_own_event_id(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)

    with pytest.raises(IntegrityError):
        await _insert(
            connection,
            "outbox_events",
            event_id=uuid.uuid4(),
            aggregate_id=document_id,
            event_type="document.processed",
            routing_key="document.processed",
            payload=f'{{"event_id": "{uuid.uuid4()}", "document_id": "{document_id}"}}',
        )


async def test_illegible_span_with_partial_bbox_is_rejected(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    page_id = await _page(
        connection, document_id, status="partially_illegible", illegible_span_count=1
    )

    with pytest.raises(IntegrityError):
        await _insert(
            connection,
            "document_illegible_spans",
            id=uuid.uuid4(),
            page_id=page_id,
            span_index=0,
            start_offset=0,
            end_offset=7,
            reason="low_ocr_confidence",
            confidence=0.3,
            raw_text="договор",
            bbox_x0=0.1,
            bbox_y0=0.1,
        )


async def test_readable_page_cannot_have_illegible_spans(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)

    with pytest.raises(IntegrityError):
        await _page(connection, document_id, illegible_span_count=1)


async def test_processed_message_unknown_status_is_rejected(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(IntegrityError):
        await _insert(
            connection,
            "processed_messages",
            event_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            pipeline_version=PIPELINE_VERSION,
            message_type="document.process.requested",
            status="in_flight",
        )


async def test_processed_message_survives_missing_document(
    connection: AsyncConnection,
) -> None:
    # Внешнего ключа нет намеренно: иначе гонка «сообщение обогнало продюсера»
    # не давала бы записать claim.
    await _insert(
        connection,
        "processed_messages",
        event_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        pipeline_version=PIPELINE_VERSION,
        message_type="document.process.requested",
        status="in_progress",
        lease_owner="worker-1",
        lease_expires_at=text("now() + interval '60 seconds'"),
    )


async def test_delete_document_cascades_pages_and_chunks_but_not_outbox(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    page_id = await _page(connection, document_id)
    await _chunk(connection, document_id, page_id)
    event_id = uuid.uuid4()
    await _insert(
        connection,
        "outbox_events",
        event_id=event_id,
        aggregate_id=document_id,
        event_type="document.processed",
        routing_key="document.processed",
        payload=f'{{"event_id": "{event_id}", "document_id": "{document_id}"}}',
    )

    await connection.execute(
        text("DELETE FROM documents WHERE id = :id"), {"id": document_id}
    )

    pages = await connection.execute(
        text("SELECT count(*) FROM document_pages WHERE document_id = :id"),
        {"id": document_id},
    )
    chunks = await connection.execute(
        text("SELECT count(*) FROM document_chunks WHERE document_id = :id"),
        {"id": document_id},
    )
    events = await connection.execute(
        text("SELECT count(*) FROM outbox_events WHERE aggregate_id = :id"),
        {"id": document_id},
    )

    assert pages.scalar() == 0
    assert chunks.scalar() == 0
    assert events.scalar() == 1
