"""Схема обязана принимать всё, что домен считает допустимым.

Каждый тест здесь — про одну колонку, которая уже, чем доменный тип за ней.
Такое расхождение не ловится ни доменными тестами, ни тестами ограничений: оба
слоя самосогласованы, и видно это только на попытке сохранить сущность.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

from document_worker.domain.value_objects.enums import (
    IllegibleReason,
    PageFailureReason,
    ProcessingStage,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration

PIPELINE_VERSION = "1.0.0"
NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

_CONSTRAINT_DEF_SQL = text(
    "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = :name"
)


async def _vocabulary_of(connection: AsyncConnection, constraint: str) -> set[str]:
    """Значения, перечисленные в CHECK ... IN (...)."""
    definition = str(await connection.scalar(_CONSTRAINT_DEF_SQL, {"name": constraint}))
    # pg_get_constraintdef печатает литералы в одинарных кавычках, поэтому
    # значения — нечётные части разбиения по кавычке.
    return set(definition.split("'")[1::2])


async def _insert(connection: AsyncConnection, table: str, **values: Any) -> None:
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    statement = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"  # noqa: S608 — имена колонок собираются тестом, не вводом
    await connection.execute(text(statement), values)


async def _document(connection: AsyncConnection, **overrides: Any) -> uuid.UUID:
    document_id: uuid.UUID = overrides.pop("id", uuid.uuid4())
    values: dict[str, Any] = {
        "id": document_id,
        "bucket": "documents",
        "object_key": f"a/{document_id}.pdf",
        "declared_mime_type": "application/pdf",
        "declared_size_bytes": 1024,
        "correlation_id": str(uuid.uuid4()),
    }
    values.update(overrides)
    await _insert(connection, "documents", **values)
    return document_id


async def test_correlation_id_column_accepts_any_domain_value(
    connection: AsyncConnection,
) -> None:
    # CorrelationId — строка из внешнего запроса, а не UUID: трассировщики
    # присылают и «trace-abc-123».
    await _document(connection, correlation_id="trace-abc-123")

    stored = await connection.scalar(text("SELECT correlation_id FROM documents"))

    assert stored == "trace-abc-123"


async def test_confidence_column_keeps_four_decimal_digits(
    connection: AsyncConnection,
) -> None:
    document_id = await _document(connection)
    await _insert(
        connection,
        "document_pages",
        id=uuid.uuid4(),
        document_id=document_id,
        pipeline_version=PIPELINE_VERSION,
        page_number=1,
        status="extracted",
        extraction_method="ocr",
        text="договор",
        text_length=7,
        ocr_confidence=Decimal("0.8123"),
        image_bucket="renders",
        image_key="a/1.png",
        render_dpi=300,
    )

    stored = await connection.scalar(text("SELECT ocr_confidence FROM document_pages"))

    assert stored == Decimal("0.8123")


async def test_illegible_reason_vocabulary_matches_domain(
    connection: AsyncConnection,
) -> None:
    allowed = await _vocabulary_of(connection, "ck__illegible_spans__reason")

    assert allowed == {reason.value for reason in IllegibleReason}


async def test_page_failure_reason_vocabulary_matches_domain(
    connection: AsyncConnection,
) -> None:
    allowed = await _vocabulary_of(connection, "ck__document_pages__failure_reason")

    assert allowed == {reason.value for reason in PageFailureReason}


async def test_job_failure_is_stored_whole(connection: AsyncConnection) -> None:
    # Код, сообщение и стадия — одна тройка: две трети в processing_jobs, а
    # третью пришлось бы доставать из другой таблицы.
    document_id = await _document(connection)
    await _insert(
        connection,
        "processing_jobs",
        id=uuid.uuid4(),
        document_id=document_id,
        pipeline_version=PIPELINE_VERSION,
        status="failed",
        trigger_event_id=uuid.uuid4(),
        correlation_id=str(uuid.uuid4()),
        failure_code="corrupted_document",
        failure_message="файл не читается",
        failure_stage=ProcessingStage.VALIDATION.value,
        finished_at=NOW,
    )

    stored = await connection.scalar(text("SELECT failure_stage FROM processing_jobs"))

    assert stored == ProcessingStage.VALIDATION.value
