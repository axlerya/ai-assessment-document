"""Сквозной путь устава: сообщение → документ → страницы → чанки → событие."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.domain.value_objects.enums import DocumentStatus, ExtractionMethod
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.domain.entities.document import Document
    from tests.e2e.conftest import Harness

pytestmark = pytest.mark.e2e

PAGES = 3


async def test_text_pdf_message_to_document_processed_event(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    harness.put_object(
        document.source.ref,
        pdf_builder.make_text_pdf(tmp_path / "source.pdf", pages=PAGES).read_bytes(),
    )

    await harness.request_processing(document)
    event = await harness.wait_for_event(document)

    assert event["pages_total"] == PAGES
    assert event["pages_failed"] == 0
    assert event["chunks_total"] > 0


async def test_document_row_reaches_terminal_status(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    harness.put_object(
        document.source.ref,
        pdf_builder.make_text_pdf(tmp_path / "source.pdf", pages=PAGES).read_bytes(),
    )

    await harness.request_processing(document)
    await harness.wait_for_event(document)

    rows = await harness.rows(
        "SELECT status, page_count FROM documents WHERE id = :id",
        id=document.id.value,
    )
    assert rows[0].status == DocumentStatus.PROCESSED.value
    assert rows[0].page_count == PAGES


async def test_scanned_pdf_is_processed_through_ocr(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    # Скан не имеет текстового слоя: если он доходит до чанков, значит работает
    # весь путь рендер → предобработка → распознавание.
    harness.put_object(
        document.source.ref,
        pdf_builder.make_ocr_scan_pdf(tmp_path / "scan.pdf", pages=PAGES).read_bytes(),
    )

    await harness.request_processing(document)
    event = await harness.wait_for_event(document)

    methods = await harness.rows(
        "SELECT DISTINCT extraction_method FROM document_pages WHERE document_id = :id",
        id=document.id.value,
    )
    assert [row.extraction_method for row in methods] == [ExtractionMethod.OCR.value]
    assert event["chunks_total"] > 0


async def test_chunks_keep_page_linkage_and_offsets(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    # Ключевой инвариант цитирования: текст чанка обязан быть точным срезом
    # текста своей страницы, прочитанным из базы.
    harness.put_object(
        document.source.ref,
        pdf_builder.make_text_pdf(tmp_path / "source.pdf", pages=PAGES).read_bytes(),
    )

    await harness.request_processing(document)
    await harness.wait_for_event(document)

    rows = await harness.rows(
        "SELECT c.text AS chunk_text, c.start_offset, c.end_offset,"
        " substring(p.text from c.start_offset + 1 for c.end_offset - c.start_offset)"
        " AS page_slice"
        " FROM document_chunks c JOIN document_pages p ON p.id = c.page_id"
        " WHERE c.document_id = :id",
        id=document.id.value,
    )
    assert rows
    for row in rows:
        assert row.chunk_text == row.page_slice


async def test_processed_event_is_published_exactly_once(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    harness.put_object(
        document.source.ref,
        pdf_builder.make_text_pdf(tmp_path / "source.pdf", pages=PAGES).read_bytes(),
    )

    await harness.request_processing(document)
    await harness.wait_for_event(document)

    events = await harness.rows(
        "SELECT event_type, published_at FROM outbox_events WHERE aggregate_id = :id",
        id=document.id.value,
    )
    assert len(events) == 1
    assert events[0].event_type == "document.processed"
    assert events[0].published_at is not None
