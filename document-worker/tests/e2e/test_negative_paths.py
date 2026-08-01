"""Обязательные сценарии устава: повтор, отказ, неразборчивость, очередь разбора."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.domain.value_objects.enums import DocumentStatus
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

    from document_worker.domain.entities.document import Document
    from tests.e2e.conftest import Harness

pytestmark = pytest.mark.e2e

PAGES = 3


async def _process(harness: Harness, document: Document, payload: bytes) -> None:
    harness.put_object(document.source.ref, payload)
    await harness.request_processing(document)


async def test_duplicate_delivery_adds_neither_page_nor_chunk(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    # Доставка «как минимум один раз» означает, что повтор случится: он не
    # должен ни удваивать страницы, ни переписывать готовый результат.
    source = pdf_builder.make_text_pdf(tmp_path / "source.pdf", pages=PAGES)
    await _process(harness, document, source.read_bytes())
    await harness.wait_for_event(document)
    before = await _counts(harness, document)

    await harness.request_processing(document)
    await harness.settle()

    assert await _counts(harness, document) == before


async def test_duplicate_delivery_does_not_republish_the_event(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    # Повторная доставка не публикует второе событие: его уже опубликовал тот,
    # кто завершил документ.
    source = pdf_builder.make_text_pdf(tmp_path / "source.pdf", pages=PAGES)
    await _process(harness, document, source.read_bytes())
    await harness.wait_for_event(document)

    await harness.request_processing(document)
    await harness.settle()

    events = await harness.rows(
        "SELECT count(*) AS n FROM outbox_events WHERE aggregate_id = :id",
        id=document.id.value,
    )
    assert events[0].n == 1


async def test_file_that_is_not_a_pdf_goes_to_dlq_without_retry(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    # Повторять нечего: файл не станет PDF от второй попытки, и копия уходит
    # в разбор вместе с кодом отказа.
    await _process(
        harness,
        document,
        pdf_builder.make_non_pdf_file(tmp_path / "not.pdf").read_bytes(),
    )

    await harness.wait_for_dlq(document)

    rows = await harness.rows(
        "SELECT status, failure_code FROM documents WHERE id = :id",
        id=document.id.value,
    )
    assert rows[0].status == DocumentStatus.FAILED.value
    assert rows[0].failure_code


async def test_missing_object_in_storage_is_a_permanent_failure(
    harness: Harness,
    document: Document,
) -> None:
    # Файл в хранилище не появится сам: это отказ, а не временная ошибка.
    await harness.request_processing(document)

    await harness.wait_for_dlq(document)

    rows = await harness.rows(
        "SELECT status FROM documents WHERE id = :id", id=document.id.value
    )
    assert rows[0].status == DocumentStatus.FAILED.value


async def test_partially_illegible_scan_is_not_reported_as_processed(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    # Документ, часть страниц которого прочитать не удалось, полностью
    # обработанным не считается — прямое требование устава.
    await _process(
        harness,
        document,
        pdf_builder.make_partially_readable_scan_pdf(
            tmp_path / "mixed.pdf"
        ).read_bytes(),
    )

    event = await harness.wait_for_event(document)

    rows = await harness.rows(
        "SELECT status FROM documents WHERE id = :id", id=document.id.value
    )
    assert rows[0].status != DocumentStatus.PROCESSED.value
    assert event["pages_total"] == 2


async def test_illegible_page_keeps_its_span_and_is_not_reconstructed(
    harness: Harness,
    document: Document,
    tmp_path: Path,
) -> None:
    # Нераспознанный фрагмент помечается, а не досочиняется.
    await _process(
        harness,
        document,
        pdf_builder.make_partially_readable_scan_pdf(
            tmp_path / "mixed.pdf"
        ).read_bytes(),
    )
    await harness.wait_for_event(document)

    spans = await harness.rows(
        "SELECT s.reason, s.raw_text FROM document_illegible_spans s"
        " JOIN document_pages p ON p.id = s.page_id"
        " WHERE p.document_id = :id",
        id=document.id.value,
    )
    assert spans
    assert all(
        row.raw_text == "" for row in spans if row.reason == "no_text_recognized"
    )


async def _counts(harness: Harness, document: Document) -> tuple[int, int]:
    rows = await harness.rows(
        "SELECT (SELECT count(*) FROM document_pages WHERE document_id = :id) AS pages,"
        " (SELECT count(*) FROM document_chunks WHERE document_id = :id) AS chunks",
        id=document.id.value,
    )
    return (rows[0].pages, rows[0].chunks)
