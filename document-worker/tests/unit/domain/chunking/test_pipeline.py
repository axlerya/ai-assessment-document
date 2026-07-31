"""Конвейер чанкования целиком."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import pytest

from document_worker.domain.chunking.pipeline import build_pipeline
from tests.unit.domain.chunking.support import (
    FakeTokenCounter,
    default_policy,
    failed_page,
    text_layer_page,
)

if TYPE_CHECKING:
    from document_worker.domain.chunking.chunk_assembler import ChunkDraft
    from document_worker.domain.entities.document_page import DocumentPage

pytestmark = pytest.mark.unit

POLICY = default_policy()
CONTENT = "Статья 1. Предмет\nИсполнитель обязуется поставить товар в срок."


def run(*pages: DocumentPage) -> tuple[ChunkDraft, ...]:
    """Прогон конвейера на готовых страницах."""
    return build_pipeline(POLICY, FakeTokenCounter()).run(pages)


def test_pipeline_is_deterministic_for_same_input() -> None:
    page = text_layer_page(CONTENT)

    assert run(page) == run(page)


def test_pipeline_returns_empty_result_for_pages_without_text() -> None:
    # Документ без текста завершается статусом partially_processed по вердикту
    # политики, а не исключением чанкования.
    assert run(text_layer_page("   \n \n")) == ()


def test_pipeline_skips_failed_pages() -> None:
    assert run(failed_page(number=1)) == ()


def test_pipeline_returns_empty_tuple_for_empty_page_list() -> None:
    assert run() == ()


def test_pipeline_orders_pages_by_number() -> None:
    first = text_layer_page("Текст первой страницы документа.", number=1)
    second = text_layer_page("Текст второй страницы документа.", number=2)

    drafts = run(second, first)

    assert [int(draft.page_number) for draft in drafts] == [1, 2]


def test_pipeline_exposes_version_and_params_hash() -> None:
    pipeline = build_pipeline(POLICY, FakeTokenCounter())

    assert pipeline.version == POLICY.version
    assert pipeline.params_hash == POLICY.params_hash()


def test_chunking_payload_is_picklable() -> None:
    # Конвейер уезжает в пул процессов: непиклящийся аргумент обрушил бы
    # обработку уже на первом документе.
    page = text_layer_page(CONTENT)

    restored = pickle.loads(pickle.dumps((page, POLICY)))  # noqa: S301
    drafts = run(page)

    assert restored[1] == POLICY
    assert pickle.loads(pickle.dumps(drafts)) == drafts  # noqa: S301
