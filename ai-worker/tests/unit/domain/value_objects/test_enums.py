"""Словари домена: их значения хранятся в базе и уезжают в сообщения."""

from __future__ import annotations

from enum import StrEnum

import pytest

from ai_worker.domain.value_objects import enums

pytestmark = pytest.mark.unit

# Ожидаемые наборы записаны литералами, а не выведены из самих перечислений:
# тест обязан падать при добавлении значения, которого нет в CHECK-ограничении
# схемы, иначе он проверяет сам себя.
EXPECTED = {
    enums.IndexStatus: {"pending", "indexing", "indexed", "failed"},
    enums.SourceStatus: {"processed", "partially_processed"},
    enums.DraftStatus: {"generated", "insufficient_evidence", "failed"},
    enums.DraftType: {"case_fact_summary"},
    enums.ClaimSection: {
        "parties",
        "documents",
        "dates",
        "amounts",
        "open_questions",
    },
    enums.RejectCode: {
        "no_citation",
        "chunk_not_in_context",
        "quote_not_found",
        "unreliable_evidence_only",
    },
    enums.ExtractionMethod: {"text_layer", "ocr", "hybrid"},
}


@pytest.mark.parametrize(("enum_class", "expected"), EXPECTED.items())
def test_vocabulary_matches_the_schema(
    enum_class: type[StrEnum],
    expected: set[str],
) -> None:
    assert {member.value for member in enum_class} == expected


@pytest.mark.parametrize("enum_class", EXPECTED)
def test_every_vocabulary_is_a_string_enum(enum_class: type[StrEnum]) -> None:
    # Значение уходит в колонку varchar и в JSON сообщения как есть: обычный
    # Enum потребовал бы .value в каждой точке сериализации.
    assert issubclass(enum_class, StrEnum)


def test_chunk_without_text_has_no_extraction_method() -> None:
    # У document-worker в словаре есть `none` — страница, с которой ничего не
    # извлечено. Чанка по такой странице не существует, поэтому индексировать
    # его нечем, и значение сюда не переносится.
    assert "none" not in {member.value for member in enums.ExtractionMethod}


def test_draft_sections_follow_the_declared_order() -> None:
    # Порядок секций — часть формы черновика, и по нему собирается тело.
    assert [member.value for member in enums.ClaimSection] == [
        "parties",
        "documents",
        "dates",
        "amounts",
        "open_questions",
    ]


def test_open_questions_is_the_section_for_missing_data() -> None:
    # Раздел про недостаток данных обязателен всегда: это единственное место,
    # где сервис говорит «подтверждения нет», вместо того чтобы додумать.
    assert enums.ClaimSection.OPEN_QUESTIONS.value == "open_questions"


def test_terminal_index_statuses_are_named() -> None:
    assert enums.IndexStatus.terminal() == frozenset(
        {enums.IndexStatus.INDEXED, enums.IndexStatus.FAILED}
    )


def test_only_successful_documents_are_indexable() -> None:
    # `document.processing.failed` не индексируется вовсе: пригодного текста
    # у такого документа нет.
    assert {member.value for member in enums.SourceStatus} == {
        "processed",
        "partially_processed",
    }
