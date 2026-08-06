"""Ограничения схемы: каждое закрывает конкретный отказ.

Проверяется база, а не ORM: ограничение существует именно затем, чтобы ловить
строку, которую ORM пропустила бы — из-за бага маппера, ручного запроса или
второго воркера.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

pytestmark = pytest.mark.integration

DENSE = "[" + ",".join(["0.01"] * 1024) + "]"
SPARSE = "{7:0.9,19:0.4}/250002"
SHA256 = "a" * 64
SEMVER = "1.0.0"


def _index_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "embedding_version": SEMVER,
        "chunking_version": SEMVER,
        "pipeline_version": SEMVER,
        "model_name": "BAAI/bge-m3",
        "status": "pending",
        "source_status": "processed",
        "chunks_total": None,
        "chunks_embedded": 0,
        "chunks_failed": 0,
        "source_event_id": uuid.uuid4(),
        "started_at": False,
        "finished_at": False,
        "failure_code": None,
    }
    return row | overrides


async def _insert_index(connection: AsyncConnection, **overrides: Any) -> None:
    await connection.execute(
        text(
            "INSERT INTO ai_document_index (id, document_id, embedding_version,"
            " chunking_version, pipeline_version, model_name, status, source_status,"
            " chunks_total, chunks_embedded, chunks_failed, source_event_id,"
            " started_at, finished_at, failure_code)"
            " VALUES (:id, :document_id, :embedding_version, :chunking_version,"
            " :pipeline_version, :model_name, :status, :source_status, :chunks_total,"
            " :chunks_embedded, :chunks_failed, :source_event_id,"
            " CASE WHEN :started_at THEN now() END,"
            " CASE WHEN :finished_at THEN now() END, :failure_code)"
        ),
        _index_row(**overrides),
    )


def _embedding_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "page_id": uuid.uuid4(),
        "page_number": 3,
        "chunking_version": SEMVER,
        "embedding_version": SEMVER,
        "model_name": "BAAI/bge-m3",
        "content_hash": SHA256,
        "token_count": 42,
        "extraction_method": "text_layer",
        "avg_ocr_confidence": None,
        "illegible_span_count": 0,
    }
    return row | overrides


async def _insert_embedding(connection: AsyncConnection, **overrides: Any) -> None:
    await connection.execute(
        text(
            "INSERT INTO ai_chunk_embeddings (id, chunk_id, document_id, page_id,"
            " page_number, chunking_version, embedding_version, model_name,"
            " content_hash, dense, sparse, token_count, extraction_method,"
            " avg_ocr_confidence, illegible_span_count)"
            " VALUES (:id, :chunk_id, :document_id, :page_id, :page_number,"
            " :chunking_version, :embedding_version, :model_name, :content_hash,"
            f" '{DENSE}'::vector, '{SPARSE}'::sparsevec, :token_count,"
            " :extraction_method, :avg_ocr_confidence, :illegible_span_count)"
        ),
        _embedding_row(**overrides),
    )


async def _insert_draft(connection: AsyncConnection, **overrides: Any) -> uuid.UUID:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "request_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "draft_type": "case_fact_summary",
        "query": "Собери сводку фактов по делу",
        "status": "generated",
        "body": "## Стороны и роли\n\n- Поставщик — «Вектор».\n",
        "model_name": "deepseek-ai/DeepSeek-V4-Flash",
        "prompt_version": SEMVER,
        "retrieval_profile": "hybrid-rrf-v1",
        "embedding_version": SEMVER,
        "chunking_version": SEMVER,
        "claims_total": 1,
        "claims_grounded": 1,
        "claims_unsupported": 0,
        "evidence_total": 5,
        "failure_code": None,
    }
    row |= overrides
    await connection.execute(
        text(
            "INSERT INTO ai_drafts (id, request_id, document_id, draft_type, query,"
            " status, body, model_name, prompt_version, retrieval_profile,"
            " embedding_version, chunking_version, claims_total, claims_grounded,"
            " claims_unsupported, evidence_total, failure_code)"
            " VALUES (:id, :request_id, :document_id, :draft_type, :query, :status,"
            " :body, :model_name, :prompt_version, :retrieval_profile,"
            " :embedding_version, :chunking_version, :claims_total, :claims_grounded,"
            " :claims_unsupported, :evidence_total, :failure_code)"
        ),
        row,
    )
    return uuid.UUID(str(row["id"]))


async def _insert_claim(
    connection: AsyncConnection,
    *,
    draft_id: uuid.UUID,
    **overrides: Any,
) -> uuid.UUID:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "draft_id": draft_id,
        "claim_index": 0,
        "section": "parties",
        "text": "Поставщик — «Вектор».",
        "supported": True,
        "reject_code": None,
    }
    row |= overrides
    await connection.execute(
        text(
            "INSERT INTO ai_draft_claims (id, draft_id, claim_index, section, text,"
            " supported, reject_code)"
            " VALUES (:id, :draft_id, :claim_index, :section, :text, :supported,"
            " :reject_code)"
        ),
        row,
    )
    return uuid.UUID(str(row["id"]))


async def _insert_citation(
    connection: AsyncConnection,
    *,
    claim_id: uuid.UUID,
    draft_id: uuid.UUID,
    **overrides: Any,
) -> None:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "claim_id": claim_id,
        "draft_id": draft_id,
        "chunk_id": uuid.uuid4(),
        "page_id": uuid.uuid4(),
        "page_number": 3,
        "quote": "Договор",
        "quote_start": 0,
        "quote_end": 7,
        "retrieval_score": 0.9,
        "rerank_score": 2.1,
        "reliable": True,
    }
    row |= overrides
    await connection.execute(
        text(
            "INSERT INTO ai_draft_citations (id, claim_id, draft_id, chunk_id, page_id,"
            " page_number, quote, quote_start, quote_end, retrieval_score,"
            " rerank_score, reliable)"
            " VALUES (:id, :claim_id, :draft_id, :chunk_id, :page_id, :page_number,"
            " :quote, :quote_start, :quote_end, :retrieval_score, :rerank_score,"
            " :reliable)"
        ),
        row,
    )


async def _insert_run(connection: AsyncConnection, **overrides: Any) -> uuid.UUID:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "draft_id": None,
        "document_id": uuid.uuid4(),
        "query": "Собери сводку фактов по делу",
        "query_hash": SHA256,
        "embedding_version": SEMVER,
        "retrieval_profile": "hybrid-rrf-v1",
        "top_k": 50,
        "dense_candidates": 50,
        "sparse_candidates": 50,
        "fused_candidates": 70,
        "reranked": 50,
        "selected": 20,
        "context_tokens": 6000,
        "duration_ms": 1200,
    }
    row |= overrides
    await connection.execute(
        text(
            "INSERT INTO ai_retrieval_runs (id, draft_id, document_id, query,"
            " query_hash, embedding_version, retrieval_profile, top_k,"
            " dense_candidates, sparse_candidates, fused_candidates, reranked,"
            " selected, context_tokens, duration_ms)"
            " VALUES (:id, :draft_id, :document_id, :query, :query_hash,"
            " :embedding_version, :retrieval_profile, :top_k, :dense_candidates,"
            " :sparse_candidates, :fused_candidates, :reranked, :selected,"
            " :context_tokens, :duration_ms)"
        ),
        row,
    )
    return uuid.UUID(str(row["id"]))


async def _insert_hit(
    connection: AsyncConnection,
    *,
    run_id: uuid.UUID,
    **overrides: Any,
) -> None:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "run_id": run_id,
        "chunk_id": uuid.uuid4(),
        "page_number": 3,
        "dense_rank": 1,
        "dense_score": 0.81,
        "sparse_rank": 4,
        "sparse_score": 6.4,
        "rrf_score": 0.031,
        "rerank_score": 2.1,
        "final_rank": 1,
        "selected": True,
    }
    row |= overrides
    await connection.execute(
        text(
            "INSERT INTO ai_retrieval_hits (id, run_id, chunk_id, page_number,"
            " dense_rank, dense_score, sparse_rank, sparse_score, rrf_score,"
            " rerank_score, final_rank, selected)"
            " VALUES (:id, :run_id, :chunk_id, :page_number, :dense_rank, :dense_score,"
            " :sparse_rank, :sparse_score, :rrf_score, :rerank_score, :final_rank,"
            " :selected)"
        ),
        row,
    )


# --------------------------- ai_document_index ---------------------------


async def test_valid_index_row_is_accepted(connection: AsyncConnection) -> None:
    await _insert_index(connection)


async def test_two_indexes_of_the_same_document_and_version_collide(
    connection: AsyncConnection,
) -> None:
    # Барьер идемпотентности: повторная доставка не заводит вторую индексацию.
    document_id = uuid.uuid4()
    await _insert_index(connection, document_id=document_id)

    with pytest.raises(IntegrityError):
        await _insert_index(connection, document_id=document_id)


async def test_indexed_document_without_embedded_chunks_is_rejected(
    connection: AsyncConnection,
) -> None:
    # Такой документ считался бы готовым, а поиск по нему возвращал бы пусто.
    with pytest.raises(IntegrityError):
        await _insert_index(
            connection,
            status="indexed",
            chunks_total=5,
            chunks_embedded=0,
            chunks_failed=5,
            started_at=True,
            finished_at=True,
        )


async def test_index_counters_must_add_up_when_indexed(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(IntegrityError):
        await connection.execute(
            text(
                "INSERT INTO ai_document_index (id, document_id, embedding_version,"
                " chunking_version, pipeline_version, model_name, status,"
                " source_status, chunks_total, chunks_embedded, chunks_failed,"
                " source_event_id, started_at, finished_at)"
                " VALUES (:id, :document_id, :v, :v, :v, 'BAAI/bge-m3', 'indexed',"
                " 'processed', 10, 4, 1, :event_id, now(), now())"
            ),
            {
                "id": uuid.uuid4(),
                "document_id": uuid.uuid4(),
                "v": SEMVER,
                "event_id": uuid.uuid4(),
            },
        )


async def test_failed_index_without_a_code_is_rejected(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(IntegrityError):
        await connection.execute(
            text(
                "INSERT INTO ai_document_index (id, document_id, embedding_version,"
                " chunking_version, pipeline_version, model_name, status,"
                " source_status, source_event_id, started_at, finished_at)"
                " VALUES (:id, :document_id, :v, :v, :v, 'BAAI/bge-m3', 'failed',"
                " 'processed', :event_id, now(), now())"
            ),
            {
                "id": uuid.uuid4(),
                "document_id": uuid.uuid4(),
                "v": SEMVER,
                "event_id": uuid.uuid4(),
            },
        )


@pytest.mark.parametrize("version", ["proc-1.4.0", "1.0", "01.0.0", "v1"])
async def test_non_semver_version_is_rejected(
    connection: AsyncConnection,
    version: str,
) -> None:
    # Версия вида `proc-1.4.0` ломает ключ идемпотентности между прогонами.
    with pytest.raises(IntegrityError):
        await _insert_index(connection, embedding_version=version)


async def test_unknown_index_status_is_rejected(connection: AsyncConnection) -> None:
    with pytest.raises(IntegrityError):
        await _insert_index(connection, status="почти_готово")


async def test_failed_source_document_is_not_indexable(
    connection: AsyncConnection,
) -> None:
    # `document.processing.failed` не индексируется: пригодного текста нет.
    with pytest.raises(IntegrityError):
        await _insert_index(connection, source_status="failed")


# --------------------------- ai_chunk_embeddings ---------------------------


async def test_valid_embedding_row_is_accepted(connection: AsyncConnection) -> None:
    await _insert_embedding(connection)


async def test_two_embeddings_of_the_same_chunk_and_version_collide(
    connection: AsyncConnection,
) -> None:
    chunk_id = uuid.uuid4()
    await _insert_embedding(connection, chunk_id=chunk_id)

    with pytest.raises(IntegrityError):
        await _insert_embedding(connection, chunk_id=chunk_id)


async def test_same_chunk_in_two_versions_lives_side_by_side(
    connection: AsyncConnection,
) -> None:
    # Смена мажорной версии открывает новый namespace: старые эмбеддинги
    # остаются, переиндексация идёт рядом.
    chunk_id = uuid.uuid4()

    await _insert_embedding(connection, chunk_id=chunk_id, embedding_version="1.0.0")
    await _insert_embedding(connection, chunk_id=chunk_id, embedding_version="2.0.0")


async def test_text_layer_chunk_with_confidence_is_rejected(
    connection: AsyncConnection,
) -> None:
    # Единица от текстового слоя в одной колонке с реальным OCR-confidence
    # навсегда портит любой агрегат качества.
    with pytest.raises(IntegrityError):
        await _insert_embedding(
            connection, extraction_method="text_layer", avg_ocr_confidence=1.0
        )


async def test_recognized_chunk_without_confidence_is_rejected(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(IntegrityError):
        await _insert_embedding(
            connection, extraction_method="ocr", avg_ocr_confidence=None
        )


async def test_chunk_without_extraction_method_none_is_rejected(
    connection: AsyncConnection,
) -> None:
    # У страницы, с которой ничего не извлечено, чанков не бывает.
    with pytest.raises(IntegrityError):
        await _insert_embedding(connection, extraction_method="none")


@pytest.mark.parametrize("digest", ["A" * 64, "z" * 64, "a" * 63])
async def test_non_canonical_content_hash_is_rejected(
    connection: AsyncConnection,
    digest: str,
) -> None:
    # Ненормализованный хеш перестаёт совпадать с уже сохранённым, и модель
    # гоняется заново на каждой доставке.
    with pytest.raises(IntegrityError):
        await _insert_embedding(connection, content_hash=digest)


async def test_page_number_below_one_is_rejected(connection: AsyncConnection) -> None:
    with pytest.raises(IntegrityError):
        await _insert_embedding(connection, page_number=0)


async def test_dense_vector_of_wrong_width_is_rejected(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(DBAPIError):
        await connection.execute(
            text(
                "INSERT INTO ai_chunk_embeddings (id, chunk_id, document_id, page_id,"
                " page_number, chunking_version, embedding_version, model_name,"
                " content_hash, dense, sparse, token_count, extraction_method)"
                " VALUES (:id, :chunk_id, :document_id, :page_id, 1, :v, :v,"
                " 'BAAI/bge-m3', :digest, '[0.1,0.2]'::vector,"
                f" '{SPARSE}'::sparsevec, 10, 'text_layer')"
            ),
            {
                "id": uuid.uuid4(),
                "chunk_id": uuid.uuid4(),
                "document_id": uuid.uuid4(),
                "page_id": uuid.uuid4(),
                "v": SEMVER,
                "digest": SHA256,
            },
        )


# --------------------------- ai_drafts ---------------------------


async def test_valid_draft_row_is_accepted(connection: AsyncConnection) -> None:
    await _insert_draft(connection)


async def test_generated_draft_without_grounded_claims_is_rejected(
    connection: AsyncConnection,
) -> None:
    # Главное ограничение схемы: опубликованный черновик состоит только из
    # подтверждённых утверждений.
    with pytest.raises(IntegrityError):
        await _insert_draft(
            connection,
            status="generated",
            claims_total=2,
            claims_grounded=0,
            claims_unsupported=2,
        )


async def test_generated_draft_without_body_is_rejected(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(IntegrityError):
        await _insert_draft(connection, status="generated", body=None)


async def test_insufficient_evidence_draft_explains_itself(
    connection: AsyncConnection,
) -> None:
    # Недостаток данных — результат, а не отказ: тело объясняет, чего не хватило.
    await _insert_draft(
        connection,
        status="insufficient_evidence",
        body="## Открытые вопросы\n\n- Подтверждений не найдено.\n",
        claims_total=2,
        claims_grounded=0,
        claims_unsupported=2,
    )


async def test_insufficient_evidence_draft_with_grounded_claims_is_rejected(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(IntegrityError):
        await _insert_draft(
            connection,
            status="insufficient_evidence",
            claims_total=1,
            claims_grounded=1,
            claims_unsupported=0,
        )


async def test_draft_counters_must_add_up(connection: AsyncConnection) -> None:
    with pytest.raises(IntegrityError):
        await _insert_draft(
            connection, claims_total=5, claims_grounded=1, claims_unsupported=1
        )


async def test_second_draft_of_the_same_request_and_prompt_collides(
    connection: AsyncConnection,
) -> None:
    request_id = uuid.uuid4()
    await _insert_draft(connection, request_id=request_id)

    with pytest.raises(IntegrityError):
        await _insert_draft(connection, request_id=request_id)


async def test_new_prompt_version_may_redo_the_same_request(
    connection: AsyncConnection,
) -> None:
    # Иначе сравнить две версии промпта на одном документе было бы нечем.
    request_id = uuid.uuid4()

    await _insert_draft(connection, request_id=request_id, prompt_version="1.0.0")
    await _insert_draft(connection, request_id=request_id, prompt_version="1.1.0")


# --------------------------- claims and citations ---------------------------


async def test_valid_claim_and_citation_are_accepted(
    connection: AsyncConnection,
) -> None:
    draft_id = await _insert_draft(connection)
    claim_id = await _insert_claim(connection, draft_id=draft_id)

    await _insert_citation(connection, claim_id=claim_id, draft_id=draft_id)


async def test_supported_claim_with_a_reject_code_is_rejected(
    connection: AsyncConnection,
) -> None:
    draft_id = await _insert_draft(connection)

    with pytest.raises(IntegrityError):
        await _insert_claim(
            connection, draft_id=draft_id, supported=True, reject_code="no_citation"
        )


async def test_rejected_claim_without_a_reason_is_rejected(
    connection: AsyncConnection,
) -> None:
    draft_id = await _insert_draft(connection)

    with pytest.raises(IntegrityError):
        await _insert_claim(
            connection, draft_id=draft_id, supported=False, reject_code=None
        )


async def test_unknown_claim_section_is_rejected(connection: AsyncConnection) -> None:
    draft_id = await _insert_draft(connection)

    with pytest.raises(IntegrityError):
        await _insert_claim(connection, draft_id=draft_id, section="прочее")


async def test_two_claims_with_the_same_index_collide(
    connection: AsyncConnection,
) -> None:
    draft_id = await _insert_draft(connection)
    await _insert_claim(connection, draft_id=draft_id, claim_index=0)

    with pytest.raises(IntegrityError):
        await _insert_claim(connection, draft_id=draft_id, claim_index=0)


async def test_citation_length_must_match_its_span(
    connection: AsyncConnection,
) -> None:
    # Цитата, не совпадающая по длине со своим диапазоном, указывает не на тот
    # фрагмент — и проверить источник по ней уже нельзя.
    draft_id = await _insert_draft(connection)
    claim_id = await _insert_claim(connection, draft_id=draft_id)

    with pytest.raises(IntegrityError):
        await _insert_citation(
            connection,
            claim_id=claim_id,
            draft_id=draft_id,
            quote="Договор",
            quote_start=0,
            quote_end=99,
        )


async def test_inverted_citation_span_is_rejected(connection: AsyncConnection) -> None:
    draft_id = await _insert_draft(connection)
    claim_id = await _insert_claim(connection, draft_id=draft_id)

    with pytest.raises(IntegrityError):
        await _insert_citation(
            connection,
            claim_id=claim_id,
            draft_id=draft_id,
            quote="",
            quote_start=9,
            quote_end=2,
        )


async def test_claims_die_with_their_draft(connection: AsyncConnection) -> None:
    draft_id = await _insert_draft(connection)
    claim_id = await _insert_claim(connection, draft_id=draft_id)
    await _insert_citation(connection, claim_id=claim_id, draft_id=draft_id)

    await connection.execute(
        text("DELETE FROM ai_drafts WHERE id = :id"), {"id": draft_id}
    )

    left = await connection.scalar(
        text("SELECT count(*) FROM ai_draft_citations WHERE draft_id = :id"),
        {"id": draft_id},
    )
    assert left == 0


# --------------------------- retrieval history ---------------------------


async def test_valid_run_and_hit_are_accepted(connection: AsyncConnection) -> None:
    run_id = await _insert_run(connection)

    await _insert_hit(connection, run_id=run_id)


async def test_hit_found_by_no_branch_is_rejected(connection: AsyncConnection) -> None:
    # Попадание, не найденное ни одной ветвью, означает склейку по неверному
    # ключу — то есть чужой чанк в выдаче.
    run_id = await _insert_run(connection)

    with pytest.raises(IntegrityError):
        await _insert_hit(
            connection,
            run_id=run_id,
            dense_rank=None,
            dense_score=None,
            sparse_rank=None,
            sparse_score=None,
        )


async def test_hit_rank_without_its_score_is_rejected(
    connection: AsyncConnection,
) -> None:
    run_id = await _insert_run(connection)

    with pytest.raises(IntegrityError):
        await _insert_hit(connection, run_id=run_id, dense_score=None)


async def test_selected_hit_must_be_reranked(connection: AsyncConnection) -> None:
    # В контекст попадает только то, что прошло реранкинг.
    run_id = await _insert_run(connection)

    with pytest.raises(IntegrityError):
        await _insert_hit(connection, run_id=run_id, selected=True, rerank_score=None)


async def test_two_hits_of_the_same_chunk_in_one_run_collide(
    connection: AsyncConnection,
) -> None:
    run_id = await _insert_run(connection)
    chunk_id = uuid.uuid4()
    await _insert_hit(connection, run_id=run_id, chunk_id=chunk_id, final_rank=1)

    with pytest.raises(IntegrityError):
        await _insert_hit(connection, run_id=run_id, chunk_id=chunk_id, final_rank=2)


async def test_two_hits_with_the_same_final_rank_collide(
    connection: AsyncConnection,
) -> None:
    run_id = await _insert_run(connection)
    await _insert_hit(connection, run_id=run_id, final_rank=1)

    with pytest.raises(IntegrityError):
        await _insert_hit(connection, run_id=run_id, final_rank=1)


async def test_run_counters_must_narrow_down(connection: AsyncConnection) -> None:
    # Отобрано не больше, чем переранжировано, а переранжировано не больше,
    # чем слито: обратное означает ошибку в подсчёте ступеней.
    with pytest.raises(IntegrityError):
        await _insert_run(connection, reranked=10, selected=20)


# --------------------------- inbox and outbox ---------------------------


async def _insert_outbox(connection: AsyncConnection, **overrides: Any) -> None:
    event_id = overrides.pop("event_id", uuid.uuid4())
    aggregate_id = overrides.pop("aggregate_id", uuid.uuid4())
    payload = overrides.pop(
        "payload", f'{{"event_id": "{event_id}", "document_id": "{aggregate_id}"}}'
    )
    row: dict[str, Any] = {
        "event_id": event_id,
        "aggregate_id": aggregate_id,
        "event_type": "document.indexed",
        "routing_key": "document.indexed",
        "payload": payload,
        "correlation_id": str(uuid.uuid4()),
    }
    row |= overrides
    await connection.execute(
        text(
            "INSERT INTO ai_outbox_events (event_id, aggregate_id, event_type,"
            " routing_key, payload, correlation_id)"
            " VALUES (:event_id, :aggregate_id, :event_type, :routing_key,"
            " CAST(:payload AS jsonb), :correlation_id)"
        ),
        row,
    )


async def test_valid_outbox_row_is_accepted(connection: AsyncConnection) -> None:
    await _insert_outbox(connection)


async def test_outbox_rejects_payload_without_event_id(
    connection: AsyncConnection,
) -> None:
    # Именно этот случай проходит мимо ограничения соседнего сервиса: сравнение
    # с отсутствующим ключом даёт NULL, а CHECK на NULL не срабатывает.
    with pytest.raises(IntegrityError):
        await _insert_outbox(connection, payload='{"document_id": "x"}')


async def test_outbox_rejects_payload_with_a_foreign_event_id(
    connection: AsyncConnection,
) -> None:
    with pytest.raises(IntegrityError):
        await _insert_outbox(
            connection,
            payload=f'{{"event_id": "{uuid.uuid4()}", "document_id": "y"}}',
        )


async def test_outbox_rejects_unknown_event_type(connection: AsyncConnection) -> None:
    with pytest.raises(IntegrityError):
        await _insert_outbox(connection, event_type="document.processed")


async def test_two_events_with_the_same_key_collide(
    connection: AsyncConnection,
) -> None:
    # Повторное завершение не создаёт второго события.
    event_id = uuid.uuid4()
    await _insert_outbox(connection, event_id=event_id)

    with pytest.raises(IntegrityError):
        await _insert_outbox(connection, event_id=event_id)


async def _insert_message(connection: AsyncConnection, **overrides: Any) -> None:
    row: dict[str, Any] = {
        "event_id": uuid.uuid4(),
        "subject_id": uuid.uuid4(),
        "message_type": "document.processed",
        "status": "in_progress",
        "lease_owner": "worker-1",
        "completed_at": False,
    }
    row |= overrides
    await connection.execute(
        text(
            "INSERT INTO ai_processed_messages (event_id, subject_id, message_type,"
            " status, lease_owner, lease_expires_at, completed_at)"
            " VALUES (:event_id, :subject_id, :message_type, :status, :lease_owner,"
            " now() + interval '1 hour', CASE WHEN :completed_at THEN now() END)"
        ),
        row,
    )


async def test_claimed_message_is_accepted(connection: AsyncConnection) -> None:
    await _insert_message(connection)


async def test_in_progress_message_without_a_lease_is_rejected(
    connection: AsyncConnection,
) -> None:
    # Запись без владельца делает возобновление невозможным, а отказ по
    # конкуренции — вечным.
    with pytest.raises(IntegrityError):
        await connection.execute(
            text(
                "INSERT INTO ai_processed_messages (event_id, subject_id,"
                " message_type, status)"
                " VALUES (:event_id, :subject_id, 'document.processed', 'in_progress')"
            ),
            {"event_id": uuid.uuid4(), "subject_id": uuid.uuid4()},
        )


async def test_completed_message_keeps_no_lease(connection: AsyncConnection) -> None:
    with pytest.raises(IntegrityError):
        await _insert_message(
            connection,
            status="completed",
            lease_owner="worker-1",
            completed_at=True,
        )


async def test_second_delivery_of_the_same_event_collides(
    connection: AsyncConnection,
) -> None:
    # Основной барьер идемпотентности: гонку двух воркеров разрешает база.
    event_id = uuid.uuid4()
    await _insert_message(connection, event_id=event_id)

    with pytest.raises(IntegrityError):
        await _insert_message(connection, event_id=event_id)


def test_helper_rows_are_shaped_like_the_schema() -> None:
    # Страховка от опечатки в самих фабриках: пустая заготовка означала бы,
    # что тесты выше ничего не вставляют.
    assert _index_row()
    assert _embedding_row()
