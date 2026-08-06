"""Начальная схема ai-worker: индекс, эмбеддинги, черновики, история, шина.

Revision ID: 0001
Revises:
Create Date: 2026-08-06 21:52:08.304484+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Создаёт схему сервиса."""
    # Расширение ставится здесь, а не руками при развёртывании: без него
    # первая же таблица с вектором не создаётся.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('ai_chunk_embeddings',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('page_id', sa.UUID(), nullable=False),
    sa.Column('page_number', sa.Integer(), nullable=False),
    sa.Column('chunking_version', sa.String(length=32), nullable=False),
    sa.Column('embedding_version', sa.String(length=32), nullable=False),
    sa.Column('model_name', sa.String(length=128), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('dense', pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=False),
    sa.Column('sparse', pgvector.sqlalchemy.sparsevec.SPARSEVEC(dim=250002), nullable=False),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('extraction_method', sa.String(length=16), nullable=False),
    sa.Column('avg_ocr_confidence', sa.NUMERIC(precision=4, scale=3), nullable=True),
    sa.Column('illegible_span_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('heading_path', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(model_name) <> ''", name=op.f('ck__ai_chunk_embeddings__model_name_not_blank')),
    sa.CheckConstraint("chunking_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_chunk_embeddings__chunking_semver')),
    sa.CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name=op.f('ck__ai_chunk_embeddings__content_hash')),
    sa.CheckConstraint("embedding_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_chunk_embeddings__embedding_semver')),
    sa.CheckConstraint("extraction_method <> 'text_layer' OR avg_ocr_confidence IS NULL", name=op.f('ck__ai_chunk_embeddings__no_confidence_for_text_layer')),
    sa.CheckConstraint("extraction_method IN ('text_layer','ocr','hybrid')", name=op.f('ck__ai_chunk_embeddings__method')),
    sa.CheckConstraint("extraction_method NOT IN ('ocr','hybrid') OR avg_ocr_confidence IS NOT NULL", name=op.f('ck__ai_chunk_embeddings__ocr_has_confidence')),
    sa.CheckConstraint("jsonb_typeof(heading_path) = 'array'", name=op.f('ck__ai_chunk_embeddings__heading_path_is_array')),
    sa.CheckConstraint('avg_ocr_confidence IS NULL OR (avg_ocr_confidence >= 0 AND avg_ocr_confidence <= 1)', name=op.f('ck__ai_chunk_embeddings__confidence_range')),
    sa.CheckConstraint('illegible_span_count >= 0', name=op.f('ck__ai_chunk_embeddings__illegible_count')),
    sa.CheckConstraint('page_number >= 1', name=op.f('ck__ai_chunk_embeddings__page_number')),
    sa.CheckConstraint('token_count >= 1', name=op.f('ck__ai_chunk_embeddings__token_count')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_chunk_embeddings')),
    sa.UniqueConstraint('chunk_id', 'embedding_version', name='uq__ai_chunk_embeddings__chunk__ver')
    )
    op.create_index('ix__ai_chunk_embeddings__chunk', 'ai_chunk_embeddings', ['chunk_id'], unique=False)
    op.create_index('ix__ai_chunk_embeddings__dense', 'ai_chunk_embeddings', ['dense'], unique=False, postgresql_using='hnsw', postgresql_ops={'dense': 'vector_cosine_ops'})
    op.create_index('ix__ai_chunk_embeddings__doc__ver', 'ai_chunk_embeddings', ['document_id', 'embedding_version'], unique=False)
    op.create_index('ix__ai_chunk_embeddings__sparse', 'ai_chunk_embeddings', ['sparse'], unique=False, postgresql_using='hnsw', postgresql_ops={'sparse': 'sparsevec_ip_ops'})
    op.create_table('ai_document_index',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('embedding_version', sa.String(length=32), nullable=False),
    sa.Column('chunking_version', sa.String(length=32), nullable=False),
    sa.Column('pipeline_version', sa.String(length=32), nullable=False),
    sa.Column('model_name', sa.String(length=128), nullable=False),
    sa.Column('status', sa.String(length=16), server_default=sa.text("'pending'"), nullable=False),
    sa.Column('source_status', sa.String(length=24), nullable=False),
    sa.Column('chunks_total', sa.Integer(), nullable=True),
    sa.Column('chunks_embedded', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('chunks_failed', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('source_event_id', sa.UUID(), nullable=False),
    sa.Column('correlation_id', sa.UUID(), nullable=True),
    sa.Column('failure_code', sa.String(length=64), nullable=True),
    sa.Column('failure_message', sa.Text(), nullable=True),
    sa.Column('started_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('finished_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("chunking_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_document_index__chunking_semver')),
    sa.CheckConstraint("embedding_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_document_index__embedding_semver')),
    sa.CheckConstraint("pipeline_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_document_index__pipeline_semver')),
    sa.CheckConstraint("source_status IN ('processed','partially_processed')", name=op.f('ck__ai_document_index__source_status')),
    sa.CheckConstraint("status <> 'failed' OR failure_code IS NOT NULL", name=op.f('ck__ai_document_index__failed_has_code')),
    sa.CheckConstraint("status <> 'indexed' OR (chunks_total IS NOT NULL AND chunks_embedded + chunks_failed = chunks_total AND chunks_embedded > 0)", name=op.f('ck__ai_document_index__indexed_is_complete')),
    sa.CheckConstraint("status <> 'indexing' OR started_at IS NOT NULL", name=op.f('ck__ai_document_index__indexing_has_start')),
    sa.CheckConstraint("status IN ('pending','indexing','indexed','failed')", name=op.f('ck__ai_document_index__status')),
    sa.CheckConstraint("status NOT IN ('indexed','failed') OR finished_at IS NOT NULL", name=op.f('ck__ai_document_index__terminal_has_finished')),
    sa.CheckConstraint('chunks_embedded >= 0 AND chunks_failed >= 0 AND (chunks_total IS NULL OR chunks_embedded + chunks_failed <= chunks_total)', name=op.f('ck__ai_document_index__counters')),
    sa.CheckConstraint('finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at', name=op.f('ck__ai_document_index__finished_after_started')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_document_index')),
    sa.UniqueConstraint('document_id', 'embedding_version', name='uq__ai_document_index__doc__ver')
    )
    op.create_index('ix__ai_document_index__stale', 'ai_document_index', ['started_at'], unique=False, postgresql_where=sa.text("status = 'indexing'"))
    op.create_table('ai_drafts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('request_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('draft_type', sa.String(length=32), nullable=False),
    sa.Column('query', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('body', sa.Text(), nullable=True),
    sa.Column('model_name', sa.String(length=128), nullable=False),
    sa.Column('prompt_version', sa.String(length=32), nullable=False),
    sa.Column('retrieval_profile', sa.String(length=32), nullable=False),
    sa.Column('embedding_version', sa.String(length=32), nullable=False),
    sa.Column('chunking_version', sa.String(length=32), nullable=False),
    sa.Column('claims_total', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('claims_grounded', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('claims_unsupported', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('evidence_total', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('groundedness', sa.NUMERIC(precision=4, scale=3), nullable=True),
    sa.Column('input_tokens', sa.Integer(), nullable=True),
    sa.Column('output_tokens', sa.Integer(), nullable=True),
    sa.Column('duration_ms', sa.Integer(), nullable=True),
    sa.Column('failure_code', sa.String(length=64), nullable=True),
    sa.Column('failure_message', sa.Text(), nullable=True),
    sa.Column('request_payload', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('correlation_id', sa.UUID(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("btrim(query) <> ''", name=op.f('ck__ai_drafts__query_not_blank')),
    sa.CheckConstraint("chunking_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_drafts__chunking_semver')),
    sa.CheckConstraint("draft_type IN ('case_fact_summary')", name=op.f('ck__ai_drafts__type')),
    sa.CheckConstraint("embedding_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_drafts__embedding_semver')),
    sa.CheckConstraint("jsonb_typeof(request_payload) = 'object'", name=op.f('ck__ai_drafts__payload_is_object')),
    sa.CheckConstraint("prompt_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_drafts__prompt_semver')),
    sa.CheckConstraint("status <> 'failed' OR failure_code IS NOT NULL", name=op.f('ck__ai_drafts__failed_has_code')),
    sa.CheckConstraint("status <> 'generated' OR (body IS NOT NULL AND btrim(body) <> '' AND claims_grounded > 0)", name=op.f('ck__ai_drafts__generated_is_grounded')),
    sa.CheckConstraint("status <> 'insufficient_evidence' OR (body IS NOT NULL AND claims_grounded = 0)", name=op.f('ck__ai_drafts__insufficient_has_body')),
    sa.CheckConstraint("status IN ('generated','insufficient_evidence','failed')", name=op.f('ck__ai_drafts__status')),
    sa.CheckConstraint('(input_tokens IS NULL OR input_tokens >= 0) AND (output_tokens IS NULL OR output_tokens >= 0)', name=op.f('ck__ai_drafts__tokens')),
    sa.CheckConstraint('claims_total >= 0 AND claims_grounded >= 0 AND claims_unsupported >= 0 AND evidence_total >= 0 AND claims_grounded + claims_unsupported = claims_total', name=op.f('ck__ai_drafts__counters')),
    sa.CheckConstraint('duration_ms IS NULL OR duration_ms >= 0', name=op.f('ck__ai_drafts__duration')),
    sa.CheckConstraint('groundedness IS NULL OR (groundedness >= 0 AND groundedness <= 1)', name=op.f('ck__ai_drafts__groundedness_range')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_drafts')),
    sa.UniqueConstraint('request_id', 'prompt_version', name='uq__ai_drafts__request__prompt')
    )
    op.create_index('ix__ai_drafts__document', 'ai_drafts', ['document_id', sa.literal_column('created_at DESC')], unique=False)
    op.create_table('ai_outbox_events',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.Column('event_id', sa.UUID(), nullable=False),
    sa.Column('aggregate_id', sa.UUID(), nullable=False),
    sa.Column('event_type', sa.String(length=64), nullable=False),
    sa.Column('routing_key', sa.String(length=255), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('headers', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('correlation_id', sa.String(length=128), nullable=False),
    sa.Column('occurred_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('available_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('lease_owner', sa.String(length=64), nullable=True),
    sa.Column('lease_expires_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('published_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('attempts', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(routing_key) <> ''", name=op.f('ck__ai_outbox_events__routing_key_not_blank')),
    sa.CheckConstraint("event_type IN ('document.indexed','draft.generated','draft.failed')", name=op.f('ck__ai_outbox_events__event_type')),
    sa.CheckConstraint("jsonb_typeof(payload) = 'object' AND jsonb_typeof(headers) = 'object'", name=op.f('ck__ai_outbox_events__jsonb_shapes')),
    sa.CheckConstraint("payload ? 'event_id' AND payload ->> 'event_id' = event_id::text", name=op.f('ck__ai_outbox_events__payload_has_event_id')),
    sa.CheckConstraint('attempts >= 0', name=op.f('ck__ai_outbox_events__attempts')),
    sa.CheckConstraint('num_nonnulls(lease_owner, lease_expires_at) IN (0, 2)', name=op.f('ck__ai_outbox_events__lease_pair')),
    sa.CheckConstraint('published_at IS NULL OR lease_owner IS NULL', name=op.f('ck__ai_outbox_events__published_has_no_lease')),
    sa.CheckConstraint('published_at IS NULL OR published_at >= occurred_at', name=op.f('ck__ai_outbox_events__published_after_occurred')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_outbox_events')),
    sa.UniqueConstraint('event_id', name='uq__ai_outbox_events__event_id')
    )
    op.create_index('ix__ai_outbox_events__aggregate', 'ai_outbox_events', ['aggregate_id', 'occurred_at'], unique=False)
    op.create_index('ix__ai_outbox_events__published_at', 'ai_outbox_events', ['published_at'], unique=False, postgresql_where=sa.text('published_at IS NOT NULL'))
    op.create_index('ix__ai_outbox_events__unpublished', 'ai_outbox_events', ['available_at', 'id'], unique=False, postgresql_where=sa.text('published_at IS NULL'))
    op.create_table('ai_processed_messages',
    sa.Column('event_id', sa.UUID(), nullable=False),
    sa.Column('subject_id', sa.UUID(), nullable=False),
    sa.Column('message_type', sa.String(length=128), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('lease_owner', sa.String(length=64), nullable=True),
    sa.Column('lease_expires_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('attempts', sa.Integer(), server_default=sa.text('1'), nullable=False),
    sa.Column('correlation_id', sa.UUID(), nullable=True),
    sa.Column('first_seen_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("status <> 'completed' OR completed_at IS NOT NULL", name=op.f('ck__ai_processed_messages__completed_has_timestamp')),
    sa.CheckConstraint("status <> 'completed' OR lease_owner IS NULL", name=op.f('ck__ai_processed_messages__completed_has_no_lease')),
    sa.CheckConstraint("status <> 'in_progress' OR lease_owner IS NOT NULL", name=op.f('ck__ai_processed_messages__in_progress_has_lease')),
    sa.CheckConstraint("status IN ('in_progress','completed')", name=op.f('ck__ai_processed_messages__status')),
    sa.CheckConstraint('attempts >= 1', name=op.f('ck__ai_processed_messages__attempts')),
    sa.CheckConstraint('completed_at IS NULL OR completed_at >= first_seen_at', name=op.f('ck__ai_processed_messages__completed_after_seen')),
    sa.CheckConstraint('num_nonnulls(lease_owner, lease_expires_at) IN (0, 2)', name=op.f('ck__ai_processed_messages__lease_pair')),
    sa.PrimaryKeyConstraint('event_id', name=op.f('pk__ai_processed_messages'))
    )
    op.create_index('ix__ai_processed_messages__stale', 'ai_processed_messages', ['lease_expires_at'], unique=False, postgresql_where=sa.text("status = 'in_progress'"))
    op.create_index('ix__ai_processed_messages__subject', 'ai_processed_messages', ['subject_id', 'message_type'], unique=False)
    op.create_table('ai_draft_claims',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('draft_id', sa.UUID(), nullable=False),
    sa.Column('claim_index', sa.Integer(), nullable=False),
    sa.Column('section', sa.String(length=48), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('supported', sa.Boolean(), nullable=False),
    sa.Column('reject_code', sa.String(length=32), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(text) <> ''", name=op.f('ck__ai_draft_claims__text_not_blank')),
    sa.CheckConstraint("reject_code IS NULL OR reject_code IN ('no_citation','chunk_not_in_context','quote_not_found','unreliable_evidence_only')", name=op.f('ck__ai_draft_claims__reject_code')),
    sa.CheckConstraint("section IN ('parties','documents','dates','amounts','open_questions')", name=op.f('ck__ai_draft_claims__section')),
    sa.CheckConstraint('(supported AND reject_code IS NULL) OR (NOT supported AND reject_code IS NOT NULL)', name=op.f('ck__ai_draft_claims__reject_only_when_unsupported')),
    sa.CheckConstraint('claim_index >= 0', name=op.f('ck__ai_draft_claims__index')),
    sa.ForeignKeyConstraint(['draft_id'], ['ai_drafts.id'], name='fk__ai_draft_claims__draft__ai_drafts', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_draft_claims')),
    sa.UniqueConstraint('draft_id', 'claim_index', name='uq__ai_draft_claims__draft__index')
    )
    op.create_table('ai_retrieval_runs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('draft_id', sa.UUID(), nullable=True),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('query', sa.Text(), nullable=False),
    sa.Column('query_hash', sa.String(length=64), nullable=False),
    sa.Column('embedding_version', sa.String(length=32), nullable=False),
    sa.Column('retrieval_profile', sa.String(length=32), nullable=False),
    sa.Column('top_k', sa.Integer(), nullable=False),
    sa.Column('dense_candidates', sa.Integer(), nullable=False),
    sa.Column('sparse_candidates', sa.Integer(), nullable=False),
    sa.Column('fused_candidates', sa.Integer(), nullable=False),
    sa.Column('reranked', sa.Integer(), nullable=False),
    sa.Column('selected', sa.Integer(), nullable=False),
    sa.Column('context_tokens', sa.Integer(), nullable=False),
    sa.Column('duration_ms', sa.Integer(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(query) <> ''", name=op.f('ck__ai_retrieval_runs__query_not_blank')),
    sa.CheckConstraint("embedding_version ~ '^[1-9][0-9]*\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'", name=op.f('ck__ai_retrieval_runs__embedding_semver')),
    sa.CheckConstraint("query_hash ~ '^[0-9a-f]{64}$'", name=op.f('ck__ai_retrieval_runs__query_hash')),
    sa.CheckConstraint('context_tokens >= 0', name=op.f('ck__ai_retrieval_runs__context_tokens')),
    sa.CheckConstraint('duration_ms >= 0', name=op.f('ck__ai_retrieval_runs__duration')),
    sa.CheckConstraint('top_k >= 1 AND dense_candidates >= 0 AND sparse_candidates >= 0 AND fused_candidates >= 0 AND reranked >= 0 AND selected >= 0 AND selected <= reranked AND reranked <= fused_candidates', name=op.f('ck__ai_retrieval_runs__counters')),
    sa.ForeignKeyConstraint(['draft_id'], ['ai_drafts.id'], name='fk__ai_retrieval_runs__draft__ai_drafts', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_retrieval_runs'))
    )
    op.create_index('ix__ai_retrieval_runs__document', 'ai_retrieval_runs', ['document_id', sa.literal_column('created_at DESC')], unique=False)
    op.create_table('ai_draft_citations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('claim_id', sa.UUID(), nullable=False),
    sa.Column('draft_id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('page_id', sa.UUID(), nullable=False),
    sa.Column('page_number', sa.Integer(), nullable=False),
    sa.Column('quote', sa.Text(), nullable=False),
    sa.Column('quote_start', sa.Integer(), nullable=False),
    sa.Column('quote_end', sa.Integer(), nullable=False),
    sa.Column('retrieval_score', sa.REAL(), nullable=False),
    sa.Column('rerank_score', sa.REAL(), nullable=False),
    sa.Column('reliable', sa.Boolean(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(quote) <> ''", name=op.f('ck__ai_draft_citations__quote_not_blank')),
    sa.CheckConstraint('char_length(quote) = quote_end - quote_start', name=op.f('ck__ai_draft_citations__quote_length_matches')),
    sa.CheckConstraint('page_number >= 1', name=op.f('ck__ai_draft_citations__page_number')),
    sa.CheckConstraint('quote_start >= 0 AND quote_end > quote_start', name=op.f('ck__ai_draft_citations__span')),
    sa.ForeignKeyConstraint(['claim_id'], ['ai_draft_claims.id'], name='fk__ai_draft_citations__claim__ai_draft_claims', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_draft_citations')),
    sa.UniqueConstraint('claim_id', 'chunk_id', 'quote_start', name='uq__ai_draft_citations__claim__chunk__start')
    )
    op.create_index('ix__ai_draft_citations__chunk', 'ai_draft_citations', ['chunk_id'], unique=False)
    op.create_index('ix__ai_draft_citations__draft', 'ai_draft_citations', ['draft_id'], unique=False)
    op.create_table('ai_retrieval_hits',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('page_number', sa.Integer(), nullable=False),
    sa.Column('dense_rank', sa.Integer(), nullable=True),
    sa.Column('dense_score', sa.REAL(), nullable=True),
    sa.Column('sparse_rank', sa.Integer(), nullable=True),
    sa.Column('sparse_score', sa.REAL(), nullable=True),
    sa.Column('rrf_score', sa.REAL(), nullable=False),
    sa.Column('rerank_score', sa.REAL(), nullable=True),
    sa.Column('final_rank', sa.Integer(), nullable=False),
    sa.Column('selected', sa.Boolean(), nullable=False),
    sa.CheckConstraint('(dense_rank IS NULL OR dense_rank >= 1) AND (sparse_rank IS NULL OR sparse_rank >= 1)', name=op.f('ck__ai_retrieval_hits__ranks')),
    sa.CheckConstraint('NOT selected OR rerank_score IS NOT NULL', name=op.f('ck__ai_retrieval_hits__selected_was_reranked')),
    sa.CheckConstraint('dense_rank IS NOT NULL OR sparse_rank IS NOT NULL', name=op.f('ck__ai_retrieval_hits__has_source')),
    sa.CheckConstraint('final_rank >= 1', name=op.f('ck__ai_retrieval_hits__final_rank')),
    sa.CheckConstraint('num_nonnulls(dense_rank, dense_score) IN (0, 2) AND num_nonnulls(sparse_rank, sparse_score) IN (0, 2)', name=op.f('ck__ai_retrieval_hits__rank_score_pairs')),
    sa.CheckConstraint('page_number >= 1', name=op.f('ck__ai_retrieval_hits__page_number')),
    sa.ForeignKeyConstraint(['run_id'], ['ai_retrieval_runs.id'], name='fk__ai_retrieval_hits__run__ai_retrieval_runs', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk__ai_retrieval_hits')),
    sa.UniqueConstraint('run_id', 'chunk_id', name='uq__ai_retrieval_hits__run__chunk'),
    sa.UniqueConstraint('run_id', 'final_rank', name='uq__ai_retrieval_hits__run__rank')
    )
    op.create_index('ix__ai_retrieval_hits__chunk', 'ai_retrieval_hits', ['chunk_id'], unique=False)


def downgrade() -> None:
    """Снимает схему сервиса.

    Расширение `vector` остаётся: оно может быть нужно другим схемам той же
    базы, и снимать его при откате своей миграции значит ломать соседей.
    """
    op.drop_index('ix__ai_retrieval_hits__chunk', table_name='ai_retrieval_hits')
    op.drop_table('ai_retrieval_hits')
    op.drop_index('ix__ai_draft_citations__draft', table_name='ai_draft_citations')
    op.drop_index('ix__ai_draft_citations__chunk', table_name='ai_draft_citations')
    op.drop_table('ai_draft_citations')
    op.drop_index('ix__ai_retrieval_runs__document', table_name='ai_retrieval_runs')
    op.drop_table('ai_retrieval_runs')
    op.drop_table('ai_draft_claims')
    op.drop_index('ix__ai_processed_messages__subject', table_name='ai_processed_messages')
    op.drop_index('ix__ai_processed_messages__stale', table_name='ai_processed_messages', postgresql_where=sa.text("status = 'in_progress'"))
    op.drop_table('ai_processed_messages')
    op.drop_index('ix__ai_outbox_events__unpublished', table_name='ai_outbox_events', postgresql_where=sa.text('published_at IS NULL'))
    op.drop_index('ix__ai_outbox_events__published_at', table_name='ai_outbox_events', postgresql_where=sa.text('published_at IS NOT NULL'))
    op.drop_index('ix__ai_outbox_events__aggregate', table_name='ai_outbox_events')
    op.drop_table('ai_outbox_events')
    op.drop_index('ix__ai_drafts__document', table_name='ai_drafts')
    op.drop_table('ai_drafts')
    op.drop_index('ix__ai_document_index__stale', table_name='ai_document_index', postgresql_where=sa.text("status = 'indexing'"))
    op.drop_table('ai_document_index')
    op.drop_index('ix__ai_chunk_embeddings__sparse', table_name='ai_chunk_embeddings', postgresql_using='hnsw', postgresql_ops={'sparse': 'sparsevec_ip_ops'})
    op.drop_index('ix__ai_chunk_embeddings__doc__ver', table_name='ai_chunk_embeddings')
    op.drop_index('ix__ai_chunk_embeddings__dense', table_name='ai_chunk_embeddings', postgresql_using='hnsw', postgresql_ops={'dense': 'vector_cosine_ops'})
    op.drop_index('ix__ai_chunk_embeddings__chunk', table_name='ai_chunk_embeddings')
    op.drop_table('ai_chunk_embeddings')
