"""Начальная схема: документы, страницы, чанки, джобы, inbox и outbox.

Словарные значения заданы varchar с именованным CHECK, а не нативным типом:
ALTER TYPE ... ADD VALUE необратим, и downgrade стал бы неисполним.

Revision ID: 0001
Revises:
Create Date: 2026-07-30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DOCUMENTS = """
CREATE TABLE documents (
    id                     uuid          NOT NULL,
    bucket                 varchar(63)   NOT NULL,
    object_key             varchar(1024) NOT NULL,
    original_filename      varchar(512),
    declared_mime_type     varchar(255)  NOT NULL,
    detected_mime_type     varchar(255),
    declared_size_bytes    bigint        NOT NULL,
    size_bytes             bigint,
    checksum_algorithm     varchar(16)   NOT NULL DEFAULT 'sha256',
    source_checksum        varchar(64),
    checksum               varchar(64),
    page_count             integer,
    status                 varchar(24)   NOT NULL DEFAULT 'pending',
    pipeline_version       varchar(32),
    source_metadata        jsonb         NOT NULL DEFAULT '{}'::jsonb,
    failure_code           varchar(64),
    failure_stage          varchar(32),
    failure_message        text,
    correlation_id         varchar(128)  NOT NULL,
    version                integer       NOT NULL DEFAULT 0,
    created_at             timestamptz   NOT NULL DEFAULT now(),
    updated_at             timestamptz   NOT NULL DEFAULT now(),
    processing_started_at  timestamptz,
    processing_finished_at timestamptz,

    CONSTRAINT pk__documents PRIMARY KEY (id),
    CONSTRAINT uq__documents__bucket__object_key UNIQUE (bucket, object_key),

    CONSTRAINT ck__documents__status CHECK (
        status IN ('pending','processing','processed','partially_processed','failed')),
    CONSTRAINT ck__documents__object_key_not_blank CHECK (btrim(object_key) <> ''),
    CONSTRAINT ck__documents__declared_size_positive CHECK (declared_size_bytes > 0),
    CONSTRAINT ck__documents__size_positive CHECK (size_bytes IS NULL OR size_bytes > 0),
    CONSTRAINT ck__documents__page_count_positive CHECK (
        page_count IS NULL OR page_count > 0),
    CONSTRAINT ck__documents__checksum_algorithm CHECK (checksum_algorithm IN ('sha256')),
    CONSTRAINT ck__documents__checksum_hex CHECK (
        checksum IS NULL OR checksum ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck__documents__source_checksum_hex CHECK (
        source_checksum IS NULL OR source_checksum ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck__documents__pipeline_version_semver CHECK (
        pipeline_version IS NULL OR pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'),
    CONSTRAINT ck__documents__version_non_negative CHECK (version >= 0),
    CONSTRAINT ck__documents__source_metadata_is_object CHECK (
        jsonb_typeof(source_metadata) = 'object'),
    CONSTRAINT ck__documents__terminal_has_finished_at CHECK (
        status NOT IN ('processed','partially_processed','failed')
        OR processing_finished_at IS NOT NULL),
    CONSTRAINT ck__documents__failed_has_failure CHECK (
        status <> 'failed' OR (failure_code IS NOT NULL AND failure_stage IS NOT NULL)),
    CONSTRAINT ck__documents__success_is_complete CHECK (
        status NOT IN ('processed','partially_processed')
        OR (page_count IS NOT NULL AND checksum IS NOT NULL
            AND size_bytes IS NOT NULL AND pipeline_version IS NOT NULL)),
    CONSTRAINT ck__documents__finished_after_started CHECK (
        processing_finished_at IS NULL OR processing_started_at IS NULL
        OR processing_finished_at >= processing_started_at)
)
"""

_PROCESSING_JOBS = """
CREATE TABLE processing_jobs (
    id               uuid        NOT NULL,
    document_id      uuid        NOT NULL,
    pipeline_version varchar(32) NOT NULL,
    status           varchar(16) NOT NULL DEFAULT 'queued',
    attempt          integer     NOT NULL DEFAULT 1,
    trigger_event_id uuid        NOT NULL,
    correlation_id   varchar(128) NOT NULL,
    pages_total      integer,
    pages_text_layer integer     NOT NULL DEFAULT 0,
    pages_ocr        integer     NOT NULL DEFAULT 0,
    pages_hybrid     integer     NOT NULL DEFAULT 0,
    pages_failed     integer     NOT NULL DEFAULT 0,
    chunks_created   integer     NOT NULL DEFAULT 0,
    failure_code     varchar(64),
    failure_stage    varchar(32),
    failure_message  text,
    heartbeat_at     timestamptz,
    started_at       timestamptz,
    finished_at      timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT pk__processing_jobs PRIMARY KEY (id),
    CONSTRAINT fk__processing_jobs__document__documents FOREIGN KEY (document_id)
        REFERENCES documents (id) ON DELETE CASCADE,
    CONSTRAINT uq__processing_jobs__document__version UNIQUE (document_id, pipeline_version),

    CONSTRAINT ck__processing_jobs__status CHECK (
        status IN ('queued','running','succeeded','failed')),
    CONSTRAINT ck__processing_jobs__attempt CHECK (attempt >= 1),
    CONSTRAINT ck__processing_jobs__pipeline_version_semver CHECK (
        pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'),
    CONSTRAINT ck__processing_jobs__counters_non_negative CHECK (
        pages_text_layer >= 0 AND pages_ocr >= 0 AND pages_hybrid >= 0
        AND pages_failed >= 0 AND chunks_created >= 0),
    CONSTRAINT ck__processing_jobs__pages_total CHECK (
        pages_total IS NULL OR pages_total >= 0),
    CONSTRAINT ck__processing_jobs__counters_fit CHECK (
        pages_total IS NULL
        OR pages_text_layer + pages_ocr + pages_hybrid + pages_failed <= pages_total),
    CONSTRAINT ck__processing_jobs__succeeded_counters_sum CHECK (
        status <> 'succeeded'
        OR (pages_total IS NOT NULL
            AND pages_text_layer + pages_ocr + pages_hybrid + pages_failed = pages_total)),
    CONSTRAINT ck__processing_jobs__running_has_start CHECK (
        status <> 'running' OR started_at IS NOT NULL),
    CONSTRAINT ck__processing_jobs__terminal_has_finished CHECK (
        status NOT IN ('succeeded','failed') OR finished_at IS NOT NULL),
    CONSTRAINT ck__processing_jobs__failed_has_failure CHECK (
        status <> 'failed'
        OR (failure_code IS NOT NULL AND failure_stage IS NOT NULL)),
    CONSTRAINT ck__processing_jobs__finished_after_started CHECK (
        finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at)
)
"""

_DOCUMENT_PAGES = """
CREATE TABLE document_pages (
    id                   uuid          NOT NULL,
    document_id          uuid          NOT NULL,
    pipeline_version     varchar(32)   NOT NULL,
    page_number          integer       NOT NULL,
    status               varchar(24)   NOT NULL,
    extraction_method    varchar(16)   NOT NULL,
    text                 text          NOT NULL DEFAULT '',
    text_length          integer       NOT NULL DEFAULT 0,
    ocr_confidence       numeric(5,4),
    illegible_span_count integer       NOT NULL DEFAULT 0,
    image_bucket         varchar(63),
    image_key            varchar(1024),
    render_dpi           smallint,
    warnings             jsonb         NOT NULL DEFAULT '[]'::jsonb,
    failure_reason       varchar(32),
    failure_message      text,
    failure_recoverable  boolean,
    created_at           timestamptz   NOT NULL DEFAULT now(),

    CONSTRAINT pk__document_pages PRIMARY KEY (id),
    CONSTRAINT fk__document_pages__document__documents FOREIGN KEY (document_id)
        REFERENCES documents (id) ON DELETE CASCADE,
    CONSTRAINT uq__document_pages__document__version__number
        UNIQUE (document_id, pipeline_version, page_number),
    CONSTRAINT uq__document_pages__id__document__number
        UNIQUE (id, document_id, page_number),

    CONSTRAINT ck__document_pages__page_number CHECK (page_number >= 1),
    CONSTRAINT ck__document_pages__status CHECK (
        status IN ('extracted','partially_illegible','illegible','failed')),
    CONSTRAINT ck__document_pages__method CHECK (
        extraction_method IN ('text_layer','ocr','hybrid','none')),
    CONSTRAINT ck__document_pages__pipeline_version_semver CHECK (
        pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'),
    CONSTRAINT ck__document_pages__no_confidence_for_text_layer CHECK (
        extraction_method <> 'text_layer' OR ocr_confidence IS NULL),
    CONSTRAINT ck__document_pages__ocr_has_confidence CHECK (
        extraction_method NOT IN ('ocr','hybrid') OR ocr_confidence IS NOT NULL),
    CONSTRAINT ck__document_pages__none_method_has_no_confidence CHECK (
        extraction_method <> 'none' OR ocr_confidence IS NULL),
    CONSTRAINT ck__document_pages__confidence_range CHECK (
        ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 1)),
    CONSTRAINT ck__document_pages__text_length_matches CHECK (
        char_length(text) = text_length),
    CONSTRAINT ck__document_pages__span_count_non_negative CHECK (
        illegible_span_count >= 0),
    CONSTRAINT ck__document_pages__status_matches_spans CHECK (
        (status = 'extracted' AND illegible_span_count = 0)
        OR (status IN ('partially_illegible','illegible') AND illegible_span_count >= 1)
        OR (status = 'failed' AND illegible_span_count = 0)),
    CONSTRAINT ck__document_pages__failure_columns_agree CHECK (
        num_nonnulls(failure_reason, failure_message, failure_recoverable) IN (0, 3)),
    CONSTRAINT ck__document_pages__failure_only_when_failed CHECK (
        failure_reason IS NULL OR status = 'failed'),
    CONSTRAINT ck__document_pages__failed_page_is_empty CHECK (
        status <> 'failed'
        OR (extraction_method = 'none' AND text = '' AND failure_reason IS NOT NULL)),
    CONSTRAINT ck__document_pages__failure_reason CHECK (
        failure_reason IS NULL OR failure_reason IN (
            'render_failed','ocr_failed','text_extraction_failed',
            'page_corrupted','timeout')),
    CONSTRAINT ck__document_pages__image_ref_complete CHECK (
        num_nonnulls(image_bucket, image_key) IN (0, 2)),
    CONSTRAINT ck__document_pages__ocr_has_image_ref CHECK (
        extraction_method NOT IN ('ocr','hybrid')
        OR (image_key IS NOT NULL AND render_dpi IS NOT NULL)),
    CONSTRAINT ck__document_pages__render_dpi_range CHECK (
        render_dpi IS NULL OR render_dpi BETWEEN 72 AND 600),
    CONSTRAINT ck__document_pages__warnings_is_array CHECK (
        jsonb_typeof(warnings) = 'array')
)
"""

_ILLEGIBLE_SPANS = """
CREATE TABLE document_illegible_spans (
    id           uuid             NOT NULL,
    page_id      uuid             NOT NULL,
    span_index   integer          NOT NULL,
    start_offset integer          NOT NULL,
    end_offset   integer          NOT NULL,
    reason       varchar(32)      NOT NULL,
    confidence   numeric(5,4)     NOT NULL,
    raw_text     text             NOT NULL,
    line_number  integer,
    bbox_x0      double precision,
    bbox_y0      double precision,
    bbox_x1      double precision,
    bbox_y1      double precision,
    created_at   timestamptz      NOT NULL DEFAULT now(),

    CONSTRAINT pk__document_illegible_spans PRIMARY KEY (id),
    CONSTRAINT fk__illegible_spans__page__document_pages FOREIGN KEY (page_id)
        REFERENCES document_pages (id) ON DELETE CASCADE,
    CONSTRAINT uq__illegible_spans__page__index UNIQUE (page_id, span_index),
    CONSTRAINT uq__illegible_spans__page__start UNIQUE (page_id, start_offset),

    CONSTRAINT ck__illegible_spans__span_index CHECK (span_index >= 0),
    CONSTRAINT ck__illegible_spans__span_bounds CHECK (
        start_offset >= 0 AND end_offset >= start_offset),
    CONSTRAINT ck__illegible_spans__reason CHECK (reason IN (
        'low_ocr_confidence','no_text_recognized','image_too_noisy',
        'handwriting','glyph_mapping_failed')),
    CONSTRAINT ck__illegible_spans__zero_length_only_for_no_text CHECK (
        end_offset > start_offset OR reason = 'no_text_recognized'),
    CONSTRAINT ck__illegible_spans__raw_text_empty_for_no_text CHECK (
        reason <> 'no_text_recognized' OR raw_text = ''),
    CONSTRAINT ck__illegible_spans__raw_text_length_matches CHECK (
        end_offset = start_offset OR char_length(raw_text) = end_offset - start_offset),
    CONSTRAINT ck__illegible_spans__confidence_range CHECK (
        confidence >= 0 AND confidence <= 1),
    CONSTRAINT ck__illegible_spans__line_number CHECK (
        line_number IS NULL OR line_number >= 1),
    CONSTRAINT ck__illegible_spans__bbox_all CHECK (
        num_nonnulls(bbox_x0, bbox_y0, bbox_x1, bbox_y1) IN (0, 4)),
    CONSTRAINT ck__illegible_spans__bbox_normalized CHECK (
        bbox_x0 IS NULL
        OR (bbox_x0 >= 0 AND bbox_y0 >= 0 AND bbox_x1 <= 1 AND bbox_y1 <= 1
            AND bbox_x1 >= bbox_x0 AND bbox_y1 >= bbox_y0))
)
"""

_DOCUMENT_CHUNKS = """
CREATE TABLE document_chunks (
    id                   uuid        NOT NULL,
    document_id          uuid        NOT NULL,
    page_id              uuid        NOT NULL,
    page_number          integer     NOT NULL,
    chunking_version     varchar(32) NOT NULL,
    chunk_index          integer     NOT NULL,
    start_offset         integer     NOT NULL,
    end_offset           integer     NOT NULL,
    text                 text        NOT NULL,
    token_count          integer     NOT NULL,
    overlap_prefix_chars integer     NOT NULL DEFAULT 0,
    extraction_method    varchar(16) NOT NULL,
    avg_ocr_confidence   numeric(5,4),
    illegible_span_count integer     NOT NULL DEFAULT 0,
    heading_path         jsonb       NOT NULL DEFAULT '[]'::jsonb,
    content_hash         varchar(64) NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT pk__document_chunks PRIMARY KEY (id),
    CONSTRAINT fk__document_chunks__document__documents FOREIGN KEY (document_id)
        REFERENCES documents (id) ON DELETE CASCADE,
    CONSTRAINT fk__document_chunks__page__document_pages
        FOREIGN KEY (page_id, document_id, page_number)
        REFERENCES document_pages (id, document_id, page_number) ON DELETE CASCADE,
    CONSTRAINT uq__document_chunks__page__start
        UNIQUE (document_id, chunking_version, page_id, start_offset),
    CONSTRAINT uq__document_chunks__document__version__index
        UNIQUE (document_id, chunking_version, chunk_index),

    CONSTRAINT ck__document_chunks__index CHECK (chunk_index >= 0),
    CONSTRAINT ck__document_chunks__page_number CHECK (page_number >= 1),
    CONSTRAINT ck__document_chunks__span CHECK (
        start_offset >= 0 AND end_offset > start_offset),
    CONSTRAINT ck__document_chunks__text_len_matches_span CHECK (
        char_length(text) = end_offset - start_offset),
    CONSTRAINT ck__document_chunks__text_not_blank CHECK (btrim(text) <> ''),
    CONSTRAINT ck__document_chunks__method CHECK (
        extraction_method IN ('text_layer','ocr','hybrid')),
    CONSTRAINT ck__document_chunks__chunking_version_semver CHECK (
        chunking_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'),
    CONSTRAINT ck__document_chunks__no_confidence_for_text_layer CHECK (
        extraction_method <> 'text_layer' OR avg_ocr_confidence IS NULL),
    CONSTRAINT ck__document_chunks__ocr_has_confidence CHECK (
        extraction_method NOT IN ('ocr','hybrid') OR avg_ocr_confidence IS NOT NULL),
    CONSTRAINT ck__document_chunks__confidence_range CHECK (
        avg_ocr_confidence IS NULL
        OR (avg_ocr_confidence >= 0 AND avg_ocr_confidence <= 1)),
    CONSTRAINT ck__document_chunks__overlap CHECK (
        overlap_prefix_chars >= 0 AND overlap_prefix_chars < end_offset - start_offset),
    CONSTRAINT ck__document_chunks__token_count CHECK (token_count >= 1),
    CONSTRAINT ck__document_chunks__illegible_count CHECK (illegible_span_count >= 0),
    CONSTRAINT ck__document_chunks__content_hash CHECK (
        content_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck__document_chunks__heading_path_is_array CHECK (
        jsonb_typeof(heading_path) = 'array')
)
"""

_PROCESSED_MESSAGES = """
CREATE TABLE processed_messages (
    event_id         uuid         NOT NULL,
    document_id      uuid         NOT NULL,
    pipeline_version varchar(32)  NOT NULL,
    message_type     varchar(128) NOT NULL,
    status           varchar(16)  NOT NULL,
    lease_owner      varchar(64),
    lease_expires_at timestamptz,
    attempts         integer      NOT NULL DEFAULT 1,
    correlation_id   varchar(128) NOT NULL,
    first_seen_at    timestamptz  NOT NULL DEFAULT now(),
    updated_at       timestamptz  NOT NULL DEFAULT now(),
    completed_at     timestamptz,

    CONSTRAINT pk__processed_messages PRIMARY KEY (event_id),

    CONSTRAINT ck__processed_messages__status CHECK (
        status IN ('in_progress','completed')),
    CONSTRAINT ck__processed_messages__pipeline_version_semver CHECK (
        pipeline_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'),
    CONSTRAINT ck__processed_messages__lease_pair CHECK (
        num_nonnulls(lease_owner, lease_expires_at) IN (0, 2)),
    CONSTRAINT ck__processed_messages__in_progress_has_lease CHECK (
        status <> 'in_progress' OR lease_owner IS NOT NULL),
    CONSTRAINT ck__processed_messages__completed_has_no_lease CHECK (
        status <> 'completed' OR lease_owner IS NULL),
    CONSTRAINT ck__processed_messages__completed_has_timestamp CHECK (
        status <> 'completed' OR completed_at IS NOT NULL),
    CONSTRAINT ck__processed_messages__attempts CHECK (attempts >= 1),
    CONSTRAINT ck__processed_messages__completed_after_seen CHECK (
        completed_at IS NULL OR completed_at >= first_seen_at)
)
"""

_OUTBOX_EVENTS = """
CREATE TABLE outbox_events (
    id               bigint       GENERATED ALWAYS AS IDENTITY,
    event_id         uuid         NOT NULL,
    aggregate_type   varchar(32)  NOT NULL DEFAULT 'document',
    aggregate_id     uuid         NOT NULL,
    event_type       varchar(64)  NOT NULL,
    routing_key      varchar(255) NOT NULL,
    payload          jsonb        NOT NULL,
    headers          jsonb        NOT NULL DEFAULT '{}'::jsonb,
    correlation_id   varchar(128) NOT NULL,
    occurred_at      timestamptz  NOT NULL DEFAULT now(),
    available_at     timestamptz  NOT NULL DEFAULT now(),
    lease_owner      varchar(64),
    lease_expires_at timestamptz,
    published_at     timestamptz,
    attempts         integer      NOT NULL DEFAULT 0,
    last_error       text,
    created_at       timestamptz  NOT NULL DEFAULT now(),

    CONSTRAINT pk__outbox_events PRIMARY KEY (id),
    CONSTRAINT uq__outbox_events__event_id UNIQUE (event_id),

    CONSTRAINT ck__outbox_events__event_type CHECK (event_type IN (
        'document.processed','document.partially_processed','document.processing.failed')),
    CONSTRAINT ck__outbox_events__routing_key_not_blank CHECK (btrim(routing_key) <> ''),
    CONSTRAINT ck__outbox_events__jsonb_shapes CHECK (
        jsonb_typeof(payload) = 'object' AND jsonb_typeof(headers) = 'object'),
    CONSTRAINT ck__outbox_events__payload_has_event_id CHECK (
        payload ->> 'event_id' = event_id::text),
    CONSTRAINT ck__outbox_events__payload_has_document_id CHECK (
        payload ->> 'document_id' = aggregate_id::text),
    CONSTRAINT ck__outbox_events__attempts CHECK (attempts >= 0),
    CONSTRAINT ck__outbox_events__lease_pair CHECK (
        num_nonnulls(lease_owner, lease_expires_at) IN (0, 2)),
    CONSTRAINT ck__outbox_events__published_has_no_lease CHECK (
        published_at IS NULL OR lease_owner IS NULL),
    CONSTRAINT ck__outbox_events__published_after_occurred CHECK (
        published_at IS NULL OR published_at >= occurred_at)
)
"""

_INDEXES = (
    "CREATE INDEX ix__documents__stale_processing ON documents (processing_started_at)"
    " WHERE status = 'processing'",
    "CREATE INDEX ix__documents__correlation_id ON documents (correlation_id)",
    "CREATE UNIQUE INDEX uq__processing_jobs__active ON processing_jobs (document_id)"
    " WHERE status = 'running'",
    "CREATE INDEX ix__processing_jobs__stale ON processing_jobs (heartbeat_at)"
    " WHERE status = 'running'",
    "CREATE INDEX ix__document_pages__resume ON document_pages"
    " (document_id, pipeline_version) INCLUDE (page_number, status)",
    "CREATE INDEX ix__document_pages__illegible ON document_pages"
    " (document_id, pipeline_version) WHERE illegible_span_count > 0",
    "CREATE INDEX ix__document_chunks__page ON document_chunks"
    " (page_id, document_id, page_number)",
    "CREATE INDEX ix__document_chunks__doc_version_page ON document_chunks"
    " (document_id, chunking_version, page_number, chunk_index)",
    "CREATE INDEX ix__document_chunks__illegible ON document_chunks"
    " (document_id, chunking_version) WHERE illegible_span_count > 0",
    "CREATE INDEX ix__processed_messages__stale ON processed_messages"
    " (lease_expires_at) WHERE status = 'in_progress'",
    "CREATE INDEX ix__processed_messages__document ON processed_messages"
    " (document_id, pipeline_version)",
    "CREATE INDEX ix__processed_messages__completed_at ON processed_messages"
    " (completed_at) WHERE status = 'completed'",
    "CREATE INDEX ix__outbox_events__unpublished ON outbox_events (available_at, id)"
    " WHERE published_at IS NULL",
    "CREATE INDEX ix__outbox_events__published_at ON outbox_events (published_at)"
    " WHERE published_at IS NOT NULL",
    "CREATE INDEX ix__outbox_events__aggregate ON outbox_events"
    " (aggregate_id, occurred_at)",
)

# Порядок важен: внешние ключи ссылаются на созданные раньше таблицы.
_TABLES = (
    ("documents", _DOCUMENTS),
    ("processing_jobs", _PROCESSING_JOBS),
    ("document_pages", _DOCUMENT_PAGES),
    ("document_illegible_spans", _ILLEGIBLE_SPANS),
    ("document_chunks", _DOCUMENT_CHUNKS),
    ("processed_messages", _PROCESSED_MESSAGES),
    ("outbox_events", _OUTBOX_EVENTS),
)


def upgrade() -> None:
    for _, statement in _TABLES:
        op.execute(statement)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    for name, _ in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
