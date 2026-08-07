"""Чтение результатов document-worker.

Единственный модуль сервиса, знающий имена чужих таблиц (ADR-0001). Всё
остальное ходит сюда, и граница между сервисами остаётся проверяемой: обход
виден как второй адаптер, а не как ещё один запрос где-то в глубине. Гейт —
`test_no_module_outside_the_read_model_touches_document_tables`.

Запросы только читают. Право на запись сюда не выдано и на уровне базы: роль
сервиса имеет `SELECT` на эти таблицы и ничего больше.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ai_worker.application.errors import PermanentError
from ai_worker.application.ports.reading import DocumentSummary
from ai_worker.domain.entities.source_chunk import ChunkQuality, ChunkRef, SourceChunk
from ai_worker.domain.value_objects.enums import ExtractionMethod, SourceStatus
from ai_worker.domain.value_objects.identifiers import ChunkId, PageId
from ai_worker.domain.value_objects.scores import Ratio
from ai_worker.domain.value_objects.versioning import ChunkingVersion, PipelineVersion

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from sqlalchemy import Row
    from sqlalchemy.ext.asyncio import AsyncConnection

    from ai_worker.domain.value_objects.identifiers import DocumentId

# Документ читается порциями: на трёхстах страницах чанков тысячи, и держать
# их разом в памяти незачем.
STREAM_ROWS = 200

_SUMMARY = text(
    "SELECT status, pipeline_version, page_count FROM documents WHERE id = :document_id"
)

_VERSIONS = text(
    "SELECT DISTINCT chunking_version FROM document_chunks"
    " WHERE document_id = :document_id ORDER BY chunking_version"
)

_CHUNKS = text(
    "SELECT id, page_id, page_number, start_offset, text, token_count,"
    " extraction_method, avg_ocr_confidence, illegible_span_count, heading_path"
    " FROM document_chunks"
    " WHERE document_id = :document_id AND chunking_version = :chunking_version"
    " ORDER BY page_number, chunk_index"
)

_TEXTS = text("SELECT id, text FROM document_chunks WHERE id = ANY(:chunk_ids)")


@dataclass(frozen=True, slots=True)
class SqlAlchemyProcessedChunkReader:
    """Читает обработанные документы соседнего сервиса, не меняя их."""

    connection: AsyncConnection

    async def document_summary(
        self,
        document_id: DocumentId,
    ) -> DocumentSummary | None:
        """Сводка документа или `None`, если строки ещё нет.

        Raises:
            PermanentError: Документ обработан неудачно — пригодного текста у
                него нет, и повтор этого не изменит.
        """
        row = (
            await self.connection.execute(_SUMMARY, {"document_id": document_id.value})
        ).one_or_none()
        if row is None:
            return None
        try:
            status = SourceStatus(row.status)
        except ValueError as error:
            raise PermanentError(
                "документ не пригоден для индексации",
                context={"document_id": str(document_id), "status": row.status},
            ) from error
        return DocumentSummary(
            document_id=document_id,
            status=status,
            pipeline_version=PipelineVersion.parse(row.pipeline_version),
            page_count=row.page_count,
        )

    async def chunking_versions(
        self,
        document_id: DocumentId,
    ) -> tuple[ChunkingVersion, ...]:
        """Версии, под которыми у документа есть чанки.

        Возвращаются все: версии сосуществуют как разные namespace, и выбор
        между ними — дело вызывающего.

        Raises:
            PermanentError: Чанков нет вовсе. Документ объявлен обработанным,
                а индексировать нечего, и повтор этого не исправит.
        """
        rows = await self.connection.execute(
            _VERSIONS, {"document_id": document_id.value}
        )
        versions = tuple(ChunkingVersion.parse(row.chunking_version) for row in rows)
        if not versions:
            raise PermanentError(
                "у обработанного документа нет ни одного чанка",
                context={"document_id": str(document_id)},
            )
        return versions

    async def chunks(
        self,
        document_id: DocumentId,
        *,
        chunking_version: ChunkingVersion,
    ) -> AsyncIterator[SourceChunk]:
        """Поток чанков документа в порядке страниц и позиций."""
        result = await self.connection.stream(
            _CHUNKS,
            {
                "document_id": document_id.value,
                "chunking_version": str(chunking_version),
            },
            execution_options={"stream_results": True, "max_row_buffer": STREAM_ROWS},
        )
        async for row in result:
            yield _to_chunk(
                row, document_id=document_id, chunking_version=chunking_version
            )

    async def chunk_texts(
        self,
        chunk_ids: Sequence[ChunkId],
    ) -> Mapping[ChunkId, str]:
        """Тексты чанков для сверки цитат с живым источником.

        Отсутствующий чанк просто не попадает в ответ: его исчезновение само
        по себе означает, что цитата не подтверждается, и падать здесь значило
        бы превращать штатный исход верификации в отказ.
        """
        if not chunk_ids:
            return {}
        rows = await self.connection.execute(
            _TEXTS, {"chunk_ids": [chunk_id.value for chunk_id in chunk_ids]}
        )
        return {ChunkId(row.id): row.text for row in rows}


def _to_chunk(
    row: Row[Any],
    *,
    document_id: DocumentId,
    chunking_version: ChunkingVersion,
) -> SourceChunk:
    """Собирает доменный чанк из строки чужой таблицы."""
    confidence = row.avg_ocr_confidence
    return SourceChunk(
        ref=ChunkRef(
            chunk_id=ChunkId(row.id),
            document_id=document_id,
            page_id=PageId(row.page_id),
            page_number=row.page_number,
        ),
        quality=ChunkQuality(
            extraction_method=ExtractionMethod(row.extraction_method),
            avg_confidence=None if confidence is None else Ratio(float(confidence)),
            illegible_span_count=row.illegible_span_count,
        ),
        text=row.text,
        token_count=row.token_count,
        chunking_version=chunking_version,
        page_offset=row.start_offset,
        heading_path=tuple(row.heading_path),
    )
