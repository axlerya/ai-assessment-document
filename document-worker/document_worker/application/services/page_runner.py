"""Цикл по страницам документа.

Уже сохранённые страницы пропускаются, не доходя ни до чтения, ни до рендера:
возобновление обязано стоить только недостающих страниц.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.dto.commands import ProcessDocumentPageCommand

if TYPE_CHECKING:
    from collections.abc import Collection

    from document_worker.application.dto.extraction import DocumentExtraction
    from document_worker.application.use_cases.process_document_page import (
        ProcessDocumentPage,
    )
    from document_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
        JobId,
    )


@dataclass(frozen=True, slots=True)
class PageSequenceRunResult:
    """Что произошло со страницами за один прогон."""

    processed: int
    skipped: int
    failed: int
    last_number: int | None


@dataclass(frozen=True, slots=True)
class PageSequenceRunner:
    """Проходит план извлечения страница за страницей."""

    process_page: ProcessDocumentPage

    async def run(
        self,
        *,
        extraction: DocumentExtraction,
        document_id: DocumentId,
        correlation_id: CorrelationId,
        job_id: JobId,
        persisted: Collection[int],
    ) -> PageSequenceRunResult:
        """Обрабатывает недостающие страницы документа."""
        processed = 0
        skipped = 0
        failed = 0
        last: int | None = None
        for entry in extraction.plan.pages:
            number = int(entry.number)
            if number in persisted:
                skipped += 1
                continue
            result = await self.process_page.execute(
                ProcessDocumentPageCommand(
                    document_id=document_id,
                    correlation_id=correlation_id,
                    job_id=job_id,
                    entry=entry,
                    extraction=extraction,
                )
            )
            processed += 1
            failed += int(result.failure_reason is not None)
            last = number
        return PageSequenceRunResult(
            processed=processed,
            skipped=skipped,
            failed=failed,
            last_number=last,
        )
