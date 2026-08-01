"""Обработка одной страницы: чтение вне транзакции, запись внутри.

Страница читается секундами, а пишется миллисекундами, поэтому транзакция
открывается последней и держит только вставку строки и счётчик прогона.

Отказ распознавания — результат обработки, а не ошибка сообщения. Повторная
доставка даст ровно тот же результат, поэтому повтор бессмыслен, а очередь
разбора вредна: документ обработан почти целиком и полезен.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from document_worker.application.dto.ocr import PageImage, PreprocessProfile
from document_worker.application.dto.results import (
    JobProgressDTO,
    ProcessDocumentPageResult,
)
from document_worker.application.errors import (
    CorruptedPageImageError,
    InvalidCommandError,
    PageLevelError,
    PageOcrTimeoutError,
    PageRenderError,
)
from document_worker.application.services.page_text import (
    PageTextAssembler,
    reproject_spans,
)
from document_worker.domain.entities.document_page import DocumentPage
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    PageFailureReason,
    PageStatus,
)
from document_worker.domain.value_objects.identifiers import PageId

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from document_worker.application.config import OcrConfig, ProcessingConfig
    from document_worker.application.dto.commands import ProcessDocumentPageCommand
    from document_worker.application.dto.ocr import OcrResult
    from document_worker.application.ports.ocr import ImagePreprocessor, OcrEngine
    from document_worker.application.ports.system import Clock, IdGenerator
    from document_worker.application.ports.unit_of_work import UnitOfWorkFactory
    from document_worker.domain.normalization.normalizer import TextNormalizer
    from document_worker.domain.policies.page_legibility import PageLegibilityPolicy
    from document_worker.domain.value_objects.quality import PageLegibilityVerdict

_FAILURE_REASONS: Final[Mapping[type[PageLevelError], PageFailureReason]] = {
    PageRenderError: PageFailureReason.RENDER_FAILED,
    PageOcrTimeoutError: PageFailureReason.TIMEOUT,
    CorruptedPageImageError: PageFailureReason.PAGE_CORRUPTED,
}


@dataclass(frozen=True, slots=True)
class _Attempt:
    """Одна попытка распознавания вместе с её вердиктом."""

    result: OcrResult
    verdict: PageLegibilityVerdict
    content: str
    dpi: int
    warnings: tuple[str, ...]

    @property
    def confidence(self) -> float:
        """Средняя уверенность попытки — по ней выбирается лучшая."""
        return self.verdict.mean_confidence.value


@dataclass(frozen=True, slots=True)
class ProcessDocumentPage:
    """Читает страницу выбранным способом и сохраняет её результат."""

    uow_factory: UnitOfWorkFactory
    normalizer: TextNormalizer
    preprocessor: ImagePreprocessor
    engine: OcrEngine
    legibility: PageLegibilityPolicy
    ids: IdGenerator
    clock: Clock
    config: ProcessingConfig
    assembler: PageTextAssembler = field(default_factory=PageTextAssembler)

    async def execute(
        self,
        command: ProcessDocumentPageCommand,
    ) -> ProcessDocumentPageResult:
        """Обрабатывает одну страницу и фиксирует её транзакцией T2ₙ."""
        now = self.clock.now()
        page = await self._read(command, now=now)
        async with self.uow_factory(statement_timeout_ms=self.config.tx.page_ms) as uow:
            persisted = await uow.pages.add(page)
            if persisted:
                # Счётчик двигается только вместе со строкой: повторная
                # доставка иначе насчитала бы страниц больше, чем в документе.
                await uow.jobs.record_progress(
                    command.job_id,
                    JobProgressDTO.for_page(page.method, at=now),
                )
            await uow.commit()
        return ProcessDocumentPageResult(
            number=page.number,
            page_id=page.id,
            status=page.status,
            method=page.method,
            confidence=page.confidence,
            char_count=page.char_count,
            failure_reason=page.failure.reason if page.failure else None,
            persisted=persisted,
        )

    async def _read(
        self,
        command: ProcessDocumentPageCommand,
        *,
        now: datetime,
    ) -> DocumentPage:
        try:
            return await self._extract(command, now=now)
        except PageLevelError as error:
            # Одна нечитаемая страница не отменяет документ: она сохраняется
            # отказом, и обработка идёт дальше.
            return self._failed(
                command,
                reason=_FAILURE_REASONS.get(
                    type(error), PageFailureReason.TEXT_EXTRACTION_FAILED
                ),
                message=error.message,
                now=now,
            )

    async def _extract(
        self,
        command: ProcessDocumentPageCommand,
        *,
        now: datetime,
    ) -> DocumentPage:
        if command.entry.method is ExtractionMethod.TEXT_LAYER:
            return await self._from_text_layer(command, now=now)
        return await self._from_recognition(command, now=now)

    async def _from_text_layer(
        self,
        command: ProcessDocumentPageCommand,
        *,
        now: datetime,
    ) -> DocumentPage:
        page = await command.extraction.pdf.read_page_text(int(command.entry.number))
        normalized = self.normalizer.normalize(
            page.text, source=ExtractionMethod.TEXT_LAYER
        )
        return DocumentPage.from_text_layer(
            page_id=PageId(self.ids.new_uuid()),
            document_id=command.document_id,
            number=command.entry.number,
            pipeline_version=self.config.pipeline_version,
            content=normalized.content,
            now=now,
        )

    async def _from_recognition(
        self,
        command: ProcessDocumentPageCommand,
        *,
        now: datetime,
    ) -> DocumentPage:
        """Распознаёт страницу, при нужде повышая разрешение."""
        best: _Attempt | None = None
        for dpi in self._dpi_ladder():
            current = await self._attempt_or_degrade(command, dpi=dpi)
            if best is None or current.confidence > best.confidence:
                best = current
            if not _needs_more_resolution(current, self.config.ocr):
                break
        return self._page_of(command, _require(best), now=now)

    def _dpi_ladder(self) -> tuple[int, ...]:
        ladder = (self.config.ocr.dpi_primary, self.config.ocr.dpi_retry)
        return ladder[: self.config.ocr.max_page_attempts]

    async def _attempt_or_degrade(
        self,
        command: ProcessDocumentPageCommand,
        *,
        dpi: int,
    ) -> _Attempt:
        """Повторяет попытку на пониженном разрешении, если первая не уложилась.

        Вторая неудача — отказ страницы: она уходит наверх ошибкой уровня
        страницы, документ при этом продолжает обрабатываться.
        """
        try:
            return await self._attempt(
                command, dpi=dpi, profile=PreprocessProfile.DEFAULT
            )
        except PageOcrTimeoutError:
            return await self._attempt(
                command,
                dpi=self.config.ocr.dpi_degraded,
                profile=PreprocessProfile.FAST,
            )

    async def _attempt(
        self,
        command: ProcessDocumentPageCommand,
        *,
        dpi: int,
        profile: PreprocessProfile,
    ) -> _Attempt:
        rendered = await command.extraction.render_session.render(
            int(command.entry.number), dpi=dpi
        )
        prepared = await self.preprocessor.prepare(
            PageImage(
                number=rendered.number,
                png=rendered.png,
                width_px=rendered.width_px,
                height_px=rendered.height_px,
                dpi=rendered.dpi,
            ),
            profile=profile,
        )
        result = await self.engine.recognize(
            prepared,
            languages=self.config.ocr.languages,
            timeout_s=self.config.ocr.page_timeout_s,
        )
        assembled = self.assembler.assemble(result.words)
        verdict = self.legibility.evaluate(
            method=command.entry.method,
            words=assembled.words,
            content=assembled.content,
        )
        return _Attempt(
            result=result,
            verdict=verdict,
            content=assembled.content,
            dpi=rendered.dpi,
            warnings=prepared.applied,
        )

    def _page_of(
        self,
        command: ProcessDocumentPageCommand,
        attempt: _Attempt,
        *,
        now: datetime,
    ) -> DocumentPage:
        """Нормализует текст и переносит диапазоны в его координаты."""
        normalized = self.normalizer.normalize(
            attempt.content, source=command.entry.method
        )
        verdict = replace(
            attempt.verdict,
            illegible_spans=reproject_spans(
                attempt.verdict.illegible_spans, normalized
            ),
            warnings=(*attempt.verdict.warnings, *attempt.warnings),
        )
        return DocumentPage.from_recognition(
            page_id=PageId(self.ids.new_uuid()),
            document_id=command.document_id,
            number=command.entry.number,
            pipeline_version=self.config.pipeline_version,
            content=normalized.content,
            method=command.entry.method,
            verdict=verdict,
            render_dpi=attempt.dpi,
            now=now,
        )

    def _failed(
        self,
        command: ProcessDocumentPageCommand,
        *,
        reason: PageFailureReason,
        message: str,
        now: datetime,
        recoverable: bool = False,
    ) -> DocumentPage:
        return DocumentPage.failed(
            page_id=PageId(self.ids.new_uuid()),
            document_id=command.document_id,
            number=command.entry.number,
            pipeline_version=self.config.pipeline_version,
            reason=reason,
            message=message,
            now=now,
            recoverable=recoverable,
        )


def _needs_more_resolution(attempt: _Attempt, config: OcrConfig) -> bool:
    """Стоит ли повторить на большем разрешении.

    Рост DPI помогает только детектору находить мелкий текст: распознаватель
    масштабирует каждый кроп строки к своей высоте, и на уже высокой строке
    прибавка разрешения не даёт ничего, кроме втрое большего времени.
    """
    if attempt.verdict.status is PageStatus.EXTRACTED:
        return False
    if attempt.confidence >= config.retry_below_confidence:
        return False
    return attempt.result.median_line_height_px < config.target_line_height_px


def _require(attempt: _Attempt | None) -> _Attempt:
    if attempt is None:  # pragma: no cover — лестница разрешений непуста
        raise InvalidCommandError("лестница разрешений пуста")
    return attempt
