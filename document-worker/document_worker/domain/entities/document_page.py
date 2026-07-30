"""Страница документа.

Самостоятельная сущность: пишется своей транзакцией, без загрузки документа.
Идентичность по id, поведенческие методы возвращают новый экземпляр.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC
from typing import TYPE_CHECKING, Self, override

from document_worker.domain.constants import MAX_RENDER_DPI, MIN_RENDER_DPI
from document_worker.domain.errors import InvariantViolation
from document_worker.domain.markers import MarkedText
from document_worker.domain.value_objects.enums import ExtractionMethod, PageStatus
from document_worker.domain.value_objects.quality import PageFailure, PageOutcome
from document_worker.domain.value_objects.recognized_text import RecognizedText

if TYPE_CHECKING:
    from datetime import datetime

    from document_worker.domain.value_objects.confidence import OcrConfidence
    from document_worker.domain.value_objects.enums import PageFailureReason
    from document_worker.domain.value_objects.identifiers import DocumentId, PageId
    from document_worker.domain.value_objects.paging import PageNumber
    from document_worker.domain.value_objects.quality import PageLegibilityVerdict
    from document_worker.domain.value_objects.storage import ObjectRef
    from document_worker.domain.value_objects.text import IllegibleSpan
    from document_worker.domain.value_objects.versioning import PipelineVersion


@dataclass(frozen=True, slots=True, eq=False)
class DocumentPage:
    """Одна страница документа с её текстом и способом извлечения."""

    id: PageId
    document_id: DocumentId
    number: PageNumber
    pipeline_version: PipelineVersion
    status: PageStatus
    text: RecognizedText
    created_at: datetime
    image_ref: ObjectRef | None = None
    render_dpi: int | None = None
    failure: PageFailure | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Сверяет статус, способ извлечения и ссылку на рендер."""
        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() != UTC.utcoffset(None)
        ):
            raise InvariantViolation(
                "created_at обязан быть в UTC с указанием зоны",
                context={"created_at": self.created_at.isoformat()},
            )
        # Делает разбиение страниц по способам и по статусам согласованным.
        if (self.text.method is ExtractionMethod.NONE) != (
            self.status is PageStatus.FAILED
        ):
            raise InvariantViolation(
                "отсутствие способа и статус failed это одно и то же состояние",
                context={"method": self.text.method.value, "status": self.status.value},
            )
        # Ровно три допустимых сочетания статуса и числа диапазонов; у нечитаемой
        # страницы помечать нечего, её причина живёт в поле failure.
        spans = len(self.text.illegible_spans)
        allowed = (
            (self.status is PageStatus.EXTRACTED and spans == 0)
            or (
                self.status in (PageStatus.PARTIALLY_ILLEGIBLE, PageStatus.ILLEGIBLE)
                and spans >= 1
            )
            or (self.status is PageStatus.FAILED and spans == 0)
        )
        if not allowed:
            raise InvariantViolation(
                "статус страницы противоречит списку неразборчивых фрагментов",
                context={"status": self.status.value, "spans": spans},
            )
        if (self.status is PageStatus.FAILED) != (self.failure is not None):
            raise InvariantViolation(
                "отказ страницы описывается собственными полями, и только он",
                context={"status": self.status.value},
            )
        if self.text.method.is_ocr_based:
            self._validate_render()

    def _validate_render(self) -> None:
        if self.image_ref is None:
            raise InvariantViolation(
                "распознанную страницу нельзя перепроверить без её рендера",
                context={"page": int(self.number)},
            )
        if self.render_dpi is None or not (
            MIN_RENDER_DPI <= self.render_dpi <= MAX_RENDER_DPI
        ):
            raise InvariantViolation(
                f"разрешение рендера вне {MIN_RENDER_DPI}..{MAX_RENDER_DPI}",
                context={"render_dpi": self.render_dpi},
            )

    @classmethod
    def from_text_layer(  # noqa: PLR0913 — страница описывается всеми этими значениями
        cls,
        *,
        page_id: PageId,
        document_id: DocumentId,
        number: PageNumber,
        pipeline_version: PipelineVersion,
        content: str,
        now: datetime,
    ) -> Self:
        """Страница, прочитанная из текстового слоя PDF."""
        return cls(
            id=page_id,
            document_id=document_id,
            number=number,
            pipeline_version=pipeline_version,
            status=PageStatus.EXTRACTED,
            text=RecognizedText(
                content=content,
                method=ExtractionMethod.TEXT_LAYER,
                confidence=None,
            ),
            created_at=now,
        )

    @classmethod
    def from_recognition(  # noqa: PLR0913 — распознавание описывается всеми этими значениями
        cls,
        *,
        page_id: PageId,
        document_id: DocumentId,
        number: PageNumber,
        pipeline_version: PipelineVersion,
        content: str,
        method: ExtractionMethod,
        verdict: PageLegibilityVerdict,
        image_ref: ObjectRef,
        render_dpi: int,
        now: datetime,
    ) -> Self:
        """Страница, полученная распознаванием: OCR или гибрид."""
        return cls(
            id=page_id,
            document_id=document_id,
            number=number,
            pipeline_version=pipeline_version,
            status=verdict.status,
            text=RecognizedText(
                content=content,
                method=method,
                confidence=verdict.mean_confidence,
                illegible_spans=verdict.illegible_spans,
            ),
            created_at=now,
            image_ref=image_ref,
            render_dpi=render_dpi,
            warnings=verdict.warnings,
        )

    @classmethod
    def failed(  # noqa: PLR0913 — страница описывается всеми этими значениями
        cls,
        *,
        page_id: PageId,
        document_id: DocumentId,
        number: PageNumber,
        pipeline_version: PipelineVersion,
        reason: PageFailureReason,
        message: str,
        now: datetime,
        recoverable: bool = False,
    ) -> Self:
        """Страница, которую не удалось прочитать по технической причине."""
        return cls(
            id=page_id,
            document_id=document_id,
            number=number,
            pipeline_version=pipeline_version,
            status=PageStatus.FAILED,
            text=RecognizedText.not_extracted(),
            created_at=now,
            failure=PageFailure(
                reason=reason, message=message, recoverable=recoverable
            ),
        )

    @property
    def method(self) -> ExtractionMethod:
        """Способ, которым получен текст."""
        return self.text.method

    @property
    def confidence(self) -> OcrConfidence | None:
        """Уверенность распознавания, если она существует."""
        return self.text.confidence

    @property
    def illegible_spans(self) -> tuple[IllegibleSpan, ...]:
        """Неразборчивые фрагменты страницы."""
        return self.text.illegible_spans

    @property
    def is_usable(self) -> bool:
        """Годится ли страница для чанкования."""
        return self.status.is_usable

    @property
    def char_count(self) -> int:
        """Длина текста страницы."""
        return self.text.char_count

    def with_warning(self, warning: str) -> Self:
        """Возвращает копию страницы с добавленным предупреждением."""
        return replace(self, warnings=(*self.warnings, warning))

    def outcome(self) -> PageOutcome:
        """Итог страницы для агрегатов документа."""
        return PageOutcome(
            page_number=self.number,
            status=self.status,
            method=self.text.method,
            confidence=self.text.confidence,
            char_count=self.text.char_count,
            illegible_char_count=self.text.illegible_char_count,
        )

    def marked(self) -> MarkedText:
        """Текст страницы с подставленными маркерами — представление для выдачи."""
        return MarkedText(text=self.text.content, spans=self.text.illegible_spans)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentPage):
            return NotImplemented
        return self.id == other.id

    @override
    def __hash__(self) -> int:
        return hash(self.id)
