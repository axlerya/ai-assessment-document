"""Тесты сущности страницы документа."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from document_worker.domain.entities.document_page import DocumentPage
from document_worker.domain.errors import InvariantViolation
from document_worker.domain.value_objects.confidence import OcrConfidence
from document_worker.domain.value_objects.enums import (
    ExtractionMethod,
    IllegibleReason,
    PageStatus,
)
from document_worker.domain.value_objects.identifiers import DocumentId, PageId
from document_worker.domain.value_objects.paging import PageNumber
from document_worker.domain.value_objects.quality import PageLegibilityVerdict
from document_worker.domain.value_objects.recognized_text import RecognizedText
from document_worker.domain.value_objects.storage import ObjectRef
from document_worker.domain.value_objects.text import IllegibleSpan, TextSpan
from document_worker.domain.value_objects.versioning import PipelineVersion

pytestmark = pytest.mark.unit

PAGE_ID = PageId(uuid.UUID("22222222-2222-5222-9222-222222222222"))
DOCUMENT_ID = DocumentId(uuid.UUID("11111111-1111-5111-9111-111111111111"))
NUMBER = PageNumber(14)
VERSION = PipelineVersion(1, 0, 0)
CREATED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
IMAGE_REF = ObjectRef(bucket="document-pages", key="doc/14.png")
RENDER_DPI = 300

CONTENT = "договор аренды"


def _identity() -> dict[str, object]:
    return {
        "page_id": PAGE_ID,
        "document_id": DOCUMENT_ID,
        "number": NUMBER,
        "pipeline_version": VERSION,
        "now": CREATED_AT,
    }


def _illegible(start: int, end: int) -> IllegibleSpan:
    return IllegibleSpan(
        span=TextSpan(start, end),
        confidence=OcrConfidence(0.3),
        reason=IllegibleReason.LOW_OCR_CONFIDENCE,
        raw_text=CONTENT[start:end],
    )


def _verdict(
    status: PageStatus = PageStatus.EXTRACTED,
    spans: tuple[IllegibleSpan, ...] = (),
) -> PageLegibilityVerdict:
    return PageLegibilityVerdict(
        status=status,
        mean_confidence=OcrConfidence(0.9),
        illegible_spans=spans,
        illegible_ratio=0.0 if not spans else 0.5,
        warnings=(),
    )


def _text_layer_page() -> DocumentPage:
    return DocumentPage.from_text_layer(content=CONTENT, **_identity())  # type: ignore[arg-type]


def _ocr_page(
    status: PageStatus = PageStatus.EXTRACTED,
    spans: tuple[IllegibleSpan, ...] = (),
) -> DocumentPage:
    return DocumentPage.from_recognition(
        content=CONTENT,
        method=ExtractionMethod.OCR,
        verdict=_verdict(status, spans),
        image_ref=IMAGE_REF,
        render_dpi=RENDER_DPI,
        **_identity(),  # type: ignore[arg-type]
    )


def test_page_from_text_layer_has_none_confidence_and_extracted_status() -> None:
    page = _text_layer_page()

    assert page.status is PageStatus.EXTRACTED
    assert page.method is ExtractionMethod.TEXT_LAYER
    assert page.confidence is None
    assert page.illegible_spans == ()


def test_page_from_recognition_keeps_verdict_status_and_spans() -> None:
    spans = (_illegible(0, 7),)

    page = _ocr_page(PageStatus.PARTIALLY_ILLEGIBLE, spans)

    assert page.status is PageStatus.PARTIALLY_ILLEGIBLE
    assert page.illegible_spans == spans
    assert page.confidence == OcrConfidence(0.9)


def test_hybrid_page_is_supported() -> None:
    page = DocumentPage.from_recognition(
        content=CONTENT,
        method=ExtractionMethod.HYBRID,
        verdict=_verdict(),
        image_ref=IMAGE_REF,
        render_dpi=RENDER_DPI,
        **_identity(),  # type: ignore[arg-type]
    )

    assert page.method is ExtractionMethod.HYBRID


def test_failed_page_has_empty_text_and_none_method() -> None:
    page = DocumentPage.failed(
        reason=IllegibleReason.OCR_FAILED,
        message="движок распознавания упал",
        **_identity(),  # type: ignore[arg-type]
    )

    assert page.status is PageStatus.FAILED
    assert page.method is ExtractionMethod.NONE
    assert page.char_count == 0
    assert page.warnings == ("движок распознавания упал",)


def test_page_with_extracted_status_and_illegible_spans_raises() -> None:
    with pytest.raises(InvariantViolation):
        _ocr_page(PageStatus.EXTRACTED, (_illegible(0, 7),))


def test_page_with_illegible_status_and_no_spans_raises() -> None:
    with pytest.raises(InvariantViolation):
        _ocr_page(PageStatus.ILLEGIBLE, ())


def test_page_with_partially_illegible_status_and_no_spans_raises() -> None:
    with pytest.raises(InvariantViolation):
        _ocr_page(PageStatus.PARTIALLY_ILLEGIBLE, ())


def test_unreadable_status_without_spans_raises() -> None:
    with pytest.raises(InvariantViolation):
        DocumentPage(
            id=PAGE_ID,
            document_id=DOCUMENT_ID,
            number=NUMBER,
            pipeline_version=VERSION,
            status=PageStatus.ILLEGIBLE,
            text=RecognizedText(
                content=CONTENT,
                method=ExtractionMethod.OCR,
                confidence=OcrConfidence(0.2),
            ),
            image_ref=IMAGE_REF,
            render_dpi=RENDER_DPI,
            created_at=CREATED_AT,
        )


def test_ocr_page_without_image_reference_raises() -> None:
    with pytest.raises(InvariantViolation):
        DocumentPage(
            id=PAGE_ID,
            document_id=DOCUMENT_ID,
            number=NUMBER,
            pipeline_version=VERSION,
            status=PageStatus.EXTRACTED,
            text=RecognizedText(
                content=CONTENT,
                method=ExtractionMethod.OCR,
                confidence=OcrConfidence(0.9),
            ),
            created_at=CREATED_AT,
        )


@pytest.mark.parametrize("dpi", [71, 601])
def test_ocr_page_with_absurd_render_dpi_raises(dpi: int) -> None:
    with pytest.raises(InvariantViolation):
        DocumentPage.from_recognition(
            content=CONTENT,
            method=ExtractionMethod.OCR,
            verdict=_verdict(),
            image_ref=IMAGE_REF,
            render_dpi=dpi,
            **_identity(),  # type: ignore[arg-type]
        )


def test_failed_status_requires_none_method() -> None:
    with pytest.raises(InvariantViolation):
        DocumentPage(
            id=PAGE_ID,
            document_id=DOCUMENT_ID,
            number=NUMBER,
            pipeline_version=VERSION,
            status=PageStatus.FAILED,
            text=RecognizedText(
                content=CONTENT,
                method=ExtractionMethod.OCR,
                confidence=OcrConfidence(0.9),
            ),
            image_ref=IMAGE_REF,
            render_dpi=RENDER_DPI,
            created_at=CREATED_AT,
        )


def test_page_created_at_must_be_timezone_aware() -> None:
    with pytest.raises(InvariantViolation):
        DocumentPage.from_text_layer(
            page_id=PAGE_ID,
            document_id=DOCUMENT_ID,
            number=NUMBER,
            pipeline_version=VERSION,
            content=CONTENT,
            now=datetime(2026, 7, 30, 12, 0),  # noqa: DTZ001
        )


def test_page_equality_is_by_identity_not_by_content() -> None:
    first = _text_layer_page()
    second = DocumentPage.from_text_layer(content="другой текст", **_identity())  # type: ignore[arg-type]

    assert first == second
    assert hash(first) == hash(second)


def test_pages_with_different_ids_are_not_equal() -> None:
    other_id = PageId(uuid.UUID("33333333-3333-5333-9333-333333333333"))
    first = _text_layer_page()
    second = DocumentPage.from_text_layer(
        page_id=other_id,
        document_id=DOCUMENT_ID,
        number=NUMBER,
        pipeline_version=VERSION,
        content=CONTENT,
        now=CREATED_AT,
    )

    assert first != second


def test_page_is_not_equal_to_other_types() -> None:
    assert _text_layer_page() != "страница"


def test_with_warning_returns_new_instance() -> None:
    page = _text_layer_page()

    updated = page.with_warning("низкое качество скана")

    assert page.warnings == ()
    assert updated.warnings == ("низкое качество скана",)
    assert updated is not page


def test_usable_pages_are_extracted_and_partially_illegible() -> None:
    assert _text_layer_page().is_usable
    assert _ocr_page(PageStatus.PARTIALLY_ILLEGIBLE, (_illegible(0, 7),)).is_usable
    assert not _ocr_page(PageStatus.ILLEGIBLE, (_illegible(0, 7),)).is_usable


def test_outcome_carries_page_metrics() -> None:
    page = _ocr_page(PageStatus.PARTIALLY_ILLEGIBLE, (_illegible(0, 7),))

    outcome = page.outcome()

    assert outcome.page_number == NUMBER
    assert outcome.status is PageStatus.PARTIALLY_ILLEGIBLE
    assert outcome.method is ExtractionMethod.OCR
    assert outcome.char_count == len(CONTENT)
    assert outcome.illegible_char_count == 7


def test_marked_renders_illegible_fragments() -> None:
    page = _ocr_page(PageStatus.PARTIALLY_ILLEGIBLE, (_illegible(0, 7),))

    rendered = page.marked().render()

    assert rendered.startswith("[НЕРАЗБОРЧИВО:")
    assert rendered.endswith(" аренды")
