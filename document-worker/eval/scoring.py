"""Сведение эталона и результата в числа.

Смещения сравниваются в нормализованных координатах: эталон и текст сервиса
расходятся переносами строк, и без приведения допуск границ съедался бы этой
разницей, а не настоящей ошибкой сборки.

Локализация неразборчивого меряется на уровне страницы, а не диапазона.
Диапазон нераспознанной вставки в тексте сервиса имеет нулевую длину — текста
там нет по построению, — и IoU по нему считал бы совпадение двух пустот.
Поэтому честность сервиса меряют три доли: пометил, не выдумал, не пометил
лишнего. IoU остаётся там, где обе стороны известны точно, — на покрытии
страницы чанками.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING

from eval.metrics import (
    boundary_f1,
    cer,
    normalize_for_scoring,
    span_iou,
    spearman,
    wer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from eval.corpus import DocumentTruth, PageTruth
    from eval.runner import ChunkOutcome, DocumentOutcome, PageOutcome

GROUND_TRUTH_DIR = "ground_truth"


@dataclass(frozen=True, slots=True)
class PageScore:
    """Числа по одной странице."""

    doc_id: str
    category: str
    number: int
    cer: float
    wer: float
    expected_method: str
    actual_method: str
    expected_status: str
    actual_status: str
    mean_confidence: float | None
    expects_unreadable: bool
    marked_illegible: bool
    chunk_coverage: float
    boundary_f1: float

    @property
    def method_matches(self) -> bool:
        """Совпал ли способ извлечения с ожидаемым."""
        return self.expected_method == self.actual_method


def score_document(
    truth: DocumentTruth,
    outcome: DocumentOutcome,
    *,
    corpus_root: Path,
) -> tuple[PageScore, ...]:
    """Считает числа по каждой странице документа."""
    truth_dir = corpus_root / truth.doc_id / GROUND_TRUTH_DIR
    by_number = {page.number: page for page in outcome.pages}
    scored: list[PageScore] = []
    for expected in truth.pages:
        actual = by_number.get(expected.number)
        reference = (truth_dir / f"page_{expected.number:04d}.txt").read_text(
            encoding="utf-8"
        )
        chunks = [
            chunk for chunk in outcome.chunks if chunk.page_number == expected.number
        ]
        scored.append(
            _score_page(
                truth=truth,
                expected=expected,
                actual=actual,
                reference=reference,
                chunks=chunks,
            )
        )
    return tuple(scored)


def _score_page(
    *,
    truth: DocumentTruth,
    expected: PageTruth,
    actual: PageOutcome | None,
    reference: str,
    chunks: Sequence[ChunkOutcome],
) -> PageScore:
    expects_unreadable = bool(expected.unreadable_text)
    hypothesis = actual.text if actual is not None else ""
    return PageScore(
        doc_id=truth.doc_id,
        category=truth.category,
        number=expected.number,
        cer=cer(reference, hypothesis),
        wer=wer(reference, hypothesis),
        expected_method=expected.expected_extraction_method,
        actual_method=actual.extraction_method if actual is not None else "missing",
        expected_status=expected.expected_page_status,
        actual_status=actual.status if actual is not None else "missing",
        mean_confidence=actual.mean_confidence if actual is not None else None,
        expects_unreadable=expects_unreadable,
        marked_illegible=bool(actual and actual.illegible_spans),
        chunk_coverage=_coverage(hypothesis, chunks),
        boundary_f1=_boundaries(
            reference,
            hypothesis,
            expected.section_boundaries,
            chunks,
        ),
    )


def _coverage(page_text: str, chunks: Sequence[ChunkOutcome]) -> float:
    """Какая часть страницы попала в чанки."""
    if not page_text:
        return 1.0 if not chunks else 0.0
    spans = [(chunk.start_offset, chunk.end_offset) for chunk in chunks]
    return span_iou([(0, len(page_text))], spans)


def _boundaries(
    reference: str,
    hypothesis: str,
    expected: Sequence[int],
    chunks: Sequence[ChunkOutcome],
) -> float:
    """Насколько границы чанков совпали с границами секций эталона."""
    wanted = [_normalized_offset(reference, offset) for offset in expected]
    found = [_normalized_offset(hypothesis, chunk.start_offset) for chunk in chunks]
    return boundary_f1(wanted, found)


def _normalized_offset(text: str, offset: int) -> int:
    return len(normalize_for_scoring(text[:offset]))


def aggregate(
    scores: Sequence[PageScore], chunks: Sequence[ChunkOutcome]
) -> dict[str, float]:
    """Сводные числа по всему корпусу."""
    if not scores:  # pragma: no cover — пустой корпус не собирается
        return {}
    text_pages = [score for score in scores if score.expected_method == "text_layer"]
    scan_pages = [score for score in scores if score.expected_method == "ocr"]
    unreadable = [score for score in scores if score.expects_unreadable]
    readable = [score for score in scores if not score.expects_unreadable]
    ocr_pages = [score for score in scores if score.mean_confidence is not None]
    return {
        "cer": mean(score.cer for score in scores),
        "wer": mean(score.wer for score in scores),
        "text_layer_detection_accuracy": _ratio(
            [score.method_matches for score in scores]
        ),
        "false_ocr_rate": _ratio(
            [score.actual_method == "ocr" for score in text_pages]
        ),
        "missed_ocr_rate": _ratio(
            [score.actual_method == "text_layer" for score in scan_pages]
        ),
        "illegible_recall": _ratio([score.marked_illegible for score in unreadable]),
        "illegible_false_positive_rate": _ratio(
            [score.marked_illegible for score in readable]
        ),
        "hallucination_rate": _ratio(
            [not score.marked_illegible for score in unreadable]
        ),
        "chunk_page_linkage_errors": float(
            sum(1 for chunk in chunks if not chunk.page_text_matches)
        ),
        "chunk_coverage": mean(score.chunk_coverage for score in scores),
        "boundary_f1": mean(score.boundary_f1 for score in scores),
        "confidence_calibration_rho": spearman(
            [score.mean_confidence or 0.0 for score in ocr_pages],
            [1.0 - score.cer for score in ocr_pages],
        ),
        "pages_total": float(len(scores)),
    }


def by_category(scores: Sequence[PageScore]) -> dict[str, Mapping[str, float]]:
    """Числа по каждой категории отдельно."""
    categories = sorted({score.category for score in scores})
    return {
        category: {
            "cer": mean(score.cer for score in scores if score.category == category),
            "wer": mean(score.wer for score in scores if score.category == category),
            "boundary_f1": mean(
                score.boundary_f1 for score in scores if score.category == category
            ),
            "pages": float(sum(1 for score in scores if score.category == category)),
        }
        for category in categories
    }


def _ratio(flags: Sequence[bool]) -> float:
    # Пустое основание — не ноль и не единица: доли просто нет. Ноль здесь
    # честнее единицы, потому что метрики-инварианты сравниваются с нулём.
    return sum(1 for flag in flags if flag) / len(flags) if flags else 0.0
