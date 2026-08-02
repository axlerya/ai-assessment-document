"""Корпус оценки качества: состав, генерация и отпечаток.

Корпус синтетический и генерируемый, а не собранный вручную: только так эталон
известен посимвольно и CER считается точно, а не относительно чужой разметки.
В git он не кладётся — генерируется за минуту и весит десятки мегабайт.

Отпечаток `corpus_hash` покрывает всё, что влияет на числа: файлы документов,
эталоны, параметры генерации, seed, версию генератора и сумму шрифта. Без него
метрики улучшаются удалением трудных документов, и по одному отчёту этого не
видно.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from eval import fonts
from eval.corpus_pdf import Degradation, PagePlan, render
from eval.corpus_text import PageContent, build_document

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


CORPUS_VERSION: Final[str] = "1.0.0"
DEFAULT_SEED: Final[int] = 20260729
MANIFEST_NAME: Final[str] = "manifest.json"
_EVERY_SECOND: Final[int] = 2
# Третья страница сложного документа — двухколоночная.
_TWO_COLUMN_PAGE: Final[int] = 2
SOURCE_NAME: Final[str] = "source.pdf"
GROUND_TRUTH_DIR: Final[str] = "ground_truth"

CLEAN = Degradation(dpi=300, skew_deg=0.0, noise_sigma=0.0, jpeg_quality=85)
NOISY = Degradation(dpi=200, skew_deg=1.8, noise_sigma=12.0, jpeg_quality=55)
_CHUNK_BYTES: Final[int] = 1024 * 1024


class Category(StrEnum):
    """Категории корпуса: по одной на способ испортить документ."""

    DIGITAL_PDF = "digital_pdf"
    CLEAN_SCAN = "clean_scan"
    NOISY_SCAN = "noisy_scan"
    COMPLEX_LAYOUT = "complex_layout"
    PARTIALLY_UNREADABLE = "partially_unreadable"


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    """Что за документ и как его портить."""

    doc_id: str
    category: Category
    page_count: int


CORPUS: Final[tuple[DocumentSpec, ...]] = (
    DocumentSpec("A01_digital_supply", Category.DIGITAL_PDF, 7),
    DocumentSpec("A02_digital_services", Category.DIGITAL_PDF, 6),
    DocumentSpec("B01_scan_clean_lease", Category.CLEAN_SCAN, 5),
    DocumentSpec("B02_scan_clean_works", Category.CLEAN_SCAN, 5),
    DocumentSpec("C01_scan_noisy_supply", Category.NOISY_SCAN, 5),
    DocumentSpec("C02_scan_noisy_lease", Category.NOISY_SCAN, 5),
    DocumentSpec("D01_complex_layout", Category.COMPLEX_LAYOUT, 6),
    DocumentSpec("E01_partially_unreadable", Category.PARTIALLY_UNREADABLE, 5),
)


@dataclass(frozen=True, slots=True)
class PageTruth:
    """Эталон одной страницы."""

    number: int
    expected_extraction_method: str
    expected_page_status: str
    section_boundaries: tuple[int, ...]
    unreadable_text: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentTruth:
    """Эталон документа целиком."""

    doc_id: str
    category: str
    page_count: int
    sha256: str
    pages: tuple[PageTruth, ...]


@dataclass(frozen=True, slots=True)
class Manifest:
    """Опись корпуса вместе с его отпечатком."""

    corpus_version: str
    seed: int
    font_version: str
    font_checksum: str
    corpus_hash: str
    documents: tuple[DocumentTruth, ...] = field(default_factory=tuple)


def generate(
    out: Path,
    *,
    font_dir: Path,
    specs: Sequence[DocumentSpec] = CORPUS,
    seed: int = DEFAULT_SEED,
) -> Manifest:
    """Строит корпус на диске и возвращает его опись."""
    fonts.verify(font_dir)
    faces = fonts.resolve(font_dir)
    out.mkdir(parents=True, exist_ok=True)
    documents = tuple(
        _build(out, spec, faces=faces, seed=seed + index * 1_000)
        for index, spec in enumerate(specs)
    )
    manifest = Manifest(
        corpus_version=CORPUS_VERSION,
        seed=seed,
        font_version=fonts.FONT_VERSION,
        font_checksum=fonts.checksum(),
        corpus_hash=fingerprint(out, seed=seed),
        documents=documents,
    )
    (out / MANIFEST_NAME).write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def fingerprint(root: Path, *, seed: int) -> str:
    """Отпечаток корпуса: файлы, эталоны, параметры, seed, версия и шрифт."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "corpus_version": CORPUS_VERSION,
                "seed": seed,
                "font_checksum": fonts.checksum(),
                "font_version": fonts.FONT_VERSION,
                "clean": asdict(CLEAN),
                "noisy": asdict(NOISY),
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(tree_hash(root).encode())
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    """Сумма всех файлов корпуса, кроме самой описи."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(_file_sha256(path).encode())
    return digest.hexdigest()


def _build(
    out: Path,
    spec: DocumentSpec,
    *,
    faces: Mapping[str, Path],
    seed: int,
) -> DocumentTruth:
    pages = build_document(seed=seed, pages=spec.page_count)
    plans = _plans(spec, len(pages))
    directory = out / spec.doc_id
    truth_dir = directory / GROUND_TRUTH_DIR
    truth_dir.mkdir(parents=True, exist_ok=True)
    source = directory / SOURCE_NAME
    render(source, pages, plans, fonts=faces, seed=seed)

    truths: list[PageTruth] = []
    for index, (page, plan) in enumerate(zip(pages, plans, strict=True)):
        visible, hidden = _split(page, plan)
        (truth_dir / f"page_{index + 1:04d}.txt").write_text(visible, encoding="utf-8")
        truths.append(
            PageTruth(
                number=index + 1,
                expected_extraction_method="text_layer"
                if plan.mode == "text"
                else "ocr",
                expected_page_status=(
                    "partially_illegible" if plan.unreadable else "extracted"
                ),
                section_boundaries=page.section_boundaries,
                unreadable_text=hidden,
            )
        )
    _write_meta(directory, spec, truths, source)
    return DocumentTruth(
        doc_id=spec.doc_id,
        category=spec.category.value,
        page_count=len(pages),
        sha256=_file_sha256(source),
        pages=tuple(truths),
    )


def _split(page: PageContent, plan: PagePlan) -> tuple[str, str | None]:
    """Делит страницу на видимое и заведомо нечитаемое.

    Нечитаемая вставка нарисована штрихами, и распознать её нечем. В эталон
    она не попадает: требовать её текст значило бы требовать выдумки. Он
    сохраняется отдельно — по нему видно, что именно сервис обязан был
    пометить, а не досочинить.
    """
    if not plan.unreadable:
        return page.text, None
    visible = PageContent(blocks=page.blocks[:-1])
    return visible.text, "\n".join(page.blocks[-1].lines)


def _plans(spec: DocumentSpec, pages: int) -> tuple[PagePlan, ...]:
    if spec.category is Category.DIGITAL_PDF:
        return tuple(PagePlan(mode="text") for _ in range(pages))
    if spec.category is Category.CLEAN_SCAN:
        return tuple(PagePlan(mode="scan", degradation=CLEAN) for _ in range(pages))
    if spec.category is Category.NOISY_SCAN:
        return tuple(PagePlan(mode="scan", degradation=NOISY) for _ in range(pages))
    if spec.category is Category.COMPLEX_LAYOUT:
        return _complex_plans(pages)
    return tuple(
        PagePlan(mode="scan", degradation=NOISY, unreadable=index == pages - 2)
        for index in range(pages)
    )


def _complex_plans(pages: int) -> tuple[PagePlan, ...]:
    # Текстовые и вклеенные страницы вперемешку, одна со штампом и одна в две
    # колонки: здесь измеряется выбор способа извлечения, а не только качество.
    built: list[PagePlan] = []
    for index in range(pages):
        if index % _EVERY_SECOND == 0:
            built.append(
                PagePlan(mode="text", columns=2 if index == _TWO_COLUMN_PAGE else 1)
            )
        else:
            built.append(
                PagePlan(mode="scan", degradation=CLEAN, stamped=index == pages - 2)
            )
    return tuple(built)


def _write_meta(
    directory: Path,
    spec: DocumentSpec,
    truths: Sequence[PageTruth],
    source: Path,
) -> None:
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "doc_id": spec.doc_id,
                "category": spec.category.value,
                "page_count": len(truths),
                "sha256": _file_sha256(source),
                "pages": [asdict(page) for page in truths],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
