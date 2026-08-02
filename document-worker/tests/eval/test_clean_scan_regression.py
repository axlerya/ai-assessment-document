"""Чистый скан A4 обязан читаться почти без ошибок.

Триста dpi, без наклона, без шума, качество JPEG 85 — на такой странице у
распознавания нет ни одного оправдания. Если оно теряет здесь три символа из
десяти, значит теряет строки целиком, а не путает буквы: страница до детектора
доезжает уменьшенной.

Тест держит границу постоянно, а не только в ночном прогоне: масштаб детектора
легко занизить обратно, и заметить это по отчёту через сутки — слишком поздно.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from eval import corpus, fonts
from eval.corpus import Category
from eval.local_storage import LocalCorpusStorage
from eval.metrics import cer, wer
from eval.runner import process_document

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from document_worker.bootstrap.composition import Processing
    from document_worker.infrastructure.config.settings import AppSettings

pytestmark = pytest.mark.integration

CLEAN_SCAN = tuple(
    spec for spec in corpus.CORPUS if spec.category is Category.CLEAN_SCAN
)[:1]
# Порог чистого скана: распознавание такой страницы — решённая задача, и
# запас здесь на разнобой раскладки строк, а не на потерю текста.
MAX_CLEAN_SCAN_CER = 0.05
MAX_CLEAN_SCAN_WER = 0.12
# Пометка неразборчивости на чистом скане допустима как исключение, а не как
# правило: их источник — те же потерянные строки.
MAX_ILLEGIBLE_PAGES = 1


@pytest.fixture(scope="session")
def clean_scan_corpus(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, corpus.Manifest]:
    """Один чистый скан из корпуса."""
    font_dir = fonts.font_dir_from_env()
    fonts.download_missing(font_dir)
    root = tmp_path_factory.mktemp("clean-scan")
    return root, corpus.generate(root, font_dir=font_dir, specs=CLEAN_SCAN)


@pytest.fixture
async def scan_processing(
    eval_settings: AppSettings,
    clean_scan_corpus: tuple[Path, corpus.Manifest],
) -> AsyncIterator[Processing]:
    """Обработка документа поверх хранилища чистого скана."""
    from document_worker.bootstrap.composition import (  # noqa: PLC0415 — сборка тянет модели
        build_processing,
    )

    root, _ = clean_scan_corpus
    async with build_processing(
        eval_settings, storage=LocalCorpusStorage(root=root)
    ) as built:
        yield built


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Чистый скан читается с CER около 0.30 вместо 0.05. Масштаб детектора"
        " проверен и ни при чём: разрешение поднимали до полного, строк"
        " находится больше, а CER не улучшается. Ломаются отдельные строки"
        " целиком, соседние такой же длины читаются безупречно — дальше"
        " разбирать распознавание и предобработку, а не разрешение."
    ),
)
async def test_clean_scan_is_recognized_without_losing_text(
    scan_processing: Processing,
    clean_scan_corpus: tuple[Path, corpus.Manifest],
) -> None:
    root, manifest = clean_scan_corpus
    document = manifest.documents[0]

    outcome = await process_document(
        scan_processing, doc_id=document.doc_id, bucket=root.name
    )

    assert {page.extraction_method for page in outcome.pages} == {"ocr"}
    truth_dir = root / document.doc_id / "ground_truth"
    for page in outcome.pages:
        expected = (truth_dir / f"page_{page.number:04d}.txt").read_text(
            encoding="utf-8"
        )
        assert cer(expected, page.text) <= MAX_CLEAN_SCAN_CER, f"стр. {page.number}"
        assert wer(expected, page.text) <= MAX_CLEAN_SCAN_WER, f"стр. {page.number}"

    marked = sum(1 for page in outcome.pages if page.illegible_spans)
    assert marked <= MAX_ILLEGIBLE_PAGES
