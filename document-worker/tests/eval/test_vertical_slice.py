"""Вертикальный срез стенда: генерация → настоящая обработка → метрики.

Проверяется связка целиком, а не её части: воспроизводимый кириллический
документ проходит через тот же `ProcessDocument`, что работает в проде, и на
его выходе считаются CER и WER.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from eval import corpus, fonts
from eval.corpus import Category, generate
from eval.local_storage import LocalCorpusStorage
from eval.metrics import cer, wer
from eval.runner import process_document

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from document_worker.bootstrap.composition import Processing
    from document_worker.infrastructure.config.settings import AppSettings

pytestmark = pytest.mark.integration

DIGITAL = tuple(
    spec for spec in corpus.CORPUS if spec.category is Category.DIGITAL_PDF
)[:1]
# Текстовый слой читается точно: остаток — это разница переносов, которую
# нормализация уже сняла.
MAX_DIGITAL_CER = 0.01


@pytest.fixture(scope="session")
def corpus_fonts() -> Path:
    """Шрифты корпуса, докачанные один раз на прогон."""
    directory = fonts.font_dir_from_env()
    fonts.download_missing(directory)
    fonts.verify(directory)
    return directory


@pytest.fixture(scope="session")
def digital_corpus(
    corpus_fonts: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, corpus.Manifest]:
    """Корпус из одного текстового документа."""
    root = tmp_path_factory.mktemp("corpus")
    return root, generate(root, font_dir=corpus_fonts, specs=DIGITAL)


@pytest.fixture
async def processing(
    eval_settings: AppSettings,
    digital_corpus: tuple[Path, corpus.Manifest],
) -> AsyncIterator[Processing]:
    """Обработка документа поверх хранилища корпуса."""
    from document_worker.bootstrap.composition import (  # noqa: PLC0415 — сборка тянет модели
        build_processing,
    )

    root, _ = digital_corpus
    async with build_processing(
        eval_settings, storage=LocalCorpusStorage(root=root)
    ) as built:
        yield built


async def test_the_same_corpus_is_reproduced_bit_for_bit(
    corpus_fonts: Path,
    digital_corpus: tuple[Path, corpus.Manifest],
    tmp_path: Path,
) -> None:
    root, manifest = digital_corpus
    again = generate(tmp_path / "again", font_dir=corpus_fonts, specs=DIGITAL)

    assert again.corpus_hash == manifest.corpus_hash
    assert corpus.tree_hash(tmp_path / "again") == corpus.tree_hash(root)


async def test_digital_document_is_read_through_the_real_pipeline(
    processing: Processing,
    digital_corpus: tuple[Path, corpus.Manifest],
) -> None:
    root, manifest = digital_corpus
    document = manifest.documents[0]

    outcome = await process_document(processing, doc_id=document.doc_id)

    assert outcome.status == "processed"
    assert len(outcome.pages) == document.page_count
    assert {page.extraction_method for page in outcome.pages} == {"text_layer"}
    assert outcome.chunks
    assert all(chunk.page_text_matches for chunk in outcome.chunks)

    truth_dir = root / document.doc_id / "ground_truth"
    for page in outcome.pages:
        expected = (truth_dir / f"page_{page.number:04d}.txt").read_text(
            encoding="utf-8"
        )
        assert cer(expected, page.text) <= MAX_DIGITAL_CER
        assert wer(expected, page.text) <= MAX_DIGITAL_CER
