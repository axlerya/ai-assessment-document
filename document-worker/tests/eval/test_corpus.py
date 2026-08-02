"""Корпус: воспроизводимость и отпечаток.

Корпус — измерительный эталон. Если он меняется незаметно, метрики перестают
быть сравнимыми между прогонами, а «улучшение качества» достигается удалением
трудных документов.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from eval import corpus, fonts
from eval.corpus import CORPUS, Category, generate

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

ONE_DIGITAL = tuple(spec for spec in CORPUS if spec.category is Category.DIGITAL_PDF)[
    :1
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def font_dir() -> Path:
    """Шрифты корпуса: докачиваются один раз на модуль."""
    directory = fonts.font_dir_from_env()
    fonts.download_missing(directory)
    fonts.verify(directory)
    return directory


def test_the_same_seed_produces_byte_identical_documents(
    font_dir: Path,
    tmp_path: Path,
) -> None:
    # Побайтовое совпадение — единственная проверка, которую нельзя обмануть
    # случайно совпавшим текстом.
    first = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)
    second = generate(tmp_path / "b", font_dir=font_dir, specs=ONE_DIGITAL)

    assert first.corpus_hash == second.corpus_hash
    for document in first.documents:
        left = tmp_path / "a" / document.doc_id / "source.pdf"
        right = tmp_path / "b" / document.doc_id / "source.pdf"
        assert _digest(left) == _digest(right)


def test_another_seed_produces_another_corpus(font_dir: Path, tmp_path: Path) -> None:
    first = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)
    other = generate(tmp_path / "b", font_dir=font_dir, specs=ONE_DIGITAL, seed=1)

    assert first.corpus_hash != other.corpus_hash


def test_corpus_hash_covers_the_seed_and_the_generator_version(
    font_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Отпечаток обязан меняться от правки генератора, даже если файлы совпали:
    # иначе изменившаяся методика сравнивалась бы со старыми числами.
    before = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)
    monkeypatch.setattr(corpus, "CORPUS_VERSION", "99.0.0")
    after = generate(tmp_path / "b", font_dir=font_dir, specs=ONE_DIGITAL)

    assert before.corpus_hash != after.corpus_hash


def test_corpus_hash_covers_the_font(
    font_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)
    monkeypatch.setattr(fonts, "checksum", lambda: "0" * 64)
    after = generate(tmp_path / "b", font_dir=font_dir, specs=ONE_DIGITAL)

    assert before.corpus_hash != after.corpus_hash


def test_corpus_hash_covers_the_ground_truth(
    font_dir: Path,
    tmp_path: Path,
) -> None:
    # Правка эталона без правки документа — самый тихий способ подкрутить числа.
    manifest = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)
    truth = tmp_path / "a" / manifest.documents[0].doc_id / "ground_truth"
    page = next(iter(sorted(truth.iterdir())))
    page.write_text(page.read_text(encoding="utf-8") + "хвост", encoding="utf-8")

    assert (
        corpus.fingerprint(tmp_path / "a", seed=corpus.DEFAULT_SEED)
        != manifest.corpus_hash
    )


def test_ground_truth_matches_the_number_of_pages(
    font_dir: Path,
    tmp_path: Path,
) -> None:
    manifest = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)

    for document in manifest.documents:
        truth = tmp_path / "a" / document.doc_id / "ground_truth"
        assert len(list(truth.iterdir())) == document.page_count
        assert len(document.pages) == document.page_count


def test_every_category_is_present_in_the_corpus() -> None:
    assert {spec.category for spec in CORPUS} == set(Category)


def test_meta_is_written_next_to_the_document(font_dir: Path, tmp_path: Path) -> None:
    manifest = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)

    meta = json.loads(
        (tmp_path / "a" / manifest.documents[0].doc_id / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["category"] == Category.DIGITAL_PDF.value
    assert meta["pages"][0]["expected_extraction_method"] == "text_layer"


def test_ground_truth_is_cyrillic(font_dir: Path, tmp_path: Path) -> None:
    # Латинский корпус измерял бы не тот сценарий: распознавание у сервиса
    # восточнославянское, и документы у него русские.
    manifest = generate(tmp_path / "a", font_dir=font_dir, specs=ONE_DIGITAL)
    truth = tmp_path / "a" / manifest.documents[0].doc_id / "ground_truth"
    text = next(iter(sorted(truth.iterdir()))).read_text(encoding="utf-8")

    assert any("а" <= symbol.lower() <= "я" for symbol in text)
