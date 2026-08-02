"""Обвязка стенда оценки: своя база со схемой и настройки без брокера.

Очередей стенду не нужно: он зовёт `ProcessDocument` напрямую. База нужна
настоящая — страницы и чанки сервис пишет только в неё, и читать их больше
неоткуда.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from alembic import command

from document_worker.infrastructure.config.settings import AppSettings
from eval import corpus, fonts
from eval.local_storage import LocalCorpusStorage
from eval.runner import process_document
from eval.scoring import aggregate, by_category, score_document
from tests.conftest import _create_database, _drop_database, _dsn_for, alembic_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from eval.runner import ChunkOutcome
    from eval.scoring import PageScore

Measured = tuple[
    "corpus.Manifest",
    "Mapping[str, float]",
    "Mapping[str, Mapping[str, float]]",
    "tuple[PageScore, ...]",
]

# Хранилище настоящему стенду не нужно, но настройки обязаны быть валидными.
UNUSED_S3 = {
    "endpoint_url": "http://storage.invalid:9000",
    "access_key": "eval",
    "secret_key": "eval-secret",
    "default_bucket": "corpus",
}


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def eval_dsn(base_dsn: str) -> AsyncIterator[str]:
    """Отдельная база со схемой на весь прогон стенда."""
    name = f"docworker_eval_{uuid.uuid4().hex[:8]}"
    await _create_database(base_dsn, name)
    dsn = _dsn_for(base_dsn, name)
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")
    try:
        yield dsn
    finally:
        await _drop_database(base_dsn, name)


@pytest.fixture(scope="session")
def eval_settings(
    eval_dsn: str,
    model_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> AppSettings:
    """Настройки, которых хватает обработке документа без брокера."""
    return AppSettings.model_validate(
        {
            "database": {"dsn": eval_dsn},
            "rabbit": {"url": "amqp://guest:guest@broker.invalid:5672/"},
            "s3": UNUSED_S3,
            "processing": {"temp_dir": str(tmp_path_factory.mktemp("eval-work"))},
            "ocr": {"model_dir": str(model_dir)},
        }
    )


SMOKE = tuple(
    spec
    for spec in corpus.CORPUS
    if spec.category
    in {corpus.Category.DIGITAL_PDF, corpus.Category.PARTIALLY_UNREADABLE}
)[:2]


async def measure(
    settings: AppSettings,
    root: Path,
    manifest: corpus.Manifest,
) -> Measured:
    """Гонит корпус через настоящую обработку и сводит числа."""
    from document_worker.bootstrap.composition import (  # noqa: PLC0415 — сборка тянет модели
        build_processing,
    )

    scores: list[PageScore] = []
    chunks: list[ChunkOutcome] = []
    async with build_processing(
        settings, storage=LocalCorpusStorage(root=root)
    ) as processing:
        for truth in manifest.documents:
            outcome = await process_document(
                processing, doc_id=truth.doc_id, bucket=root.name
            )
            scores.extend(score_document(truth, outcome, corpus_root=root))
            chunks.extend(outcome.chunks)
    return manifest, aggregate(scores, chunks), by_category(scores), tuple(scores)


@pytest.fixture(scope="session")
def smoke_corpus(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, corpus.Manifest]:
    """Подмножество корпуса: текстовый документ и нечитаемая вставка."""
    font_dir = fonts.font_dir_from_env()
    fonts.download_missing(font_dir)
    root = tmp_path_factory.mktemp("smoke-corpus")
    return root, corpus.generate(root, font_dir=font_dir, specs=SMOKE)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def smoke_measured(
    eval_settings: AppSettings,
    smoke_corpus: tuple[Path, corpus.Manifest],
) -> Measured:
    """Быстрый прогон подмножества — им проверяются инварианты на PR."""
    return await measure(eval_settings, *smoke_corpus)


@pytest.fixture(scope="session")
def full_corpus(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, corpus.Manifest]:
    """Весь корпус, собранный один раз на прогон."""
    font_dir = fonts.font_dir_from_env()
    fonts.download_missing(font_dir)
    root = tmp_path_factory.mktemp("full-corpus")
    return root, corpus.generate(root, font_dir=font_dir)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def measured(
    eval_settings: AppSettings,
    full_corpus: tuple[Path, corpus.Manifest],
) -> Measured:
    """Один прогон всего корпуса через настоящую обработку."""
    return await measure(eval_settings, *full_corpus)
