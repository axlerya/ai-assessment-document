"""Реестр файлов модели: что нужно, откуда взять, как проверить."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ai_worker.application.errors import EmbeddingModelMissing
from ai_worker.infrastructure.embedding.model_registry import (
    DEFAULT_MODEL_DIR,
    MODEL_DIR_ENV,
    OFFLINE_ENVIRONMENT,
    REQUIRED_FILES,
    ModelFile,
    missing_files,
    model_dir_from_env,
    verify,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.unit


def _declare(directory: Path, name: str, content: bytes) -> ModelFile:
    (directory / name).write_bytes(content)
    return ModelFile(
        name=name,
        file_name=name,
        url=f"https://example.invalid/{name}",
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_every_required_file_is_declared_once() -> None:
    names = [model_file.file_name for model_file in REQUIRED_FILES]

    assert sorted(names) == sorted(set(names))


def test_weights_and_tokenizer_are_both_required() -> None:
    # Без токенизатора модель считает не то же самое, что при обучении, и
    # молча: ошибки нет, векторы другие.
    names = {model_file.file_name for model_file in REQUIRED_FILES}

    assert {"pytorch_model.bin", "sparse_linear.pt", "tokenizer.json"} <= names


def test_every_source_is_pinned_to_a_revision() -> None:
    # Ссылка на ветку означает, что модель может смениться между сборками.
    floating = [
        model_file.file_name
        for model_file in REQUIRED_FILES
        if "/resolve/main/" in model_file.url
    ]

    assert not floating, f"источник не закреплён за ревизией: {floating}"


def test_every_checksum_is_a_sha256() -> None:
    wrong = [
        model_file.file_name
        for model_file in REQUIRED_FILES
        if len(model_file.sha256) != 64
    ]

    assert not wrong, f"контрольная сумма не sha256: {wrong}"


def test_model_dir_comes_from_the_service_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV, "/opt/embedding-model")

    assert model_dir_from_env() == Path("/opt/embedding-model")


def test_model_dir_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)

    assert model_dir_from_env() == DEFAULT_MODEL_DIR


def test_verify_accepts_a_complete_directory(tmp_path: Path) -> None:
    declared = [_declare(tmp_path, "weights.bin", "веса".encode())]

    verify(tmp_path, files=declared)

    assert missing_files(tmp_path, files=declared) == ()


def test_verify_reports_a_missing_file(tmp_path: Path) -> None:
    absent = ModelFile(
        name="weights",
        file_name="weights.bin",
        url="https://example.invalid/weights.bin",
        sha256="0" * 64,
    )

    with pytest.raises(EmbeddingModelMissing, match="отсутствует"):
        verify(tmp_path, files=[absent])


def test_verify_reports_a_substituted_file(tmp_path: Path) -> None:
    # Подменённая модель считает что-то другое и делает это молча, поэтому
    # отказ обязан быть на старте, а не на первом сообщении.
    declared: Sequence[ModelFile] = [_declare(tmp_path, "weights.bin", "веса".encode())]
    (tmp_path / "weights.bin").write_bytes("другие веса".encode())

    with pytest.raises(EmbeddingModelMissing, match="сумма"):
        verify(tmp_path, files=declared)


def test_missing_file_is_listed_for_download(tmp_path: Path) -> None:
    declared = _declare(tmp_path, "weights.bin", "веса".encode())
    (tmp_path / "weights.bin").unlink()

    assert missing_files(tmp_path, files=[declared]) == (declared,)


def test_runtime_environment_forbids_going_online() -> None:
    # Рантайм в сеть не ходит: скачивание в середине обработки превращает
    # ошибку конфигурации во временную и отправляет очередь на бесконечный
    # повтор.
    assert OFFLINE_ENVIRONMENT["HF_HUB_OFFLINE"] == "1"
    assert OFFLINE_ENVIRONMENT["TRANSFORMERS_OFFLINE"] == "1"
