"""Реестр моделей распознавания.

Отсутствие или подмена модели — отказ при старте, а не временная ошибка на
первом сообщении: ошибка конфигурации, замаскированная под временную, отправит
всю очередь на бесконечный повтор.
"""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

import pytest

from document_worker.application.errors import OcrModelsUnavailableError
from document_worker.infrastructure.ocr import model_registry
from document_worker.infrastructure.ocr.model_registry import (
    REQUIRED_MODELS,
    ModelFile,
    missing_models,
    resolve,
    verify,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

PAYLOAD = b"onnx-model-bytes"
FILE_NAME = "rec.onnx"


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Каталог с единственной объявленной моделью на месте."""
    monkeypatch.setattr(
        model_registry,
        "REQUIRED_MODELS",
        (
            ModelFile(
                name="rec",
                file_name=FILE_NAME,
                url="https://example.invalid/rec.onnx",
                sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            ),
        ),
    )
    (tmp_path / FILE_NAME).write_bytes(PAYLOAD)
    return tmp_path


def test_every_declared_model_has_a_url_and_a_hash() -> None:
    assert REQUIRED_MODELS
    for model in REQUIRED_MODELS:
        assert model.url.startswith("https://")
        assert len(model.sha256) == 64


def test_verify_passes_on_complete_directory(model_dir: Path) -> None:
    verify(model_dir)


def test_missing_model_is_refused(model_dir: Path) -> None:
    (model_dir / FILE_NAME).unlink()

    with pytest.raises(OcrModelsUnavailableError, match="отсутств"):
        verify(model_dir)


def test_replaced_model_is_refused(model_dir: Path) -> None:
    # Подменённая модель распознаёт что-то другое, и молча.
    (model_dir / FILE_NAME).write_bytes(b"something else")

    with pytest.raises(OcrModelsUnavailableError, match="контрольная сумма"):
        verify(model_dir)


def test_absent_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OcrModelsUnavailableError):
        verify(tmp_path / "nowhere")


def test_resolve_returns_path_of_every_model(model_dir: Path) -> None:
    paths = resolve(model_dir)

    assert set(paths) == {"rec"}
    assert paths["rec"].read_bytes() == PAYLOAD


def test_missing_models_lists_what_the_build_must_download(tmp_path: Path) -> None:
    assert [model.name for model in missing_models(tmp_path)] == [
        model.name for model in REQUIRED_MODELS
    ]


def _serve(payload: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет сеть: BytesIO ведёт себя как ответ — контекст и read."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )


def test_download_missing_fetches_and_verifies(
    model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (model_dir / FILE_NAME).unlink()
    _serve(PAYLOAD, monkeypatch)

    assert model_registry.download_missing(model_dir) == ("rec",)
    verify(model_dir)


def test_download_skips_models_already_in_place(
    model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(b"never used", monkeypatch)

    assert model_registry.download_missing(model_dir) == ()


def test_download_of_a_replaced_model_is_refused(
    model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Подменённая на зеркале модель не должна попасть в образ: она распознаёт
    # что-то другое и делает это молча.
    (model_dir / FILE_NAME).unlink()
    _serve(b"tampered", monkeypatch)

    with pytest.raises(OcrModelsUnavailableError, match="скачанная"):
        model_registry.download_missing(model_dir)

    assert not list(model_dir.iterdir())
