"""Модели распознавания: что нужно, откуда взять, как проверить.

Файлы укладываются в образ на сборке, рантайм в сеть не ходит. Отсутствие или
подмена модели — отказ при старте контейнера, а не временная ошибка на первом
сообщении: ошибка конфигурации, выданная за временную, отправляет на
бесконечный повтор всю очередь, а подменённая модель распознаёт что-то другое
и делает это молча.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from document_worker.application.errors import OcrModelsUnavailableError

if TYPE_CHECKING:
    from collections.abc import Mapping

MODEL_DIR_ENV: Final[str] = "OCR__MODEL_DIR"
DEFAULT_MODEL_DIR: Final[Path] = Path(".models")

_BASE_URL: Final[str] = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx"
)
_DOWNLOAD_TIMEOUT_S: Final[float] = 180.0
_CHUNK_BYTES: Final[int] = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelFile:
    """Одна модель: имя роли, файл, источник и контрольная сумма."""

    name: str
    file_name: str
    url: str
    sha256: str


# Распознавание восточнославянское: словарь PP-OCRv5 eslav покрывает кириллицу
# и латиницу разом, поэтому второй модели под латиницу не нужно. Список
# символов зашит в саму onnx-модель, отдельного файла словаря нет.
REQUIRED_MODELS: Final[tuple[ModelFile, ...]] = (
    ModelFile(
        name="det",
        file_name="ch_PP-OCRv5_det_mobile.onnx",
        url=f"{_BASE_URL}/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        sha256="4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    ModelFile(
        name="cls",
        file_name="ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        url=f"{_BASE_URL}/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        sha256="e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    ),
    ModelFile(
        name="rec",
        file_name="eslav_PP-OCRv5_rec_mobile.onnx",
        url=f"{_BASE_URL}/PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile.onnx",
        sha256="08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab",
    ),
)


def model_dir_from_env() -> Path:
    """Каталог моделей для сборки образа и тестов.

    Рантайм читает его через настройки: окружение разбирает ровно одно место,
    иначе опечатка в имени переменной тихо превращается в значение по умолчанию.
    """
    return Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)


def resolve(model_dir: Path) -> Mapping[str, Path]:
    """Пути ко всем объявленным моделям."""
    return {model.name: model_dir / model.file_name for model in REQUIRED_MODELS}


def missing_models(model_dir: Path) -> tuple[ModelFile, ...]:
    """Модели, которых нет на месте или чья сумма не сошлась."""
    return tuple(
        model
        for model in REQUIRED_MODELS
        if _checksum_of(model_dir / model.file_name) != model.sha256
    )


def verify(model_dir: Path) -> None:
    """Требует, чтобы все модели были на месте и не подменены.

    Raises:
        OcrModelsUnavailableError: Файла нет либо его сумма не совпала.
    """
    for model in REQUIRED_MODELS:
        path = model_dir / model.file_name
        if not path.is_file():
            raise OcrModelsUnavailableError(
                "модель распознавания отсутствует",
                context={"model": model.name, "path": str(path)},
            )
        actual = _checksum_of(path)
        if actual != model.sha256:
            raise OcrModelsUnavailableError(
                "контрольная сумма модели не совпала",
                context={
                    "model": model.name,
                    "expected": model.sha256,
                    "actual": actual,
                },
            )


def download_missing(model_dir: Path) -> tuple[str, ...]:
    """Докачивает недостающие модели и возвращает их имена.

    Вызывается на сборке образа и в тестах, но не в рантайме: сеть во время
    обработки документа запрещена.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for model in missing_models(model_dir):
        _download(model, model_dir / model.file_name)
        downloaded.append(model.name)
    return tuple(downloaded)


def _download(model: ModelFile, destination: Path) -> None:
    partial = destination.with_suffix(f"{destination.suffix}.part")
    with (
        urllib.request.urlopen(model.url, timeout=_DOWNLOAD_TIMEOUT_S) as response,  # noqa: S310 — адрес зафиксирован в реестре
        partial.open("wb") as target,
    ):
        while chunk := response.read(_CHUNK_BYTES):
            target.write(chunk)
    actual = _checksum_of(partial)
    if actual != model.sha256:
        partial.unlink(missing_ok=True)
        raise OcrModelsUnavailableError(
            "скачанная модель не совпала с контрольной суммой",
            context={"model": model.name, "expected": model.sha256, "actual": actual},
        )
    partial.replace(destination)


def _checksum_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
