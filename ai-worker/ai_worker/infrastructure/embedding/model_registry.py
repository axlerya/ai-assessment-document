"""Файлы модели: что нужно, откуда взять, как проверить.

Веса укладываются в образ на сборке, рантайм в сеть не ходит. Отсутствие или
подмена модели — отказ при старте контейнера, а не временная ошибка на первом
сообщении: ошибка конфигурации, выданная за временную, отправляет на бесконечный
повтор всю очередь, а подменённая модель считает другие векторы и делает это
молча — ни ошибки, ни дубля, просто другая геометрия.

Источник закреплён за ревизией репозитория модели, а не за веткой: иначе между
двумя сборками образа в него приезжают другие веса под тем же именем.

Сверка сумм читает файлы целиком, и на весах это секунды. Плата разовая, на
старте процесса, и она покупает единственную защиту от подмены.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ai_worker.application.errors import EmbeddingModelMissing

if TYPE_CHECKING:
    from collections.abc import Sequence

MODEL_DIR_ENV: Final[str] = "EMBEDDING__MODEL_DIR"
DEFAULT_MODEL_DIR: Final[Path] = Path(".models")

# Ревизия репозитория BAAI/bge-m3, под которую посчитаны суммы ниже.
MODEL_REVISION: Final[str] = "5617a9f61b028005a4858fdac845db406aefb181"
_BASE_URL: Final[str] = f"https://huggingface.co/BAAI/bge-m3/resolve/{MODEL_REVISION}"

_DOWNLOAD_TIMEOUT_S: Final[float] = 600.0
_CHUNK_BYTES: Final[int] = 1024 * 1024

# Переменные, запрещающие библиотекам ходить в сеть за моделью. Выставляются в
# рабочем процессе до импорта transformers: после импорта их уже не читают.
OFFLINE_ENVIRONMENT: Final[dict[str, str]] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


@dataclass(frozen=True, slots=True)
class ModelFile:
    """Один файл модели: роль, имя, источник и контрольная сумма."""

    name: str
    file_name: str
    url: str
    sha256: str


def _declare(name: str, file_name: str, sha256: str) -> ModelFile:
    return ModelFile(
        name=name,
        file_name=file_name,
        url=f"{_BASE_URL}/{file_name}",
        sha256=sha256,
    )


# Веса, конфигурация и токенизатор — обязательны все: другой токенизатор даёт
# другие идентификаторы токенов, а значит другой разреженный вектор при том же
# тексте. Голова разреженных весов лежит отдельным файлом и в состав модели
# transformers не входит — без неё было бы только плотное представление.
REQUIRED_FILES: Final[tuple[ModelFile, ...]] = (
    _declare(
        "config",
        "config.json",
        "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
    ),
    _declare(
        "weights",
        "pytorch_model.bin",
        "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
    ),
    _declare(
        "sparse_head",
        "sparse_linear.pt",
        "45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9",
    ),
    _declare(
        "tokenizer",
        "tokenizer.json",
        "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
    ),
    _declare(
        "tokenizer_config",
        "tokenizer_config.json",
        "a62b2b6784f990259fddef5f16388693a8043be4f69179e6a5257eeb3f9abac4",
    ),
    _declare(
        "special_tokens",
        "special_tokens_map.json",
        "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    ),
    _declare(
        "sentencepiece",
        "sentencepiece.bpe.model",
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
)


def model_dir_from_env() -> Path:
    """Каталог модели для сборки образа и тестов.

    Рантайм читает его через настройки: окружение разбирает ровно одно место,
    иначе опечатка в имени переменной тихо превращается в значение по умолчанию.
    """
    return Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)


def missing_files(
    model_dir: Path,
    *,
    files: Sequence[ModelFile] = REQUIRED_FILES,
) -> tuple[ModelFile, ...]:
    """Файлы, которых нет на месте или чья сумма не сошлась."""
    return tuple(
        model_file
        for model_file in files
        if _checksum_of(model_dir / model_file.file_name) != model_file.sha256
    )


def verify(
    model_dir: Path,
    *,
    files: Sequence[ModelFile] = REQUIRED_FILES,
) -> None:
    """Требует, чтобы все файлы были на месте и не подменены.

    Raises:
        EmbeddingModelMissing: Файла нет либо его сумма не совпала.
    """
    for model_file in files:
        path = model_dir / model_file.file_name
        if not path.is_file():
            raise EmbeddingModelMissing(
                "файл модели отсутствует",
                context={"file": model_file.name, "path": str(path)},
            )
        actual = _checksum_of(path)
        if actual != model_file.sha256:
            raise EmbeddingModelMissing(
                "контрольная сумма файла модели не совпала",
                context={
                    "file": model_file.name,
                    "expected": model_file.sha256,
                    "actual": actual,
                },
            )


def download_missing(
    model_dir: Path,
    *,
    files: Sequence[ModelFile] = REQUIRED_FILES,
) -> tuple[str, ...]:
    """Докачивает недостающие файлы и возвращает их роли.

    Вызывается на сборке образа и вручную перед медленными тестами, но не в
    рантайме: сеть во время обработки сообщения запрещена.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for model_file in missing_files(model_dir, files=files):
        _download(model_file, model_dir / model_file.file_name)
        downloaded.append(model_file.name)
    return tuple(downloaded)


def _download(model_file: ModelFile, destination: Path) -> None:
    partial = destination.with_suffix(f"{destination.suffix}.part")
    with (
        urllib.request.urlopen(  # noqa: S310 — адрес зафиксирован в реестре
            model_file.url, timeout=_DOWNLOAD_TIMEOUT_S
        ) as response,
        partial.open("wb") as target,
    ):
        while chunk := response.read(_CHUNK_BYTES):
            target.write(chunk)
    actual = _checksum_of(partial)
    if actual != model_file.sha256:
        partial.unlink(missing_ok=True)
        raise EmbeddingModelMissing(
            "скачанный файл модели не совпал с контрольной суммой",
            context={
                "file": model_file.name,
                "expected": model_file.sha256,
                "actual": actual,
            },
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
