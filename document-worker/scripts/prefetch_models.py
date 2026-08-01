"""Прогрев моделей и кэша токенизатора на этапе сборки образа.

Сеть в рантайме запрещена: первый же документ упёрся бы в недоступный интернет
в середине обработки. Поэтому и словарь BPE, и модели распознавания
укладываются в образ здесь, а их контрольные суммы сверяются на месте.
"""

from __future__ import annotations

import os
import sys

import tiktoken

from document_worker.domain.chunking.policy import DEFAULT_CHUNKING_POLICY
from document_worker.infrastructure.ocr.model_registry import (
    download_missing,
    model_dir_from_env,
    verify,
)


def warm_tokenizer() -> None:
    """Загружает кодировку в каталог кэша и проверяет, что она работает.

    Raises:
        RuntimeError: Каталог кэша не задан, и словарь уехал бы в домашний.
    """
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_dir:
        msg = "TIKTOKEN_CACHE_DIR не задан: кэш уехал бы в домашний каталог"
        raise RuntimeError(msg)
    name = os.environ.get("CHUNKING__ENCODING", DEFAULT_CHUNKING_POLICY.encoding)
    tokens = len(tiktoken.get_encoding(name).encode("договор поставки"))
    print(f"кодировка {name} прогрета в {cache_dir}, проверка дала {tokens} токенов")


def fetch_ocr_models() -> None:
    """Кладёт модели распознавания в каталог и сверяет их суммы."""
    directory = model_dir_from_env()
    downloaded = download_missing(directory)
    verify(directory)
    print(f"модели распознавания в {directory}, скачано: {len(downloaded)}")


def main() -> int:
    """Готовит всё, за чем сервис иначе полез бы в сеть."""
    warm_tokenizer()
    fetch_ocr_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())
