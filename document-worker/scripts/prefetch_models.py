"""Прогрев кэша токенизатора на этапе сборки образа.

tiktoken скачивает словарь BPE при первом обращении к кодировке. Сеть в
рантайме запрещена, поэтому словарь укладывается в образ здесь: без прогрева
первый же документ упёрся бы в недоступный интернет в середине обработки.
"""

from __future__ import annotations

import os
import sys

import tiktoken

from document_worker.domain.chunking.policy import DEFAULT_CHUNKING_POLICY


def main() -> int:
    """Загружает кодировку в каталог кэша и проверяет, что она работает."""
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_dir:
        print("TIKTOKEN_CACHE_DIR не задан: кэш уехал бы в домашний каталог")
        return 1
    name = os.environ.get("CHUNKING__ENCODING", DEFAULT_CHUNKING_POLICY.encoding)
    encoding = tiktoken.get_encoding(name)
    tokens = len(encoding.encode("договор поставки"))
    print(f"кодировка {name} прогрета в {cache_dir}, проверка дала {tokens} токенов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
