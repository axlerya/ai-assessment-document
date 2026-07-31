"""Счётчик токенов на tiktoken.

Кодировка строится лениво: объект кодировки не пиклится и создаётся заново в
каждом процессе пула, а импорт модуля обязан оставаться дешёвым — spawn
переимпортирует его при каждом запуске рабочего процесса.

Словарь BPE укладывается в образ на сборке: сеть в рантайме запрещена, а первый
`get_encoding` иначе полез бы её скачивать.
"""

from __future__ import annotations

import tiktoken


class TiktokenTokenCounter:
    """Считает токены выбранной кодировкой."""

    __slots__ = ("_encoding", "_encoding_name")

    def __init__(self, encoding_name: str) -> None:
        """Запоминает имя кодировки, не загружая её."""
        self._encoding_name = encoding_name
        self._encoding: tiktoken.Encoding | None = None

    def count(self, text: str) -> int:
        """Число токенов в тексте."""
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        # Без этого последовательность вида <|endoftext|>, встретившаяся в
        # тексте документа, роняет подсчёт посреди чанкования.
        return len(self._encoding.encode(text, disallowed_special=()))
