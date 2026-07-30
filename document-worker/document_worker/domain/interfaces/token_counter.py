"""Подсчёт токенов — единственная внешняя зависимость чанкования."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Считает число токенов в тексте."""

    def count(self, text: str) -> int:
        """Возвращает число токенов."""
        ...
