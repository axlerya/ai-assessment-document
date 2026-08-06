"""Контрольная сумма содержимого чанка.

Нужна ровно для одного: не пересчитывать эмбеддинг чанка, текст которого не
изменился. Формат закреплён нижним регистром и полной длиной — усечённый или
записанный в верхнем регистре хеш перестал бы совпадать с уже сохранённым, и
модель гоняли бы заново на каждой доставке.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Self, override

from ai_worker.domain.errors import InvalidValueObject

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ContentHash:
    """sha256 текста в шестнадцатеричной записи нижнего регистра."""

    value: str

    def __post_init__(self) -> None:
        """Требует канонической записи sha256.

        Raises:
            InvalidValueObject: Строка не является шестнадцатеричным sha256
                нижнего регистра.
        """
        if _SHA256_HEX.match(self.value) is None:
            raise InvalidValueObject(
                "контрольная сумма не является sha256 в нижнем регистре",
                context={"value": self.value},
            )

    @classmethod
    def sha256_of(cls, text: str) -> Self:
        """Считает контрольную сумму текста в UTF-8."""
        return cls(hashlib.sha256(text.encode("utf-8")).hexdigest())

    @override
    def __str__(self) -> str:
        return self.value
