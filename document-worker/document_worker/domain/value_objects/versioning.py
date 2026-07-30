"""Версии пайплайна и чанкования.

Две отдельные величины и два отдельных типа: подстановка одной вместо другой
ловится mypy, а не рантаймом. Другого формата версии в проекте нет.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, NoReturn, Self, override

from document_worker.domain.constants import (
    MAX_VERSION_PART,
    MIN_VERSION_MAJOR,
    MIN_VERSION_PART,
)
from document_worker.domain.errors import (
    InvalidChunkingVersion,
    InvalidPipelineVersion,
    InvalidValueObject,
)

# Ведущие нули запрещены: "01.0.0" и "1.0.0" дали бы разные ключи идемпотентности.
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True, slots=True, order=True)
class _SemanticVersion:
    """Общая механика семвера. Наружу отдаются только потомки."""

    major: int
    minor: int
    patch: int

    _error: ClassVar[type[InvalidValueObject]] = InvalidValueObject

    def __post_init__(self) -> None:
        """Проверяет границы частей версии."""
        if not MIN_VERSION_MAJOR <= self.major <= MAX_VERSION_PART:
            self._reject(f"major вне {MIN_VERSION_MAJOR}..{MAX_VERSION_PART}")
        for name, part in (("minor", self.minor), ("patch", self.patch)):
            if not MIN_VERSION_PART <= part <= MAX_VERSION_PART:
                self._reject(f"{name} вне {MIN_VERSION_PART}..{MAX_VERSION_PART}")

    @classmethod
    def _reject(cls, reason: str) -> NoReturn:
        raise cls._error(f"некорректная версия: {reason}", context={"reason": reason})

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Разбирает строку вида `2.1.0`."""
        match = _SEMVER_RE.match(raw)
        if match is None:
            cls._reject(f"{raw!r} не в формате major.minor.patch")
        major, minor, patch = (int(part) for part in match.groups())
        return cls(major, minor, patch)

    @override
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, other: Self) -> bool:
        """Совпадает ли мажорная часть."""
        return self.major == other.major

    def is_newer_than(self, other: Self) -> bool:
        """Строго ли эта версия больше другой."""
        return (self.major, self.minor, self.patch) > (
            other.major,
            other.minor,
            other.patch,
        )


@dataclass(frozen=True, slots=True, order=True)
class PipelineVersion(_SemanticVersion):
    """Версия извлечения текста. Смена мажорной части открывает новый namespace."""

    _error: ClassVar[type[InvalidValueObject]] = InvalidPipelineVersion


@dataclass(frozen=True, slots=True, order=True)
class ChunkingVersion(_SemanticVersion):
    """Версия чанкования."""

    _error: ClassVar[type[InvalidValueObject]] = InvalidChunkingVersion
