"""Версии эмбеддингов, промпта, чанкования и обработки.

Четыре величины и четыре типа: подстановка одной вместо другой ловится mypy, а
не рантаймом. Две первые — свои, две последние приходят от document-worker и
здесь только разбираются. Другого формата версии в проекте нет.

На версии держится вся идемпотентность: сохранение эмбеддингов, выбор корпуса
чанков и сравнение прогонов оценки. Две записи одной величины — это два
namespace вместо одного, причём расхождение не даёт ни ошибки, ни дубля.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NoReturn, Self, override

from ai_worker.domain.constants import (
    MAX_VERSION_PART,
    MIN_VERSION_MAJOR,
    MIN_VERSION_PART,
)
from ai_worker.domain.errors import (
    InvalidChunkingVersion,
    InvalidEmbeddingVersion,
    InvalidPipelineVersion,
    InvalidPromptVersion,
    InvalidVersion,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Ведущие нули запрещены: "01.0.0" и "1.0.0" дали бы разные ключи
# идемпотентности при одинаковом смысле.
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True, slots=True, order=True)
class _SemanticVersion:
    """Общая механика семвера. Наружу отдаются только потомки."""

    major: int
    minor: int
    patch: int

    _error: ClassVar[type[InvalidVersion]] = InvalidVersion

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

    @classmethod
    def highest_of(cls, versions: Iterable[Self]) -> Self:
        """Наибольшая версия из перечисленных.

        Так выбирается версия чанкования документа, когда их несколько: брать
        первую попавшуюся значило бы индексировать устаревший корпус.

        Raises:
            InvalidVersion: Выбирать не из чего.
        """
        ordered = sorted(versions)
        if not ordered:
            cls._reject("выбор наибольшей версии из пустого набора")
        return ordered[-1]

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
class EmbeddingVersion(_SemanticVersion):
    """Версия эмбеддингов. Смена мажорной части открывает новый namespace."""

    _error: ClassVar[type[InvalidVersion]] = InvalidEmbeddingVersion


@dataclass(frozen=True, slots=True, order=True)
class PromptVersion(_SemanticVersion):
    """Версия промпта. Правка текста без инкремента ломает сравнимость оценок."""

    _error: ClassVar[type[InvalidVersion]] = InvalidPromptVersion


@dataclass(frozen=True, slots=True, order=True)
class ChunkingVersion(_SemanticVersion):
    """Версия чанкования document-worker: выбирает корпус чанков документа."""

    _error: ClassVar[type[InvalidVersion]] = InvalidChunkingVersion


@dataclass(frozen=True, slots=True, order=True)
class PipelineVersion(_SemanticVersion):
    """Версия обработки document-worker: переносится в индекс как есть."""

    _error: ClassVar[type[InvalidVersion]] = InvalidPipelineVersion
