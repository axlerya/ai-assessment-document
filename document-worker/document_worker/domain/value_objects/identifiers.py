"""Идентификаторы домена.

Типы намеренно не взаимозаменяемы: подстановка PageId туда, где ждут ChunkId,
ловится mypy.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

from document_worker.domain.constants import (
    MAX_CORRELATION_ID_LENGTH,
    MIN_CORRELATION_ID_LENGTH,
    NS_DOCWORKER,
)
from document_worker.domain.errors import InvalidCorrelationId, InvalidIdentifier

if TYPE_CHECKING:
    from document_worker.domain.value_objects.versioning import PipelineVersion

_CORRELATION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


@dataclass(frozen=True, slots=True)
class _UuidId:
    """Общая механика UUID-идентификатора. Наружу отдаются только потомки."""

    value: uuid.UUID

    def __post_init__(self) -> None:
        """Отвергает nil-UUID."""
        if self.value.int == 0:
            raise InvalidIdentifier(
                f"{type(self).__name__} не может быть nil-UUID",
                context={"type": type(self).__name__},
            )

    @classmethod
    def generate(cls) -> Self:
        """Создаёт случайный идентификатор."""
        return cls(uuid.uuid4())

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Разбирает каноническую строку UUID."""
        try:
            value = uuid.UUID(raw)
        except (ValueError, AttributeError, TypeError) as error:
            raise InvalidIdentifier(
                f"{raw!r} не является UUID",
                context={"type": cls.__name__, "raw": raw},
            ) from error
        return cls(value)

    @override
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class DocumentId(_UuidId):
    """Идентификатор документа. Создаётся не воркером, а сервисом приёма файлов."""


@dataclass(frozen=True, slots=True)
class PageId(_UuidId):
    """Идентификатор страницы."""


@dataclass(frozen=True, slots=True)
class ChunkId(_UuidId):
    """Идентификатор чанка."""


@dataclass(frozen=True, slots=True)
class JobId(_UuidId):
    """Идентификатор прогона обработки."""


@dataclass(frozen=True, slots=True)
class EventId(_UuidId):
    """Идентификатор доменного события."""

    @classmethod
    def deterministic(
        cls,
        *,
        document_id: DocumentId,
        pipeline_version: PipelineVersion,
        event_type: str,
    ) -> Self:
        """Ключ дедупликации outbox: повторное завершение даёт тот же event_id."""
        name = f"{document_id}|{pipeline_version}|{event_type}"
        return cls(uuid.uuid5(NS_DOCWORKER, name))


@dataclass(frozen=True, slots=True)
class CorrelationId:
    """Сквозной идентификатор запроса, приходит извне и не генерируется здесь."""

    value: str

    def __post_init__(self) -> None:
        """Проверяет длину и набор символов."""
        if (
            not MIN_CORRELATION_ID_LENGTH
            <= len(self.value)
            <= MAX_CORRELATION_ID_LENGTH
        ):
            raise InvalidCorrelationId(
                f"длина correlation_id вне "
                f"{MIN_CORRELATION_ID_LENGTH}..{MAX_CORRELATION_ID_LENGTH}",
                context={"length": len(self.value)},
            )
        if _CORRELATION_ID_RE.match(self.value) is None:
            raise InvalidCorrelationId(
                "correlation_id содержит недопустимые символы",
                context={"value": self.value},
            )

    @override
    def __str__(self) -> str:
        """Возвращает исходную строку."""
        return self.value
