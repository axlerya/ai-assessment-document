"""Идентификаторы домена.

Типы намеренно не взаимозаменяемы: подстановка `ChunkId` туда, где ждут
`DraftId`, ловится mypy. Часть идентификаторов чужая — `DocumentId`, `ChunkId`
и `PageId` создаёт document-worker, — но типизированы они здесь наравне со
своими: чужое происхождение не повод возить их безымянным UUID.

Ключи, участвующие в идемпотентности, детерминированы: повторная доставка
сообщения строит те же значения, и дубли гасятся уникальными ограничениями
базы, а не проверками в Python.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

from ai_worker.domain.constants import NS_AIWORKER
from ai_worker.domain.errors import InvalidIdentifier

if TYPE_CHECKING:
    from ai_worker.domain.value_objects.versioning import (
        EmbeddingVersion,
        PromptVersion,
    )


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
        """Разбирает каноническую строку UUID.

        Raises:
            InvalidIdentifier: Строка не является UUID.
        """
        try:
            value = uuid.UUID(raw)
        except (ValueError, AttributeError, TypeError) as error:
            raise InvalidIdentifier(
                f"{raw!r} не является UUID",
                context={"type": cls.__name__, "raw": raw},
            ) from error
        return cls(value)

    @classmethod
    def _from_name(cls, name: str) -> Self:
        return cls(uuid.uuid5(NS_AIWORKER, name))

    @override
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class DocumentId(_UuidId):
    """Идентификатор документа. Создаётся не здесь, а сервисом приёма файлов."""


@dataclass(frozen=True, slots=True)
class PageId(_UuidId):
    """Идентификатор страницы. Приходит вместе с чанком и нужен для цитаты."""


@dataclass(frozen=True, slots=True)
class ChunkId(_UuidId):
    """Идентификатор чанка. Создаёт document-worker, здесь он ключ индекса."""


@dataclass(frozen=True, slots=True)
class CorrelationId(_UuidId):
    """Сквозной идентификатор запроса, приходит извне и не генерируется здесь."""


@dataclass(frozen=True, slots=True)
class RequestId(_UuidId):
    """Идентификатор просьбы подготовить черновик."""


@dataclass(frozen=True, slots=True)
class RetrievalRunId(_UuidId):
    """Идентификатор прогона поиска."""


@dataclass(frozen=True, slots=True)
class EmbeddingId(_UuidId):
    """Идентификатор эмбеддинга чанка в конкретной версии."""

    @classmethod
    def deterministic(
        cls,
        *,
        chunk_id: ChunkId,
        embedding_version: EmbeddingVersion,
    ) -> Self:
        """Ключ строки эмбеддинга: повторная индексация даёт то же значение."""
        return cls._from_name(f"{chunk_id}|{embedding_version}")


@dataclass(frozen=True, slots=True)
class IndexId(_UuidId):
    """Идентификатор состояния индексации документа в конкретной версии."""

    @classmethod
    def deterministic(
        cls,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> Self:
        """Ключ строки индексации: повторная доставка не создаёт вторую."""
        return cls._from_name(f"{document_id}|{embedding_version}")


@dataclass(frozen=True, slots=True)
class DraftId(_UuidId):
    """Идентификатор черновика."""

    @classmethod
    def deterministic(
        cls,
        *,
        request_id: RequestId,
        prompt_version: PromptVersion,
    ) -> Self:
        """Ключ черновика.

        В ключ входит версия промпта: повтор доставки той же версии гасится, а
        новая версия даёт второй черновик по тому же запросу — иначе сравнить
        две версии промпта на одном документе было бы нечем.
        """
        return cls._from_name(f"{request_id}|{prompt_version}")


@dataclass(frozen=True, slots=True)
class ClaimId(_UuidId):
    """Идентификатор утверждения черновика."""

    @classmethod
    def deterministic(cls, *, draft_id: DraftId, claim_index: int) -> Self:
        """Ключ утверждения: позиция в черновике."""
        return cls._from_name(f"{draft_id}|{claim_index}")


@dataclass(frozen=True, slots=True)
class CitationId(_UuidId):
    """Идентификатор ссылки утверждения на источник."""

    @classmethod
    def deterministic(
        cls,
        *,
        claim_id: ClaimId,
        chunk_id: ChunkId,
        quote_start: int,
    ) -> Self:
        """Ключ цитаты: утверждение, чанк и начало фрагмента."""
        return cls._from_name(f"{claim_id}|{chunk_id}|{quote_start}")


@dataclass(frozen=True, slots=True)
class EventId(_UuidId):
    """Ключ дедупликации исходящего события.

    Форм две, потому что потоков два: индексация опознаётся документом и
    версией эмбеддингов, черновик — запросом и версией промпта. Общей формы у
    них нет, и сводить их к одной строке значило бы прятать это различие.
    """

    @classmethod
    def for_indexing(
        cls,
        *,
        document_id: DocumentId,
        embedding_version: EmbeddingVersion,
    ) -> Self:
        """Ключ события `document.indexed`."""
        return cls._from_name(f"{document_id}|{embedding_version}|document.indexed")

    @classmethod
    def for_draft(
        cls,
        *,
        request_id: RequestId,
        prompt_version: PromptVersion,
        event_type: str,
    ) -> Self:
        """Ключ события о черновике."""
        return cls._from_name(f"{request_id}|{prompt_version}|{event_type}")
