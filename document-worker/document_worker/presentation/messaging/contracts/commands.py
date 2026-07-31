"""Входящая команда обработки документа.

Здесь проверяется форма сообщения, а не предметная область: список
поддерживаемых типов файлов — доменная политика, и в транспортном контракте
она стала бы невидимой для остальных слоёв.

Полей, которые сервис не читает, в контракте нет намеренно: молча
игнорируемое поле в межсервисном протоколе хуже его отсутствия. Незнакомые
поля продюсера при этом принимаются — за это отвечает `extra="ignore"`.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from document_worker.domain.constants import (
    MAX_BUCKET_NAME_LENGTH,
    MAX_OBJECT_KEY_LENGTH,
    MIN_BUCKET_NAME_LENGTH,
)
from document_worker.presentation.messaging.contracts.base import (
    IncomingMessageBase,
)

DOCUMENT_KEY_PREFIX = "documents"

ObjectKey = Annotated[str, Field(min_length=1, max_length=MAX_OBJECT_KEY_LENGTH)]
BucketName = Annotated[
    str,
    Field(
        min_length=MIN_BUCKET_NAME_LENGTH,
        max_length=MAX_BUCKET_NAME_LENGTH,
        pattern=r"^[a-z0-9][a-z0-9.\-]*[a-z0-9]$",
    ),
]
MimeType = Annotated[
    str, Field(min_length=3, max_length=127, pattern=r"^[a-z]+/[a-z0-9.+-]+$")
]

_TRAVERSAL = re.compile(r"(^/)|(//)|(\\)|(^\.\.$)|(^\.\./)|(/\.\./)|(/\.\.$)")


class ProcessDocumentRequestedV1(IncomingMessageBase):
    """Просьба обработать документ."""

    schema_version: Literal[1] = 1

    document_id: UUID
    object_key: ObjectKey
    mime_type: MimeType
    # Бакет необязателен: обязательный зашивал бы деталь хранилища в
    # межсервисный протокол. При отсутствии подставляется бакет по умолчанию.
    bucket: BucketName | None = None

    @field_validator("object_key")
    @classmethod
    def reject_path_traversal(cls, value: str) -> str:
        """Не пропускает выход за пределы префикса.

        Ключ приходит извне, и обработать по нему чужой файл значит записать в
        документ содержимое другого. Хранилище проверяет то же самое для ключей
        любого происхождения — здесь отсекается конкретно сообщение.

        Raises:
            ValueError: Ключ ведёт за пределы своего префикса.
        """
        if _TRAVERSAL.search(value) is not None:
            raise ValueError("ключ объекта ведёт за пределы своего префикса")
        return value

    @model_validator(mode="after")
    def key_belongs_to_the_document(self) -> Self:
        """Требует, чтобы ключ лежал в префиксе своего документа.

        Raises:
            ValueError: Ключ принадлежит другому документу.
        """
        if not self.object_key.startswith(f"{DOCUMENT_KEY_PREFIX}/{self.document_id}/"):
            raise ValueError("ключ объекта не принадлежит документу")
        return self
