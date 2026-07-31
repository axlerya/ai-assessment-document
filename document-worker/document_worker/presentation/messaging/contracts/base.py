"""Общая часть контрактов сообщений.

Входящие модели терпимы к незнакомым полям: продюсер вправе добавить своё, и
ломать этим потребителя нельзя. Исходящие, наоборот, строги — опечатка в имени
поля не должна уехать в межсервисный контракт.

Доменные события живут отдельно и про Pydantic не знают: это транспортные
модели слоя presentation, между ними явные мапперы.
"""

from __future__ import annotations

from datetime import UTC
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

SchemaVersion = Annotated[int, Field(ge=1, le=999)]


class IncomingMessageBase(BaseModel):
    """Общие поля входящего сообщения."""

    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    event_id: UUID
    correlation_id: UUID = Field(default_factory=uuid4)
    occurred_at: AwareDatetime

    @field_validator("occurred_at")
    @classmethod
    def normalize_to_utc(cls, value: AwareDatetime) -> AwareDatetime:
        """Приводит момент к UTC.

        Расхождение часов продюсера сообщение не отклоняет: работа дороже
        точности отметки времени.
        """
        return value.astimezone(UTC)
