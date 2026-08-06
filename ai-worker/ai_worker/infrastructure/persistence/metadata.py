"""Метаданные схемы сервиса для Alembic.

Импортирует все модели: без этого `--autogenerate` не видит таблицу и молча
предлагает её удалить. Чужие таблицы сюда не попадают намеренно — иначе первый
же автоген предложит удалить и их.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ai_worker.infrastructure.persistence.base import Base
from ai_worker.infrastructure.persistence.models import (  # noqa: F401 — регистрация
    draft,
    index,
    messaging,
    retrieval,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData

TARGET_METADATA: Final[MetaData] = Base.metadata


def include_object(
    obj: object,  # noqa: ARG001 — сигнатура задана Alembic
    name: str | None,
    type_: str,
    reflected: bool,  # noqa: FBT001 — сигнатура задана Alembic
    compare_to: object,  # noqa: ARG001 — сигнатура задана Alembic
) -> bool:
    """Отсекает объекты, которыми сервис не владеет.

    В одной базе живут таблицы обоих сервисов. Без фильтра `--autogenerate`
    видит чужие таблицы отсутствующими в своих метаданных и предлагает их
    удалить — а однажды кто-нибудь согласится.
    """
    if type_ == "table" and reflected:
        return name in TARGET_METADATA.tables
    return True
