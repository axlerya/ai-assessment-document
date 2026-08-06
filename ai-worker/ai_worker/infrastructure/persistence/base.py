"""Базовый класс ORM и конвенция имён ограничений.

Конвенция одна на metadata и на Alembic: без неё autogenerate придумывает свои
имена, а downgrade потом не находит реальные объекты.

Многоколоночные ограничения именуются явно: шаблон `column_0_N_name` быстро
выходит за 63 байта, а PostgreSQL молча усекает длинные имена — после чего
`drop_constraint` в откате ищет объект, которого нет.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any, Final

from sqlalchemy import MetaData, String
from sqlalchemy.dialects.postgresql import JSONB, NUMERIC, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION: Final[dict[str, str]] = {
    "pk": "pk__%(table_name)s",
    "uq": "uq__%(table_name)s__%(column_0_N_name)s",
    "ck": "ck__%(table_name)s__%(constraint_name)s",
    "fk": "fk__%(table_name)s__%(column_0_N_name)s__%(referred_table_name)s",
    "ix": "ix__%(table_name)s__%(column_0_N_name)s",
}

# Уверенность копируется из document-worker как есть: своя точность сделала бы
# агрегаты двух сервисов несравнимыми.
CONFIDENCE_TYPE: Final[NUMERIC[Decimal]] = NUMERIC(precision=4, scale=3)

DENSE_DIMENSIONS: Final[int] = 1024
SPARSE_DIMENSIONS: Final[int] = 250_002


class Base(DeclarativeBase):
    """Общий предок ORM-моделей сервиса."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {  # noqa: RUF012 — так объявляется в SQLAlchemy 2.0
        uuid.UUID: PG_UUID(as_uuid=True),
        dt.datetime: TIMESTAMP(timezone=True),
        Decimal: CONFIDENCE_TYPE,
        str: String(255),
        dict[str, Any]: JSONB(),
        list[str]: JSONB(),
    }
