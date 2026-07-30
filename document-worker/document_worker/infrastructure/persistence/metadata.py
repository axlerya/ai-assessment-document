"""Полная metadata схемы.

Модель попадает в metadata только при импорте своего модуля, поэтому Alembic и
сверка со схемой берут metadata отсюда, а не из base.py: там она пустая.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_worker.infrastructure.persistence import models  # noqa: F401
from document_worker.infrastructure.persistence.base import Base

if TYPE_CHECKING:
    from sqlalchemy import MetaData

TARGET_METADATA: MetaData = Base.metadata
