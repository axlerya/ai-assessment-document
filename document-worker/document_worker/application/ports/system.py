"""Системные зависимости: время, идентификаторы, временный каталог."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime
    from pathlib import Path
    from uuid import UUID


@runtime_checkable
class Clock(Protocol):
    """Источник времени. Прямых вызовов datetime.now в слоях нет."""

    def now(self) -> datetime:
        """Текущий момент в UTC с указанием зоны."""
        ...


@runtime_checkable
class IdGenerator(Protocol):
    """Источник случайных идентификаторов."""

    def new_uuid(self) -> UUID:
        """Новый случайный UUID."""
        ...


@runtime_checkable
class TempWorkspace(Protocol):
    """Временный каталог одной обработки."""

    @property
    def root(self) -> Path:
        """Корень рабочего каталога."""
        ...

    def path_for(self, name: str) -> Path:
        """Путь внутри рабочего каталога."""
        ...


@runtime_checkable
class TempWorkspaceFactory(Protocol):
    """Создаёт рабочий каталог и убирает его при любом исходе."""

    def __call__(self, *, prefix: str) -> AbstractAsyncContextManager[TempWorkspace]:
        """Открывает рабочий каталог."""
        ...
