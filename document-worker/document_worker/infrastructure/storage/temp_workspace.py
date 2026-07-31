"""Временный каталог одной обработки.

Уборка идёт синхронным `rmtree` в `finally`: любой `await` здесь — точка отмены,
и при остановке воркера каталог остался бы на диске. Файлов в нём единицы сотен,
так что блокировка цикла событий измеряется миллисекундами и стоит гарантии.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

WORKSPACE_MODE = 0o700


@dataclass(frozen=True, slots=True)
class TempDirWorkspace:
    """Каталог обработки. Наружу отдаёт только пути внутри себя."""

    root: Path

    def path_for(self, name: str) -> Path:
        """Путь к файлу внутри каталога.

        Raises:
            ValueError: Имя содержит разделители или ведёт за пределы каталога.
        """
        if not name or name in {".", ".."} or Path(name).name != name:
            msg = f"имя файла внутри рабочего каталога недопустимо: {name!r}"
            raise ValueError(msg)
        return self.root / name


@dataclass(frozen=True, slots=True)
class TempDirWorkspaceFactory:
    """Создаёт рабочий каталог и убирает его при любом исходе."""

    base_dir: Path | None = None

    @contextlib.asynccontextmanager
    async def __call__(self, *, prefix: str) -> AsyncIterator[TempDirWorkspace]:
        """Открывает рабочий каталог с непредсказуемым именем."""
        root = Path(
            tempfile.mkdtemp(
                prefix=prefix,
                dir=None if self.base_dir is None else str(self.base_dir),
            )
        )
        # Создание и уборка каталога синхронны намеренно, см. модуль.
        root.chmod(WORKSPACE_MODE)  # noqa: ASYNC240
        try:
            yield TempDirWorkspace(root=root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def sweep(self, *, prefix: str, older_than_s: float) -> int:
        """Убирает каталоги, пережившие падение воркера.

        Возвращает число удалённых. Чужие каталоги не трогает: во временной
        директории живут не только наши.
        """
        base = self.base_dir or Path(tempfile.gettempdir())
        deadline = time.time() - older_than_s
        removed = 0
        for entry in base.iterdir():
            if not entry.is_dir() or not entry.name.startswith(prefix):
                continue
            if entry.stat().st_mtime > deadline:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        return removed
