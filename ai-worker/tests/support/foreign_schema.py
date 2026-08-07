"""Схема document-worker в тестовой базе.

Она накатывается его собственными миграциями, а не копией DDL в фикстуре.
Копия была бы вторым источником истины для чужой схемы: она разошлась бы с
оригиналом молча, и проверялось бы то, чего в бою нет.

Цена — тестам ai-worker нужно синхронизированное окружение соседнего сервиса.
Это ровно та зависимость, которая есть и при развёртывании: миграции
document-worker выполняются раньше (ADR-0001).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DOCUMENT_WORKER = Path(__file__).resolve().parents[3] / "document-worker"


def skip_unless_supported() -> None:
    """Пропускает тест там, где чужие миграции не запустить."""
    if sys.platform not in {"win32", "linux", "darwin"}:  # pragma: no cover
        pytest.skip("нет способа запустить чужие миграции")


def apply_foreign_schema(dsn: str) -> None:
    """Накатывает миграции document-worker его собственным окружением."""
    # Путь до `uv` разрешается явно: без него окружение соседнего сервиса не
    # поднять, и сказать об этом прямо честнее, чем упасть внутри запуска.
    executable = shutil.which("uv")
    if executable is None:  # pragma: no cover — на машине разработчика uv есть
        pytest.skip("для схемы document-worker нужен uv в PATH")
    completed = subprocess.run(  # noqa: S603
        [
            executable,
            "run",
            "--directory",
            str(DOCUMENT_WORKER),
            "alembic",
            "upgrade",
            "head",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ | {"POSTGRES__DSN": dsn},
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(
            "схема document-worker не накатилась — нужно его окружение "
            f"(`uv sync --directory {DOCUMENT_WORKER}`):\n{completed.stderr[-800:]}"
        )
