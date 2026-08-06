"""Тесты обвязки: prek-хуки и контракты import-linter."""

from __future__ import annotations

import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVICE_ROOT.parent
PREK_CONFIG = REPO_ROOT / "prek.toml"

# Хуки по python-файлам обязаны отбирать файлы обоих сервисов: настроенный на
# один каталог хук молча пропускает второй сервис целиком.
PYTHON_FILE_HOOKS = frozenset({"ruff-check", "ruff-format", "debug-statements"})

# Проверки, которые должны существовать для каждого сервиса отдельно: у них
# разные окружения, разные lock-файлы и разные границы слоёв.
PER_SERVICE_HOOKS = frozenset(
    {
        "mypy-ai-worker",
        "import-linter-ai-worker",
        "deptry-ai-worker",
        "uv-lock-check-ai-worker",
        "pytest-unit-ai-worker",
    }
)


def _iter_prek_hooks() -> Iterator[dict[str, Any]]:
    config: dict[str, Any] = tomllib.loads(PREK_CONFIG.read_text(encoding="utf-8"))
    for repo in config["repos"]:
        yield from repo["hooks"]


def _venv_script(name: str) -> Path:
    bin_dir = Path(sys.executable).parent
    for candidate in (bin_dir / name, bin_dir / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    msg = f"консольный скрипт {name} не установлен рядом с {sys.executable}"
    raise AssertionError(msg)


def test_prek_hooks_cover_ai_worker_files() -> None:
    targeted = {
        hook["id"]: str(hook.get("files", ""))
        for hook in _iter_prek_hooks()
        if hook["id"] in PYTHON_FILE_HOOKS
    }

    assert set(targeted) == set(PYTHON_FILE_HOOKS), (
        f"не все хуки по python-файлам настроены: не хватает "
        f"{sorted(PYTHON_FILE_HOOKS - set(targeted))}"
    )
    for hook_id, files in targeted.items():
        assert "ai-worker" in files, (
            f"хук {hook_id} не отбирает файлы ai-worker: files={files!r}"
        )


def test_every_per_service_check_exists_for_ai_worker() -> None:
    declared = {hook["id"] for hook in _iter_prek_hooks()}

    missing = sorted(PER_SERVICE_HOOKS - declared)
    assert not missing, f"нет хуков проверки ai-worker: {missing}"


def test_ai_worker_hooks_run_in_the_service_directory() -> None:
    # Без `--directory` проверка выполнилась бы в окружении другого сервиса и
    # прошла бы, ничего не проверив.
    for hook in _iter_prek_hooks():
        if hook["id"] not in PER_SERVICE_HOOKS:
            continue
        entry = str(hook.get("entry", ""))
        assert "ai-worker" in entry, (
            f"хук {hook['id']} не указывает каталог сервиса: entry={entry!r}"
        )


def test_ai_worker_hook_entries_point_to_existing_scripts() -> None:
    # Проверяются только хуки этого сервиса: чужие запускаются из чужого
    # каталога, и отсюда их пути не разрешить.
    missing: list[str] = []
    for hook in _iter_prek_hooks():
        entry = str(hook.get("entry", ""))
        if "ai-worker" not in entry:
            continue
        for token in shlex.split(entry):
            if not token.endswith(".py"):
                continue
            # Хук запускается либо из корня репозитория, либо из каталога
            # сервиса, поэтому проверяем оба корня.
            if not any((root / token).is_file() for root in (REPO_ROOT, SERVICE_ROOT)):
                missing.append(f"{hook['id']}: {token}")

    assert not missing, f"хуки ссылаются на несуществующие скрипты: {missing}"


def test_import_linter_contracts_pass() -> None:
    result = subprocess.run(  # noqa: S603
        [str(_venv_script("lint-imports")), "--no-cache"],
        cwd=SERVICE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, (
        f"контракты import-linter нарушены:\n{result.stdout}\n{result.stderr}"
    )
