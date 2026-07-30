"""Тесты инструментальной обвязки: prek-хуки, import-linter, формат сообщения коммита."""

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
COMMIT_MESSAGE_CHECK = REPO_ROOT / "scripts" / "check_commit_message.py"

# Хуки, которые обязаны работать по файлам сервиса, а не по несуществующему parser/.
PYTHON_FILE_HOOKS = frozenset({"ruff-check", "ruff-format", "debug-statements"})


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


def _run_commit_message_check(
    tmp_path: Path,
    message: str,
) -> subprocess.CompletedProcess[str]:
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(message, encoding="utf-8")
    return subprocess.run(  # noqa: S603
        [sys.executable, str(COMMIT_MESSAGE_CHECK), str(message_file)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_prek_config_targets_document_worker_paths() -> None:
    hooks = list(_iter_prek_hooks())

    stale = [hook["id"] for hook in hooks if "parser/" in str(hook.get("files", ""))]
    assert not stale, f"хуки нацелены на несуществующий каталог parser/: {stale}"

    targeted = {
        hook["id"]: str(hook.get("files", ""))
        for hook in hooks
        if hook["id"] in PYTHON_FILE_HOOKS
    }
    assert set(targeted) == set(PYTHON_FILE_HOOKS), (
        f"не все хуки по python-файлам настроены: не хватает "
        f"{sorted(PYTHON_FILE_HOOKS - set(targeted))}"
    )
    for hook_id, files in targeted.items():
        assert "document-worker" in files, (
            f"хук {hook_id} не отбирает файлы сервиса: files={files!r}"
        )


def test_every_prek_hook_entry_points_to_existing_script() -> None:
    missing: list[str] = []
    for hook in _iter_prek_hooks():
        for token in shlex.split(str(hook.get("entry", ""))):
            if not token.endswith(".py"):
                continue
            # Хуки запускаются либо из корня репозитория, либо из каталога сервиса
            # (`uv run --directory document-worker`), поэтому проверяем оба корня.
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


@pytest.mark.parametrize(
    "message",
    [
        "feature: добавил каркас слоёв",
        "fix: починил авторизацию на сайте",
        "docs: обновил правила для коммитов",
        "Merge branch 'main' into feature/document-worker-skeleton",
        "Revert \"feature: добавил каркас слоёв\"",
    ],
)
def test_commit_message_check_accepts_valid_message(tmp_path: Path, message: str) -> None:
    result = _run_commit_message_check(tmp_path, f"{message}\n")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "message",
    [
        "chore: обновил зависимости",
        "feat: добавил каркас слоёв",
        "добавил каркас слоёв",
        "feature:",
        "feature: ",
        "feature:добавил каркас слоёв",
        "",
    ],
)
def test_commit_message_check_rejects_invalid_message(
    tmp_path: Path,
    message: str,
) -> None:
    result = _run_commit_message_check(tmp_path, f"{message}\n")
    # Ровно 1, а не «любой ненулевой»: код 2 означает, что упал сам интерпретатор,
    # и тест зеленел бы при отсутствующем скрипте.
    assert result.returncode == 1, f"сообщение {message!r} должно быть отклонено"
    assert result.stderr.strip(), "отклонение обязано объяснять причину"


def test_commit_message_check_ignores_comment_lines(tmp_path: Path) -> None:
    message = (
        "\n"
        "# Please enter the commit message for your changes.\n"
        "feature: добавил каркас слоёв\n"
        "# Changes to be committed:\n"
    )
    result = _run_commit_message_check(tmp_path, message)
    assert result.returncode == 0, result.stderr


def test_commit_message_check_reports_missing_file(tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(COMMIT_MESSAGE_CHECK), str(tmp_path / "нет-такого-файла")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 1
    assert result.stderr.strip()
