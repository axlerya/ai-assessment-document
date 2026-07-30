"""Временная рабочая директория одной обработки.

Каталог обязан исчезать при любом исходе. Оставленный после падения или
остановки воркера, он переживёт перезапуск и тихо съест диск: скачанный PDF и
рендеры трёхсот страниц это сотни мегабайт на документ.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from document_worker.application.ports.system import TempWorkspace, TempWorkspaceFactory
from document_worker.infrastructure.storage.temp_workspace import (
    TempDirWorkspaceFactory,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.unit

PREFIX = "docworker-"


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """Каталог, внутри которого фабрика создаёт рабочие директории."""
    return tmp_path


@pytest.fixture
def factory(base_dir: Path) -> TempDirWorkspaceFactory:
    return TempDirWorkspaceFactory(base_dir=base_dir)


def test_factory_satisfies_its_port(factory: TempDirWorkspaceFactory) -> None:
    assert isinstance(factory, TempWorkspaceFactory)


async def test_workspace_satisfies_its_port(
    factory: TempDirWorkspaceFactory,
) -> None:
    async with factory(prefix=PREFIX) as workspace:
        assert isinstance(workspace, TempWorkspace)


async def test_workspace_directory_exists_inside_the_block(
    factory: TempDirWorkspaceFactory,
) -> None:
    async with factory(prefix=PREFIX) as workspace:
        assert workspace.root.is_dir()


async def test_workspace_directory_is_removed_on_success(
    factory: TempDirWorkspaceFactory,
) -> None:
    async with factory(prefix=PREFIX) as workspace:
        root = workspace.root
        (root / "source.pdf").write_bytes(b"%PDF-1.7")

    assert not root.exists()


async def test_workspace_directory_is_removed_on_exception(
    factory: TempDirWorkspaceFactory,
) -> None:
    seen: list[Path] = []

    with pytest.raises(RuntimeError):
        await _fails_inside(factory, seen)

    assert not seen[0].exists()


async def _fails_inside(
    factory: TempDirWorkspaceFactory,
    seen: list[Path],
) -> None:
    async with factory(prefix=PREFIX) as workspace:
        seen.append(workspace.root)
        workspace.path_for("page-1.png").write_bytes(b"png")
        raise RuntimeError


async def test_workspace_directory_is_removed_on_cancellation(
    factory: TempDirWorkspaceFactory,
) -> None:
    # Остановка воркера приходит именно так, и каталог не должен пережить её.
    seen: list[Path] = []

    async def work() -> None:
        async with factory(prefix=PREFIX) as workspace:
            seen.append(workspace.root)
            await asyncio.sleep(3600)

    task = asyncio.create_task(work())
    await _until(lambda: bool(seen))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not seen[0].exists()


async def test_workspace_name_is_not_predictable(
    factory: TempDirWorkspaceFactory,
) -> None:
    names = set()
    for _ in range(5):
        async with factory(prefix=PREFIX) as workspace:
            names.add(workspace.root.name)

    assert len(names) == 5
    assert all(name.startswith(PREFIX) for name in names)


async def test_workspace_is_created_inside_the_given_base_directory(
    factory: TempDirWorkspaceFactory,
    base_dir: Path,
) -> None:
    # Рабочий каталог не должен появляться рядом с исходниками сервиса.
    async with factory(prefix=PREFIX) as workspace:
        assert workspace.root.parent == base_dir


async def test_no_files_are_left_outside_workspace(
    factory: TempDirWorkspaceFactory,
    base_dir: Path,
) -> None:
    before = set(base_dir.iterdir())

    async with factory(prefix=PREFIX) as workspace:
        workspace.path_for("source.pdf").write_bytes(b"%PDF-1.7")

    assert set(base_dir.iterdir()) == before


async def test_path_for_stays_inside_the_workspace(
    factory: TempDirWorkspaceFactory,
) -> None:
    async with factory(prefix=PREFIX) as workspace:
        assert workspace.path_for("page-1.png").parent == workspace.root


@pytest.mark.parametrize("name", ["../escape", "sub/dir", "", ".", "..", "/abs"])
async def test_path_for_rejects_anything_but_a_plain_name(
    factory: TempDirWorkspaceFactory,
    name: str,
) -> None:
    # Имя приходит из обработки, а не из константы: выход за пределы каталога
    # обязан быть невозможен по построению.
    async with factory(prefix=PREFIX) as workspace:
        with pytest.raises(ValueError, match="имя файла"):
            workspace.path_for(name)


async def test_sweeper_removes_stale_workspaces(
    factory: TempDirWorkspaceFactory,
    base_dir: Path,
) -> None:
    stale = base_dir / f"{PREFIX}stale"
    stale.mkdir()
    (stale / "source.pdf").write_bytes(b"%PDF-1.7")

    removed = factory.sweep(prefix=PREFIX, older_than_s=0.0)

    assert removed == 1
    assert not stale.exists()


async def test_sweeper_keeps_fresh_workspaces(
    factory: TempDirWorkspaceFactory,
    base_dir: Path,
) -> None:
    fresh = base_dir / f"{PREFIX}fresh"
    fresh.mkdir()

    removed = factory.sweep(prefix=PREFIX, older_than_s=3600.0)

    assert removed == 0
    assert fresh.exists()


async def test_sweeper_ignores_foreign_directories(
    factory: TempDirWorkspaceFactory,
    base_dir: Path,
) -> None:
    foreign = base_dir / "someone-elses"
    foreign.mkdir()

    removed = factory.sweep(prefix=PREFIX, older_than_s=0.0)

    assert removed == 0
    assert foreign.exists()


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)
    msg = "условие так и не выполнилось"
    raise AssertionError(msg)
