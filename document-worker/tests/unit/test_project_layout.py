"""Тесты структуры пакета сервиса и чистоты слоя domain."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

SERVICE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = SERVICE_ROOT / "src" / "document_worker"
LAYER_PACKAGES = (
    "domain",
    "application",
    "infrastructure",
    "presentation",
    "bootstrap",
)


def _iter_imported_modules(module_path: Path) -> Iterator[str]:
    """Возвращает имена модулей, импортируемых файлом.

    Относительные импорты пропускаются: они по определению не выводят за пределы
    собственного пакета, а сам их запрет обеспечивает ruff.

    Args:
        module_path: Путь к файлу с исходным кодом Python.

    Yields:
        Полное имя импортируемого модуля.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


@pytest.mark.parametrize("layer", LAYER_PACKAGES)
def test_all_layer_packages_exist(layer: str) -> None:
    marker = PACKAGE_ROOT / layer / "__init__.py"
    assert marker.is_file(), f"нет пакета слоя {layer}: ожидался {marker}"


def test_package_ships_py_typed_marker() -> None:
    marker = PACKAGE_ROOT / "py.typed"
    assert marker.is_file(), (
        "без маркера py.typed аннотации пакета не видны потребителям (PEP 561)"
    )


def test_domain_package_has_no_third_party_imports() -> None:
    domain_root = PACKAGE_ROOT / "domain"
    assert domain_root.is_dir(), f"пакет domain не создан: ожидался {domain_root}"

    offenders: list[str] = []
    for module_path in sorted(domain_root.rglob("*.py")):
        relative = module_path.relative_to(SERVICE_ROOT).as_posix()
        for imported in _iter_imported_modules(module_path):
            top_level = imported.split(".", maxsplit=1)[0]
            if top_level == "document_worker":
                if not imported.startswith("document_worker.domain"):
                    offenders.append(f"{relative}: {imported}")
            elif top_level not in sys.stdlib_module_names:
                offenders.append(f"{relative}: {imported}")

    assert not offenders, "domain обязан зависеть только от stdlib: " + ", ".join(
        offenders
    )
