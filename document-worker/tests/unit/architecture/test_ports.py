"""Сторожа границ слоя application."""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "document_worker"
DOMAIN_ROOT = PACKAGE_ROOT / "domain"
APPLICATION_ROOT = PACKAGE_ROOT / "application"
PORTS_ROOT = APPLICATION_ROOT / "ports"
COMMANDS_MODULE = APPLICATION_ROOT / "dto" / "commands.py"


def _protocol_names(root: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names.extend(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id == "Protocol"
                for base in node.bases
            )
        )
    return names


def _dataclass_names(path: Path) -> list[ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def test_no_protocol_name_is_declared_in_two_layers() -> None:
    names = _protocol_names(DOMAIN_ROOT) + _protocol_names(APPLICATION_ROOT)

    duplicates = [name for name, count in Counter(names).items() if count > 1]

    assert not duplicates, f"имя Protocol объявлено дважды: {duplicates}"


def test_ports_are_runtime_checkable_protocols() -> None:
    assert PORTS_ROOT.is_dir(), f"портов нет: {PORTS_ROOT}"

    without_decorator: list[str] = []
    for path in sorted(PORTS_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_protocol = any(
                isinstance(base, ast.Name) and base.id == "Protocol"
                for base in node.bases
            )
            decorated = any(
                isinstance(item, ast.Name) and item.id == "runtime_checkable"
                for item in node.decorator_list
            )
            if is_protocol and not decorated:
                without_decorator.append(f"{path.name}:{node.name}")

    assert not without_decorator, f"порт без runtime_checkable: {without_decorator}"


def test_every_command_declares_correlation_id() -> None:
    assert COMMANDS_MODULE.is_file(), f"команд нет: {COMMANDS_MODULE}"

    without_correlation: list[str] = []
    for node in _dataclass_names(COMMANDS_MODULE):
        if not node.name.endswith("Command"):
            continue
        fields = {
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        }
        if "correlation_id" not in fields:
            without_correlation.append(node.name)

    assert not without_correlation, f"команда без correlation_id: {without_correlation}"


def test_application_imports_no_third_party_packages() -> None:
    allowed_roots = {"document_worker"}
    offenders: list[str] = []
    for path in sorted(APPLICATION_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top_level = module.split(".", maxsplit=1)[0]
                if top_level in allowed_roots:
                    continue
                if top_level not in sys.stdlib_module_names:
                    offenders.append(f"{path.name}: {module}")

    assert not offenders, f"application зависит только от stdlib и domain: {offenders}"


def test_application_does_not_import_infrastructure_or_presentation() -> None:
    forbidden = ("document_worker.infrastructure", "document_worker.presentation")
    offenders: list[str] = []
    for path in sorted(APPLICATION_ROOT.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.name}: {prefix}" for prefix in forbidden if prefix in content
        )

    assert not offenders
