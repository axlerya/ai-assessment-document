"""Сторожа границ слоя политик."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

POLICIES_ROOT = Path(__file__).resolve().parents[4] / "document_worker/domain/policies"

# Структурные значения: индексы, признак пустоты, границы единичного отрезка.
# Настоящий порог таким быть не может, поэтому они не мешают проверке.
STRUCTURAL_NUMBERS = frozenset({0, 1, -1})

# Сущности политика не конструирует: она возвращает вердикт.
FORBIDDEN_IMPORTS = ("document_worker.domain.entities",)


def _policy_modules() -> list[Path]:
    modules = sorted(POLICIES_ROOT.glob("*.py"))
    assert modules, f"нет модулей политик: {POLICIES_ROOT}"
    return modules


def test_policies_do_not_import_entities() -> None:
    offenders: list[str] = []
    for path in _policy_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            else:
                continue
            if module.startswith(FORBIDDEN_IMPORTS):
                offenders.append(f"{path.name}: {module}")

    assert not offenders, (
        f"политика возвращает вердикт, а не строит сущности: {offenders}"
    )


def test_no_magic_numbers_in_policy_bodies() -> None:
    offenders: list[str] = []
    for path in _policy_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            offenders.extend(
                f"{path.name}:{inner.lineno} {inner.value}"
                for inner in ast.walk(node)
                if isinstance(inner, ast.Constant)
                and isinstance(inner.value, int | float)
                and not isinstance(inner.value, bool)
                and inner.value not in STRUCTURAL_NUMBERS
            )

    assert not offenders, (
        f"пороги живут полями политики, а не в теле метода: {offenders}"
    )
