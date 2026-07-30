"""Тесты контекста логирования."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from document_worker.application.context import (
    current_logging_context,
    logging_context,
)

pytestmark = pytest.mark.unit

APPLICATION_ROOT = Path(__file__).resolve().parents[3] / "document_worker/application"
CONTEXT_MODULE = "context.py"


def test_context_is_empty_outside_any_block() -> None:
    assert current_logging_context() == {}


def test_values_are_visible_inside_block() -> None:
    with logging_context(correlation_id="req-1", document_id="doc-1"):
        assert current_logging_context() == {
            "correlation_id": "req-1",
            "document_id": "doc-1",
        }


def test_logging_context_resets_on_exit() -> None:
    with logging_context(correlation_id="req-1"):
        pass

    assert current_logging_context() == {}


def test_logging_context_resets_after_error() -> None:
    with pytest.raises(RuntimeError), logging_context(correlation_id="req-1"):
        raise RuntimeError

    assert current_logging_context() == {}


def test_nested_block_restores_outer_values() -> None:
    with logging_context(correlation_id="outer"):
        with logging_context(correlation_id="inner"):
            assert current_logging_context()["correlation_id"] == "inner"
        assert current_logging_context()["correlation_id"] == "outer"


def test_missing_values_are_omitted() -> None:
    with logging_context(correlation_id="req-1"):
        assert "document_id" not in current_logging_context()


def test_no_module_outside_logging_reads_context_variables() -> None:
    # Источник истины для correlation_id — явная передача в команде; ContextVar
    # нужен только логам.
    offenders: list[str] = []
    for path in sorted(APPLICATION_ROOT.rglob("*.py")):
        if path.name == CONTEXT_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Name) and node.id == "ContextVar"
            for node in ast.walk(tree)
        ):
            offenders.append(path.name)

    assert not offenders, f"ContextVar вне модуля логирования: {offenders}"
