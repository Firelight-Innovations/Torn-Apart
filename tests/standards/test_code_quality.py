"""Code-quality gate — ruff (lint + format), mypy --strict, pylint, vulture.

Standards A (1–5): everything typed, no dead code, no duplicate code, consistent
formatting, performance hygiene. Each tool is its own test so they fail
independently. All shell out; none import panda3d.

Docs: docs/systems/standards.md#code-quality
"""

from __future__ import annotations

from tests.standards._runner import fail_message, py, run_tool

_DELEGATE = (
    "Do NOT fix inline. Spin up a sub-agent scoped to the offending package, have "
    "it clear the reported lint/type/dead-code/duplication findings (downward "
    "imports only; no random.*/unseeded np.random/pickle/per-voxel loops), then run "
    "`pytest -q tests/standards/test_code_quality.py` to confirm green and return."
)


def test_ruff_lint() -> None:
    """Ruff lint: dead imports/vars, commented-out code, complexity, perf, numpy lints."""
    result = run_tool("ruff check", py("-m", "ruff", "check", "."))
    assert result.ok, fail_message(result, _DELEGATE)


def test_ruff_format() -> None:
    """Ruff format is the sole formatter; unformatted code fails (standard 4)."""
    result = run_tool("ruff format --check", py("-m", "ruff", "format", "--check", "."))
    assert result.ok, fail_message(result, _DELEGATE)


def test_mypy_strict() -> None:
    """mypy --strict owns typing/annotation coverage (standard 1)."""
    result = run_tool("mypy --strict", py("-m", "mypy"))
    assert result.ok, fail_message(result, _DELEGATE)


def test_pylint_duplication_and_module_length() -> None:
    """pylint (narrow): duplicate-code (R0801) + too-many-lines/500 (standards 3, 8)."""
    result = run_tool(
        "pylint duplicate-code,too-many-lines",
        py(
            "-m",
            "pylint",
            "--disable=all",
            "--enable=duplicate-code,too-many-lines",
            "fire_engine",
        ),
    )
    assert result.ok, fail_message(result, _DELEGATE)


def test_vulture_dead_code() -> None:
    """vulture: cross-module dead code (standard 2). Whitelist lives in vulture_whitelist.py."""
    result = run_tool("vulture", py("-m", "vulture", "fire_engine", "vulture_whitelist.py"))
    assert result.ok, fail_message(result, _DELEGATE)
