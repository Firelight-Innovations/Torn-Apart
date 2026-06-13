"""Coverage gate — branch-coverage ratchet (standard 18).

Re-runs the headless suite under ``coverage`` and enforces
``--cov-fail-under`` = ``[tool.firelight] coverage_fail_under``. This is the
heavy gate, so it is marked ``coverage`` and deselected from the default run
(pytest.ini); CI/nightly runs ``pytest -m coverage``. ``tests/standards`` is
excluded from the inner run to avoid recursion.

The threshold is a ratchet: set to the current measured branch coverage
(rounded down) and only ever raised. Standard 17 (every module has a test) keeps
new code honest while the floor climbs. See DECISIONS.md "coverage ratchet".

Docs: docs/systems/standards.md#coverage
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``tools`` importable when pytest's rootdir isn't on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.standards._runner import fail_message, py, run_tool
from tools.standards_config import load_config

_DELEGATE = (
    "Do NOT lower the floor. Coverage dropped below the ratchet — spin up a "
    "sub-agent to add the missing unit/property tests for the uncovered branches "
    "(hypothesis encouraged), then run `pytest -m coverage` to confirm green. "
    "Raise the floor in pyproject [tool.firelight] only after it is comfortably met."
)


@pytest.mark.coverage
def test_branch_coverage_ratchet() -> None:
    """Suite branch coverage must meet the configured ratchet floor."""
    floor = load_config().coverage_fail_under
    result = run_tool(
        "coverage ratchet",
        py(
            "-m",
            "pytest",
            "--ignore=tests/standards",
            "--cov=fire_engine",
            "--cov-branch",
            "--cov-report=term-missing:skip-covered",
            f"--cov-fail-under={floor}",
            "-q",
            "-p",
            "no:cacheprovider",
        ),
    )
    assert result.ok, fail_message(result, _DELEGATE)
