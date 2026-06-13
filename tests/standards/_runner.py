"""Shared subprocess helper for the standards gates.

Centralises "run a tool from the repo root, capture everything, build a
delegate-to-sub-agent failure message" so the four gate modules stay DRY (and
don't trip the duplicate-code check themselves).

Docs: docs/systems/standards.md#integration
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root (``tests/standards/_runner.py`` -> two parents up)."""

_MAX_OUTPUT_CHARS = 6000  # keep assertion text readable; the tail holds the summary.


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one gate sub-process."""

    name: str
    returncode: int
    output: str

    @property
    def ok(self) -> bool:
        """Whether the tool exited 0."""
        return self.returncode == 0


def run_tool(name: str, args: list[str]) -> ToolResult:
    """Run ``args`` from the repo root with the current interpreter's environment.

    A leading ``"-m"`` style invocation should pass ``sys.executable`` explicitly
    via :func:`py`; ``args`` is used verbatim as argv.
    """
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return ToolResult(name=name, returncode=proc.returncode, output=proc.stdout + proc.stderr)


def py(*args: str) -> list[str]:
    """Build an argv invoking the active interpreter (the venv's python under pytest)."""
    return [sys.executable, *args]


def fail_message(result: ToolResult, delegate: str) -> str:
    """Compose a capped, delegate-to-sub-agent assertion message."""
    body = result.output.strip()
    if len(body) > _MAX_OUTPUT_CHARS:
        body = "...(truncated)...\n" + body[-_MAX_OUTPUT_CHARS:]
    return (
        f"\nFAIL: standards gate '{result.name}' (exit {result.returncode}).\n\n"
        f"{body}\n\n"
        f"=> DELEGATE: {delegate}\n"
    )
