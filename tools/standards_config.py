"""standards_config.py — single source of truth for the Firelight standards gate.

Loads the structure/docs/coverage limits from ``[tool.firelight]`` in
``pyproject.toml`` so every custom check (``check_repo_structure.py``,
``check_docs.py``) and the pytest gates read identical numbers. Tune a limit in
exactly one place — ``pyproject.toml`` — and the whole gate moves with it.

Docs: docs/systems/standards.md#configuration

Example:
    >>> from tools.standards_config import load_config
    >>> cfg = load_config()
    >>> cfg.max_modules_per_dir
    10
"""

from __future__ import annotations

import contextlib
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Absolute path to the repository root (the parent of ``tools/``)."""


def force_utf8() -> None:
    """Force stdout/stderr to UTF-8 so emoji/em-dash output survives Windows cp1252.

    No-op where the streams can't be reconfigured (e.g. already wrapped).

    Docs: docs/systems/standards.md#configuration
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class StandardsConfig:
    """Resolved limits for the standards gate, read from ``[tool.firelight]``.

    All paths are repo-root-relative strings exactly as written in
    ``pyproject.toml``; resolve them against :data:`REPO_ROOT` when walking.

    Docs: docs/systems/standards.md#configuration
    """

    source_roots: tuple[str, ...]
    test_root: str
    docs_root: str
    max_subdirs: int
    max_modules_per_dir: int
    max_lines: int
    coverage_fail_under: float
    grouping_modules: tuple[str, ...]
    exclude: tuple[str, ...] = field(default_factory=tuple)

    def is_excluded(self, path: Path) -> bool:
        """Return ``True`` if ``path`` falls under any configured exclude prefix.

        ``path`` may be absolute or relative; it is compared by path *parts*
        against the configured exclude names, so ``saves`` matches
        ``fire_engine/saves`` and a top-level ``saves/`` alike.
        """
        parts = set(path.parts)
        return any(token in parts for token in self.exclude)


def load_config(pyproject: Path | None = None) -> StandardsConfig:
    """Parse ``[tool.firelight]`` from ``pyproject.toml`` into a config object.

    Args:
        pyproject: Override path to ``pyproject.toml`` (defaults to repo root).

    Returns:
        A frozen :class:`StandardsConfig`.

    Raises:
        FileNotFoundError: if ``pyproject.toml`` is missing.
        KeyError: if the ``[tool.firelight]`` table is absent.
    """
    path = pyproject or (REPO_ROOT / "pyproject.toml")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    fl = data["tool"]["firelight"]
    return StandardsConfig(
        source_roots=tuple(fl["source_roots"]),
        test_root=str(fl["test_root"]),
        docs_root=str(fl["docs_root"]),
        max_subdirs=int(fl["max_subdirs"]),
        max_modules_per_dir=int(fl["max_modules_per_dir"]),
        max_lines=int(fl["max_lines"]),
        coverage_fail_under=float(fl.get("coverage_fail_under", 0.0)),
        grouping_modules=tuple(fl.get("grouping_modules", ())),
        exclude=tuple(fl.get("exclude", ())),
    )


if __name__ == "__main__":
    cfg = load_config()
    for name in (
        "source_roots",
        "test_root",
        "docs_root",
        "max_subdirs",
        "max_modules_per_dir",
        "max_lines",
        "coverage_fail_under",
        "grouping_modules",
    ):
        print(f"{name:22} = {getattr(cfg, name)!r}")
    sys.exit(0)
