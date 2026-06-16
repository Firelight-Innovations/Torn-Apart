"""check_repo_structure.py — Firelight repo-structure & file-content gate.

Enforces the "deep & narrow, one idea per file" standards that no off-the-shelf
tool covers. Every limit comes from ``[tool.firelight]`` via
:mod:`tools.standards_config` — this script holds zero magic numbers.

Standards enforced (see docs/systems/standards.md):
  * 6  — max sub-folders per folder (the source root itself is exempt — it is the
         engine subsystem index, not a leaf package; matches check_docs.py).
  * 7  — max Python modules per folder (excluding ``__init__.py``).
  * 10 — one public top-level class per module (with §C exemptions).
  * 11 — one cohesive responsibility per module (module docstring required;
         grouping modules ``events.py`` / ``types.py`` … are exempt from 10).
  * 17 — every public source module has a matching test module (mirrored path,
         legacy flat ``tests/test_<stem>.py`` also accepted). Modules that import
         panda3d are exempt — Hard Rule 1 bars them from the headless suite, so a
         headless mirror is impossible; they are integration-verified at app launch.

Run standalone (exit 1 on any violation):
    python tools/check_repo_structure.py

Docs: docs/systems/standards.md#repo-structure
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Make ``tools`` importable whether launched as a file or via ``-m``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.standards_config import REPO_ROOT, StandardsConfig, force_utf8, load_config

_DELEGATE = (
    "Do NOT fix these inline - it would blow the orchestrator's context. Spin up "
    "ONE sub-agent per offending folder/file, scoped to split the package/file "
    "along real responsibility seams (updating the matching docs/systems/ doc in "
    "the same change), then run `pytest -q tests/standards/test_repo_structure.py` "
    "to confirm green and return."
)


def _iter_dirs(root: Path, cfg: StandardsConfig) -> list[Path]:
    """Yield ``root`` and every non-excluded sub-directory beneath it."""
    out: list[Path] = []
    for path in [root, *root.rglob("*")]:
        if not path.is_dir():
            continue
        if cfg.is_excluded(path.relative_to(REPO_ROOT)):
            continue
        out.append(path)
    return out


def _subdirs(directory: Path, cfg: StandardsConfig) -> list[Path]:
    """Immediate non-excluded child directories of ``directory``."""
    return [
        child
        for child in directory.iterdir()
        if child.is_dir() and not cfg.is_excluded(child.relative_to(REPO_ROOT))
    ]


def _modules(directory: Path) -> list[Path]:
    """Immediate ``*.py`` modules in ``directory``, excluding ``__init__.py``."""
    return [
        child
        for child in directory.iterdir()
        if child.is_file() and child.suffix == ".py" and child.name != "__init__.py"
    ]


def _public_top_level_classes(tree: ast.Module) -> list[str]:
    """Names of public (non-underscore) classes defined at module top level."""
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def _imports_panda3d(tree: ast.Module) -> bool:
    """True if the module imports ``panda3d`` or ``direct`` (the Panda3D SDK).

    Such a module is, by Hard Rule 1, a ``render/``/``lighting/`` bridge that the
    headless test suite *excludes by construction* (importing panda3d at collection
    would need a GPU/window). It therefore cannot have a headless test mirror and is
    exempt from standard 17 — it is integration-verified by launching the app, not
    unit-mirrored. A pure module under ``render/`` that does NOT import panda3d (e.g.
    a GLSL-string builder) is still headless-testable and still requires its mirror.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in ("panda3d", "direct") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in ("panda3d", "direct"):
                return True
    return False


def _test_paths_for(module: Path, cfg: StandardsConfig) -> list[Path]:
    """Candidate test-module paths for a source ``module`` (any one satisfies).

    Canonical: mirrored path under ``test_root`` with a ``test_`` prefix, e.g.
    ``fire_engine/core/rng.py`` -> ``tests/core/test_rng.py``. Legacy flat
    ``tests/test_rng.py`` is also accepted so existing tests are recognised.
    """
    rel = module.relative_to(REPO_ROOT)
    # Drop the leading source-root package component (e.g. "fire_engine/").
    inner = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
    mirrored = REPO_ROOT / cfg.test_root / inner.parent / f"test_{module.stem}.py"
    flat = REPO_ROOT / cfg.test_root / f"test_{module.stem}.py"
    return [mirrored, flat]


def collect_violations(cfg: StandardsConfig) -> list[str]:
    """Return a list of human-readable structure/content violations (empty == clean)."""
    violations: list[str] = []

    for root_name in cfg.source_roots:
        root = REPO_ROOT / root_name
        if not root.exists():
            violations.append(f"source root '{root_name}' does not exist")
            continue

        for directory in _iter_dirs(root, cfg):
            rel_dir = directory.relative_to(REPO_ROOT).as_posix()

            # The source root itself (e.g. ``fire_engine/``) is the engine's
            # subsystem index, governed by ARCHITECTURE.md — not a leaf package the
            # narrowness cap is meant for. check_docs.py already exempts it (``if
            # pkg == root: continue``); mirror that here so the two checkers agree.
            n_subdirs = len(_subdirs(directory, cfg))
            if directory != root and n_subdirs > cfg.max_subdirs:
                violations.append(
                    f"[6] {rel_dir}/ has {n_subdirs} sub-folders (max {cfg.max_subdirs}) "
                    f"- introduce a new sub-package instead of widening."
                )

            modules = _modules(directory)
            if len(modules) > cfg.max_modules_per_dir:
                violations.append(
                    f"[7] {rel_dir}/ has {len(modules)} modules (max "
                    f"{cfg.max_modules_per_dir}) - split into a sub-package."
                )

            for module in modules:
                violations.extend(_check_module(module, cfg))

    return violations


def _check_module(module: Path, cfg: StandardsConfig) -> list[str]:
    """Per-module checks: docstring, one public class, matching test (10, 11, 17)."""
    out: list[str] = []
    rel = module.relative_to(REPO_ROOT).as_posix()
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"[parse] {rel}: {exc}"]

    if ast.get_docstring(tree) is None:
        out.append(f"[11] {rel}: missing module docstring (one responsibility per file).")

    if module.name not in cfg.grouping_modules:
        public = _public_top_level_classes(tree)
        if len(public) > 1:
            out.append(
                f"[10] {rel}: defines {len(public)} public classes {public} (max 1). "
                f"Move the unrelated class(es) into their own module, or - if these are "
                f"trivial support types - into a dedicated {cfg.grouping_modules}."
            )

    # Standard 17 is a *headless*-test-mirror rule. A module that imports panda3d
    # is excluded from the headless suite by Hard Rule 1, so it cannot have one —
    # exempt it (it is integration-verified via the app launch instead).
    if not _imports_panda3d(tree) and not any(p.exists() for p in _test_paths_for(module, cfg)):
        canonical = _test_paths_for(module, cfg)[0].relative_to(REPO_ROOT).as_posix()
        out.append(f"[17] {rel}: no test module (expected {canonical}).")

    return out


def main() -> int:
    """CLI entry point: print violations and return exit code (0 clean, 1 dirty)."""
    force_utf8()
    cfg = load_config()
    violations = collect_violations(cfg)
    if not violations:
        print("OK: Repo-structure gate clean.")
        return 0
    print(f"FAIL: Repo-structure gate - {len(violations)} violation(s):\n")
    for line in violations:
        print(f"  {line}")
    print(f"\n{_DELEGATE}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
