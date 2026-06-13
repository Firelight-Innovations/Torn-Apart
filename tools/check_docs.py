"""check_docs.py — Firelight documentation gate.

A separate, independently-failing check that the grep-first ``docs/`` knowledge
base stays complete and that code points back to it. Pairs with
``mkdocs build --strict`` (run from the docs pytest gate) which independently
catches dead nav/links.

Standards enforced (see docs/systems/standards.md):
  * 12 — every public module/class/function docstring carries a resolvable
         ``Docs: docs/systems/<path>.md[#anchor]`` pointer; dead links fail.
  * 13 — every package has its ``docs/systems/<mirrored-path>.md`` with the
         canonical H2 schema (Role / Public API / Imports Allowed / Events /
         Units & Invariants / Examples / Gotchas) and a ``keywords:`` line.

Run standalone (exit 1 on any violation):
    python tools/check_docs.py

Docs: docs/systems/standards.md#documentation
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Make ``tools`` importable whether launched as a file or via ``-m``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.standards_config import REPO_ROOT, StandardsConfig, force_utf8, load_config

REQUIRED_H2: tuple[str, ...] = (
    "Role",
    "Public API",
    "Imports Allowed",
    "Events",
    "Units & Invariants",
    "Examples",
    "Gotchas",
)
"""The canonical H2 headings every ``docs/systems/*.md`` must contain (CLAUDE.md)."""

_POINTER_RE = re.compile(r"Docs:\s*(docs/systems/[\w/.\-]+\.md)(#[\w\-]+)?")
_MAX_SHOWN = 40  # cap displayed lines; full count is always reported (no silent truncation).

_DELEGATE = (
    "Do NOT fix these inline. Spin up a sub-agent scoped per package: bring its "
    "docs/systems/ doc to the H2 schema and add the `Docs:` pointers to its public "
    "docstrings (same commit as any API change), then run "
    "`pytest -q tests/standards/test_docs.py` to confirm green and return."
)


def _slug(heading: str) -> str:
    """Slugify a markdown heading the way MkDocs/GitHub anchors do."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s\-]", "", text)
    return re.sub(r"[\s_]+", "-", text)


def _doc_path_for(pkg_dir: Path, cfg: StandardsConfig) -> Path:
    """Map a package directory to its mirrored ``docs/systems/<path>.md``.

    ``fire_engine/core`` -> ``docs/systems/core.md``;
    ``fire_engine/procedural/textures`` -> ``docs/systems/procedural/textures.md``.
    """
    rel = pkg_dir.relative_to(REPO_ROOT)
    inner = Path(*rel.parts[1:])  # drop the source-root package component
    return REPO_ROOT / cfg.docs_root / inner.with_suffix(".md")


def _packages(cfg: StandardsConfig) -> list[Path]:
    """All sub-package directories (have ``__init__.py``) under the source roots.

    The source-root package itself (e.g. ``fire_engine/``) is excluded — it only
    re-exports — so the top-level doc set stays one-per-subsystem.
    """
    out: list[Path] = []
    for root_name in cfg.source_roots:
        root = REPO_ROOT / root_name
        for init in root.rglob("__init__.py"):
            pkg = init.parent
            if pkg == root:
                continue
            if cfg.is_excluded(pkg.relative_to(REPO_ROOT)):
                continue
            out.append(pkg)
    return sorted(out)


def _check_doc_schema(doc: Path) -> list[str]:
    """Validate one system doc has the keywords line and every required H2."""
    rel = doc.relative_to(REPO_ROOT).as_posix()
    text = doc.read_text(encoding="utf-8")
    out: list[str] = []
    if not re.search(r"(?im)^keywords:", text):
        out.append(f"[13] {rel}: missing `keywords:` line.")
    headings = {m.group(1).strip() for m in re.finditer(r"(?m)^##\s+(.+?)\s*$", text)}
    out.extend(
        f"[13] {rel}: missing required H2 section `## {required}`."
        for required in REQUIRED_H2
        if required not in headings
    )
    return out


def _anchor_exists(doc: Path, anchor: str) -> bool:
    """Whether ``#anchor`` resolves to an H2/H3 heading slug in ``doc``."""
    text = doc.read_text(encoding="utf-8")
    slugs = {_slug(m.group(1)) for m in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", text)}
    return anchor.lstrip("#") in slugs


def _is_public_func(node: ast.AST) -> bool:
    """Whether ``node`` is a public (non-underscore) function/method definition."""
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith(
        "_"
    )


def _public_doc_targets(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """(label, node) for every public module/class/function needing a pointer."""
    targets: list[tuple[str, ast.AST]] = [("module", tree)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            targets.append((f"class {node.name}", node))
            targets.extend(
                (f"method {node.name}.{sub.name}", sub) for sub in node.body if _is_public_func(sub)
            )
        elif _is_public_func(node):
            targets.append((f"function {node.name}", node))  # type: ignore[attr-defined]
    return targets


def _check_pointers(module: Path) -> list[str]:
    """Standard 12: each public docstring carries a resolvable Docs: pointer."""
    rel = module.relative_to(REPO_ROOT).as_posix()
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"[parse] {rel}: {exc}"]

    out: list[str] = []
    for label, node in _public_doc_targets(tree):
        doc = ast.get_docstring(node)  # type: ignore[arg-type]
        if doc is None:
            out.append(f"[12] {rel}: {label} has no docstring (needs a `Docs:` pointer).")
            continue
        match = _POINTER_RE.search(doc)
        if match is None:
            out.append(f"[12] {rel}: {label} docstring lacks a `Docs: docs/systems/...md` pointer.")
            continue
        target = REPO_ROOT / match.group(1)
        if not target.exists():
            out.append(f"[12] {rel}: {label} points to missing doc `{match.group(1)}`.")
        elif match.group(2) and not _anchor_exists(target, match.group(2)):
            out.append(f"[12] {rel}: {label} dead anchor `{match.group(2)}` in `{match.group(1)}`.")
    return out


def _source_modules(cfg: StandardsConfig) -> list[Path]:
    """All non-``__init__`` ``*.py`` modules under the source roots."""
    out: list[Path] = []
    for root_name in cfg.source_roots:
        root = REPO_ROOT / root_name
        for path in root.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            if cfg.is_excluded(path.relative_to(REPO_ROOT)):
                continue
            out.append(path)
    return out


def collect_violations(cfg: StandardsConfig) -> list[str]:
    """Return all documentation violations (standards 12 & 13); empty == clean."""
    violations: list[str] = []

    # Standard 13 — package docs exist and follow the schema.
    for pkg in _packages(cfg):
        doc = _doc_path_for(pkg, cfg)
        rel_pkg = pkg.relative_to(REPO_ROOT).as_posix()
        if not doc.exists():
            violations.append(
                f"[13] package {rel_pkg}/ has no doc (expected "
                f"{doc.relative_to(REPO_ROOT).as_posix()})."
            )
            continue
        violations.extend(_check_doc_schema(doc))

    # Standard 12 — code points back to its docs.
    for module in _source_modules(cfg):
        violations.extend(_check_pointers(module))

    return violations


def main() -> int:
    """CLI entry point: print violations (capped) and return 0/1."""
    force_utf8()
    cfg = load_config()
    violations = collect_violations(cfg)
    if not violations:
        print("OK: Docs gate clean.")
        return 0
    print(f"FAIL: Docs gate - {len(violations)} violation(s):\n")
    for line in violations[:_MAX_SHOWN]:
        print(f"  {line}")
    if len(violations) > _MAX_SHOWN:
        print(f"  ... and {len(violations) - _MAX_SHOWN} more (showing first {_MAX_SHOWN}).")
    print(f"\n{_DELEGATE}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
