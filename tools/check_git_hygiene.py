"""check_git_hygiene.py — Firelight git-hygiene gate.

Keeps the branch list tidy: fails when a local branch (or, when enabled, an
``origin`` remote-tracking branch) is fully merged into the default branch yet
still lingers. Old merged branches are noise — once their work is on the default
branch they should be deleted so ``git branch -a`` and the GitHub branch list
stay clean for the next agent.

Run standalone (exit 1 on any stale branch):
    python tools/check_git_hygiene.py

Detection. A branch counts as *merged & stale* when its tip is an ancestor of
the default branch (ordinary merge / fast-forward) OR every one of its commits
already has an equivalent change in the default branch by patch-id
(``git cherry`` — this catches squash- and rebase-merges, which leave no
ancestry link). The current branch and the configured default branches are
never flagged.

Graceful skip (exit 0). Outside a git work tree, on a shallow clone, or when no
default branch can be resolved, the gate cannot reason about merge state, so it
prints ``SKIP`` and passes rather than failing spuriously (e.g. in a CI checkout).

All knobs come from ``[tool.firelight.git]`` via :mod:`tools.standards_config` —
this script holds zero magic values.

Docs: docs/systems/standards.md#git-hygiene
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

# Make ``tools`` importable whether launched as a file or via ``-m``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.standards_config import REPO_ROOT, StandardsConfig, force_utf8, load_config

_DELEGATE = (
    "Delete the stale merged branches so the gate stays green:\n"
    "    git branch -d <branch>             # local (already merged)\n"
    "    git push origin --delete <branch>  # remote\n"
    "    git remote prune origin            # drop stale remote-tracking refs\n"
    "Then run `pytest -q tests/standards/test_git_hygiene.py` to confirm green and return."
)


def _git(*args: str) -> tuple[int, str]:
    """Run a ``git`` sub-command from the repo root; return (exit_code, stripped_output)."""
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _is_work_tree() -> bool:
    """Whether the repo root is inside a git work tree."""
    code, out = _git("rev-parse", "--is-inside-work-tree")
    return code == 0 and out == "true"


def _is_shallow() -> bool:
    """Whether this is a shallow clone (merge state is unreliable, so we skip)."""
    code, out = _git("rev-parse", "--is-shallow-repository")
    return code == 0 and out == "true"


def _ref_exists(ref: str) -> bool:
    """Whether ``ref`` resolves to a commit."""
    code, _ = _git("rev-parse", "--verify", "--quiet", ref)
    return code == 0


def _short(ref: str) -> str:
    """Strip the ``refs/heads/`` / ``refs/remotes/`` namespace for display."""
    for prefix in ("refs/heads/", "refs/remotes/"):
        if ref.startswith(prefix):
            return ref[len(prefix) :]
    return ref


def _resolve_compare_ref(cfg: StandardsConfig) -> str | None:
    """First existing default-branch ref (local preferred, then ``origin``), or ``None``."""
    for name in cfg.git_default_branches:
        for ref in (f"refs/heads/{name}", f"refs/remotes/origin/{name}"):
            if _ref_exists(ref):
                return ref
    return None


def _current_branch() -> str:
    """Short name of the checked-out branch, or ``""`` when HEAD is detached."""
    code, out = _git("symbolic-ref", "--quiet", "--short", "HEAD")
    return out if code == 0 else ""


def _is_merged(branch: str, compare_ref: str) -> bool:
    """Whether ``branch`` is fully contained in ``compare_ref`` (ancestry or patch-id)."""
    if _git("merge-base", "--is-ancestor", branch, compare_ref)[0] == 0:
        return True
    # Squash / rebase merge leaves no ancestry link: every unique commit on the
    # branch must already have an equivalent change upstream (git cherry '-').
    code, out = _git("cherry", compare_ref, branch)
    if code != 0:
        return False
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("-") for ln in lines)


def _branch_names(namespace: str) -> list[str]:
    """Short branch names under a ref namespace (e.g. ``refs/heads``)."""
    code, out = _git("for-each-ref", "--format=%(refname:short)", namespace)
    if code != 0:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _stale_in(
    names: list[str],
    compare_ref: str,
    cfg: StandardsConfig,
    current: str,
    *,
    is_remote: bool,
) -> list[str]:
    """Violations for one namespace: merged branches that should have been deleted."""
    out: list[str] = []
    kind = "remote" if is_remote else "local"
    for short in names:
        if is_remote:
            if "/" not in short:  # the bare 'origin' -> refs/remotes/origin/HEAD pointer
                continue
            logical = short.split("/", 1)[1]
        else:
            logical = short
        if logical == "HEAD" or logical in cfg.git_default_branches:
            continue
        if not is_remote and short == current:
            continue
        if any(fnmatch.fnmatch(logical, pat) for pat in cfg.git_protected):
            continue
        if _is_merged(short, compare_ref):
            out.append(
                f"[git] {kind} branch '{short}' is fully merged into "
                f"'{_short(compare_ref)}' but still exists - delete it."
            )
    return out


def collect_violations(cfg: StandardsConfig) -> list[str] | None:
    """Stale merged branches (empty == clean); ``None`` means the gate skipped."""
    if not _is_work_tree() or _is_shallow():
        return None
    compare_ref = _resolve_compare_ref(cfg)
    if compare_ref is None:
        return None
    current = _current_branch()
    violations = _stale_in(_branch_names("refs/heads"), compare_ref, cfg, current, is_remote=False)
    if cfg.git_check_remotes:
        violations += _stale_in(
            _branch_names("refs/remotes/origin"), compare_ref, cfg, current, is_remote=True
        )
    return violations


def main() -> int:
    """CLI entry point: print stale branches and return exit code (0 clean/skip, 1 dirty)."""
    force_utf8()
    cfg = load_config()
    violations = collect_violations(cfg)
    if violations is None:
        print("SKIP: git-hygiene gate - no git work tree / shallow clone / no default branch.")
        return 0
    if not violations:
        print("OK: git-hygiene gate clean - no stale merged branches.")
        return 0
    print(f"FAIL: git-hygiene gate - {len(violations)} stale branch(es):\n")
    for line in violations:
        print(f"  {line}")
    print(f"\n{_DELEGATE}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
