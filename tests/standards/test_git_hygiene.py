"""Git-hygiene gate — no stale merged branches linger.

Runs ``tools/check_git_hygiene.py``: fails when a local or ``origin``
remote-tracking branch is fully merged into the default branch yet still exists.
Keeps ``git branch -a`` and the GitHub branch list tidy so the next agent isn't
wading through dead branches. Headless; the checker only shells out to ``git``
and skips gracefully outside a work tree / on a shallow clone.

Docs: docs/systems/standards.md#git-hygiene
"""

from __future__ import annotations

from tests.standards._runner import REPO_ROOT, fail_message, py, run_tool

_DELEGATE = (
    "Delete the stale merged branches: `git branch -d <b>` (local), "
    "`git push origin --delete <b>` (remote), `git remote prune origin`, then run "
    "`pytest -q tests/standards/test_git_hygiene.py` to confirm green and return."
)


def test_git_hygiene() -> None:
    """No local/remote branch fully merged into the default branch still lingers."""
    script = REPO_ROOT / "tools" / "check_git_hygiene.py"
    result = run_tool("check_git_hygiene", py(str(script)))
    assert result.ok, fail_message(result, _DELEGATE)
