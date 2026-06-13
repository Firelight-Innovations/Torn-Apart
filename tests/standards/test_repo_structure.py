"""Repo-structure gate — deep & narrow, one idea per file.

Standards B/C (6, 7, 10, 11, 17): runs ``tools/check_repo_structure.py``.
Headless; the checker is a pure AST/tree walk.

Docs: docs/systems/standards.md#repo-structure
"""

from __future__ import annotations

from tests.standards._runner import REPO_ROOT, fail_message, py, run_tool

_DELEGATE = (
    "Do NOT fix inline. Spin up ONE sub-agent per offending folder/file to split "
    "the package/file along real responsibility seams (updating the matching "
    "docs/systems/ doc in the same change), then run "
    "`pytest -q tests/standards/test_repo_structure.py` to confirm green and return."
)


def test_repo_structure() -> None:
    """<=5 sub-folders & <=10 modules per dir, one public class per module, test mirror."""
    script = REPO_ROOT / "tools" / "check_repo_structure.py"
    result = run_tool("check_repo_structure", py(str(script)))
    assert result.ok, fail_message(result, _DELEGATE)
