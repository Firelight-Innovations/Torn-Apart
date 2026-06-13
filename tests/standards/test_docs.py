"""Docs gate — code points to its docs, docs follow the schema, nav builds clean.

Standards D (12–15): runs ``tools/check_docs.py`` and ``mkdocs build --strict``
as two independent checks. Headless.

Docs: docs/systems/standards.md#documentation
"""

from __future__ import annotations

from tests.standards._runner import REPO_ROOT, fail_message, py, run_tool

_DELEGATE = (
    "Do NOT fix inline. Spin up a sub-agent per package to bring its docs/systems/ "
    "doc to the H2 schema and add resolvable `Docs:` pointers to its public "
    "docstrings (same commit as any API change), then run "
    "`pytest -q tests/standards/test_docs.py` to confirm green and return."
)


def test_doc_pointers_and_schema() -> None:
    """Every package has a schema-valid doc; every public docstring has a live pointer."""
    script = REPO_ROOT / "tools" / "check_docs.py"
    result = run_tool("check_docs", py(str(script)))
    assert result.ok, fail_message(result, _DELEGATE)


def test_mkdocs_build_strict() -> None:
    """`mkdocs build --strict` — free broken-link/nav checker (standard 15)."""
    result = run_tool(
        "mkdocs build --strict",
        py("-m", "mkdocs", "build", "--strict", "--site-dir", "site"),
    )
    assert result.ok, fail_message(result, _DELEGATE)
