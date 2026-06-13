"""Firelight standards gate — independently-failing pytest checks.

Each module here shells out to one analyzer (ruff, mypy, pylint, vulture, the
custom structure/docs checks, mkdocs, coverage) and asserts it passes. A
standards violation fails the build exactly like a failing test. All modules
stay headless — they shell out, they never import panda3d.

Docs: docs/systems/standards.md
"""
