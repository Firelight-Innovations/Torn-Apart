"""
render/_impl — Private implementation helpers for the render package.

Functions here are extracted from fat classes to keep each module under 500
lines (C0302).  They take the owning instance as their first argument and are
called from the class via ``_func(self, ...)``.  Not part of the public API.

Docs: docs/systems/render.md
"""
