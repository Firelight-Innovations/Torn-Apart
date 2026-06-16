"""
devtools/fields.py — the declarative panel/field model shared by every dev tool.

This is the *contract* between the headless dev-tools layer and whatever draws
the panels on screen (today: ``world/devtools_overlay.py`` using Panda3D
DirectGUI; tomorrow possibly Dear ImGui — the renderer is swappable because it
only ever consumes the data structures defined here).

A dev tool produces a :class:`Panel`.  A panel is a list of :class:`Section`s
(each a titled group of :class:`Field`s) plus a list of :class:`Button`s
(one-shot actions).  Every :class:`Field` carries **live** ``get``/``set``
callables rather than a snapshot value, so the renderer can poll the current
value each frame (read-only rows update live) and write edits straight back
through the engine's public setters (``set`` is ``None`` ⇒ read-only).

Nothing here imports panda3d (CLAUDE.md hard rule 1) — the whole model is
plain Python and fully headless-testable.

Class definitions live in :mod:`fire_engine.devtools.types` (Field, Section,
Button, Panel) and :mod:`fire_engine.devtools.enums` (FieldKind); this module
re-exports them to preserve every historical import path.

Docs: docs/systems/devtools.md

Example
-------
    from fire_engine.devtools.fields import Field, FieldKind, Section, Panel

    speed = 10.0
    def get_speed() -> float: return speed
    def set_speed(v: float) -> None:
        nonlocal speed; speed = v          # (illustrative)

    panel = Panel(
        tool_id="demo",
        title="Demo",
        sections=[Section("Tuning", [
            Field("speed", FieldKind.FLOAT, get_speed, set_speed, step=0.5),
        ])],
    )
"""

from __future__ import annotations

from fire_engine.devtools.enums import FieldKind
from fire_engine.devtools.types import Button, Field, Panel, Section

__all__ = ["Button", "Field", "FieldKind", "Panel", "Section"]
