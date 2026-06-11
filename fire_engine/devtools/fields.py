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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class FieldKind(Enum):
    """
    The widget family a :class:`Field` maps to.

    The renderer chooses a concrete control per kind:

    LABEL   — read-only text (a formatted string from ``get``).
    FLOAT   — single numeric entry; ``get``→float, ``set``(float).
    INT     — single integer entry; ``get``→int, ``set``(int).
    BOOL    — toggle; ``get``→bool, ``set``(bool).
    STRING  — text entry; ``get``→str, ``set``(str).
    VEC3    — three numeric entries; ``get``→(x, y, z) tuple of float,
              ``set``((x, y, z)).  Used for positions, scales, and
              euler-angle views of rotations.
    ENUM    — choice from ``choices``; ``get``→str, ``set``(str).
    """

    LABEL = auto()
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    VEC3 = auto()
    ENUM = auto()


@dataclass
class Field:
    """
    One inspectable / editable property in a panel.

    Parameters
    ----------
    label : str
        Human-readable name shown next to the control.
    kind : FieldKind
        Which widget family to render (see :class:`FieldKind`).
    get : Callable[[], Any]
        Returns the current value, in the units implied by ``kind``.  Called
        every frame for live read-only rows and to refresh unfocused editors.
    set : Callable[[Any], None] | None
        Applies a new value through the engine's public setter.  ``None`` marks
        the field read-only (the renderer shows it but offers no editor).
    choices : tuple[str, ...] | None
        Allowed values for ``FieldKind.ENUM``; ignored otherwise.
    step : float
        Suggested increment for numeric drag/step controls (renderer hint).
    units : str
        Optional unit suffix for display (e.g. ``"m"``, ``"deg"``).

    Notes
    -----
    ``get``/``set`` are intentionally closures, not a cached value: the panel
    a tool returns each frame stays in sync with live engine state without the
    tool having to diff anything.
    """

    label: str
    kind: FieldKind
    get: Callable[[], Any]
    set: Optional[Callable[[Any], None]] = None
    choices: Optional[tuple[str, ...]] = None
    step: float = 0.1
    units: str = ""

    @property
    def read_only(self) -> bool:
        """True when this field has no setter (display only)."""
        return self.set is None


@dataclass
class Section:
    """
    A titled group of related fields (renders as a labelled sub-block).

    Parameters
    ----------
    title : str
        Section heading (e.g. ``"Transform"``, ``"FlyController"``).
    fields : list[Field]
        The rows in this section, top-to-bottom.
    """

    title: str
    fields: list[Field]


@dataclass
class Button:
    """
    A one-shot action button (spawn something, fire an event, reset state...).

    Parameters
    ----------
    label : str
        Button caption.
    on_click : Callable[[], None]
        Invoked when the user presses the button.  Should be cheap / fire-and-
        forget; long work belongs off the UI path.
    """

    label: str
    on_click: Callable[[], None]


@dataclass
class Panel:
    """
    The complete on-screen description of one dev tool for the current frame.

    A tool rebuilds this each frame via :meth:`DevTool.build`.  Construction is
    cheap (dataclasses + closures), so there is no diffing on the tool side.

    Parameters
    ----------
    tool_id : str
        Stable identifier (used by the renderer to keep widget state across
        frames — same ``tool_id`` ⇒ same on-screen panel).
    title : str
        Panel caption shown in its title bar.
    sections : list[Section]
        Field groups, top-to-bottom.
    buttons : list[Button]
        Action buttons shown at the foot of the panel.
    revision : int
        Bumps whenever the panel's *structure* (which sections/fields/buttons
        exist) changes — e.g. when the inspector's selection changes.  The
        renderer rebuilds its widgets only when ``revision`` changes; between
        bumps it just polls ``Field.get`` to refresh values.  Editing a value
        does **not** bump the revision.
    """

    tool_id: str
    title: str
    sections: list[Section]
    buttons: list[Button] = field(default_factory=list)
    revision: int = 0
