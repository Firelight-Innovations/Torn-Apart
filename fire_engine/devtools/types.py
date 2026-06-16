"""
Shared trivial support types (dataclasses / NamedTuples) for fire_engine.devtools.

Groups Field, Section, Button, Panel (panel model) and Handle, DragState (gizmo
drag state) into one exempt module so the grouping files fields.py / gizmo.py
stay thin re-export shims.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.devtools.enums import FieldKind, GizmoMode, HandleType

# ---------------------------------------------------------------------------
# Panel / field model (formerly in fields.py)
# ---------------------------------------------------------------------------


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
    set: Callable[[Any], None] | None = None
    choices: tuple[str, ...] | None = None
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


# ---------------------------------------------------------------------------
# Gizmo support types (formerly in gizmo.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Handle:
    """
    One grabbable part of the gizmo.

    Parameters
    ----------
    type : HandleType
    axis : int
        ``0=X / 1=Y / 2=Z``.  For PLANE/RING it is the plane's *normal* axis;
        for UNIFORM it is unused (always 0).
    """

    type: HandleType
    axis: int


@dataclass
class DragState:
    """
    Captured reference pose for an in-progress drag (returned by :meth:`Gizmo.begin`).

    Holds the object's pose at grab time plus the one reference quantity the
    handle needs (axis parameter, plane point, ring angle, or radial distance),
    so :func:`update_drag` can compute an absolute new pose each frame.
    """

    mode: GizmoMode
    handle: Handle
    pivot: Vec3
    size: float
    start_position: Vec3
    start_rotation: Quat
    start_scale: Vec3
    ref_scalar: float = 0.0
    ref_point: np.ndarray | None = None
    ref_angle: float = 0.0
    ref_dist: float = 0.0


__all__ = [
    "Button",
    "DragState",
    "Field",
    "Handle",
    "Panel",
    "Section",
]
