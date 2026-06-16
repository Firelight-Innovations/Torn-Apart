"""
render/overlay/_overlay_panels.py — DirectGUI panel / widget build and teardown.

Extracted from devtools_overlay.py; called as free functions taking the overlay
instance as first argument (C0302 fat-class split pattern).

Docs: docs/systems/render.overlay.md
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import (
    DirectButton,
    DirectEntry,
    DirectFrame,
    DirectLabel,
)
from panda3d.core import TextNode

from fire_engine.devtools import Button, Field, FieldKind, Panel

if TYPE_CHECKING:
    from fire_engine.render.overlay.devtools_overlay import DevOverlay

# --- Layout constants (aspect2d units) ------------------------------------
_TEXT_SCALE = 0.040
_ROW_H = 0.052
_PANEL_W = 0.64  # left column panel width
_INSPECTOR_W = 0.74  # right column panel width
_MARGIN_X = 0.04
_TOP_Z = -0.06
_LABEL_COL = 0.30  # x offset where the value/control begins within a panel
_ENTRY_SCALE = 0.038

_PANEL_BG = (0.05, 0.06, 0.08, 0.74)
_TITLE_FG = (0.55, 0.85, 1.0, 1.0)
_SECTION_FG = (1.0, 0.82, 0.4, 1.0)
_VALUE_FG = (0.92, 0.95, 1.0, 1.0)


def _fmt(value: object) -> str:
    """Compact display string for a scalar field value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# ---------------------------------------------------------------------------
# Widget teardown
# ---------------------------------------------------------------------------


def clear_widgets(self_obj: DevOverlay) -> None:
    """Destroy all live DirectGui widgets and clear per-frame updaters.

    Docs: docs/systems/render.overlay.md
    """
    for w in self_obj._widgets:
        with contextlib.suppress(Exception):
            w.destroy()
    self_obj._widgets.clear()
    self_obj._updaters.clear()


# ---------------------------------------------------------------------------
# Panel rebuild
# ---------------------------------------------------------------------------


def rebuild(self_obj: DevOverlay) -> None:
    """Tear down all widgets and rebuild them from the current panel list.

    Docs: docs/systems/render.overlay.md
    """
    clear_widgets(self_obj)
    left_z = _TOP_Z
    right_z = _TOP_Z
    for panel in self_obj.manager.panels():
        if panel.tool_id == "inspector":
            parent = self_obj._base.a2dTopRight
            x = -_INSPECTOR_W - _MARGIN_X
            width = _INSPECTOR_W
            right_z = build_panel(self_obj, panel, parent, x, right_z, width)
            right_z -= _ROW_H * 0.6
        else:
            parent = self_obj._base.a2dTopLeft
            x = _MARGIN_X
            width = _PANEL_W
            left_z = build_panel(self_obj, panel, parent, x, left_z, width)
            left_z -= _ROW_H * 0.6


def build_panel(
    self_obj: DevOverlay, panel: Panel, parent: Any, x: float, z: float, width: float
) -> float:
    """Render one panel starting at ``z``; return the z below the panel.

    Docs: docs/systems/render.overlay.md
    """
    top = z
    rows: list[tuple[str, Any]] = []  # deferred widget creation so the bg frame sits behind

    # Title
    rows.append(("title", panel.title))
    for section in panel.sections:
        if section.title:
            rows.append(("section", section.title))
        rows.extend(("field", fld) for fld in section.fields)
    if panel.buttons:
        rows.append(("buttons", panel.buttons))

    # Background frame (sized to the row count) — created first so it's behind.
    n_rows = len(rows)
    height = n_rows * _ROW_H + 0.04
    bg = DirectFrame(
        parent=parent,
        frameColor=_PANEL_BG,
        frameSize=(x - 0.02, x + width, top - height, top + 0.02),
        state="normal",  # eats clicks so picking ignores the panel area
    )
    self_obj._widgets.append(bg)

    cz = top - _ROW_H + 0.01
    for kind, payload in rows:
        if kind == "title":
            mk_label(self_obj, parent, x, cz, payload, _TITLE_FG, _TEXT_SCALE * 1.05)
        elif kind == "section":
            mk_label(self_obj, parent, x + 0.01, cz, payload, _SECTION_FG, _TEXT_SCALE * 0.95)
        elif kind == "field":
            mk_field(self_obj, parent, x, cz, payload, width)
        elif kind == "buttons":
            mk_buttons(self_obj, parent, x, cz, payload)
        cz -= _ROW_H
    return top - height


# ---------------------------------------------------------------------------
# Row widgets
# ---------------------------------------------------------------------------


def mk_label(
    self_obj: DevOverlay,
    parent: Any,
    x: float,
    z: float,
    text: str,
    fg: Any,
    scale: float,
) -> DirectLabel:
    """Create and register a single DirectLabel.

    Docs: docs/systems/render.overlay.md
    """
    lbl = DirectLabel(
        parent=parent,
        text=str(text),
        text_fg=fg,
        text_scale=scale,
        text_align=TextNode.ALeft,
        relief=None,
        pos=(x + 0.02, 0, z),
    )
    self_obj._widgets.append(lbl)
    return lbl


def mk_field(
    self_obj: DevOverlay, parent: Any, x: float, z: float, fld: Field, width: float
) -> None:
    """Build the appropriate widget(s) for a single panel field.

    Docs: docs/systems/render.overlay.md
    """
    # Field name on the left.
    mk_label(self_obj, parent, x + 0.01, z, fld.label, _VALUE_FG, _TEXT_SCALE * 0.9)
    vx = x + _LABEL_COL

    if fld.kind == FieldKind.LABEL or fld.read_only:
        val_lbl = mk_label(
            self_obj, parent, vx, z, _fmt(fld.get()), (0.7, 0.9, 0.7, 1.0), _TEXT_SCALE * 0.9
        )
        self_obj._updaters.append(lambda lbl=val_lbl, f=fld: lbl.__setitem__("text", _fmt(f.get())))
        return

    if fld.kind == FieldKind.BOOL:
        btn = DirectButton(
            parent=parent,
            text=checkbox(fld.get()),
            text_scale=_TEXT_SCALE,
            text_align=TextNode.ALeft,
            relief=None,
            text_fg=(0.8, 1.0, 0.8, 1.0),
            pos=(vx, 0, z),
            command=lambda f=fld: f.set(not f.get()),
        )
        self_obj._widgets.append(btn)
        self_obj._updaters.append(lambda b=btn, f=fld: b.__setitem__("text", checkbox(f.get())))
        return

    if fld.kind == FieldKind.VEC3:
        mk_field_vec3(self_obj, parent, vx, z, fld)
        return

    # FLOAT / INT / STRING — single entry
    mk_field_scalar(self_obj, parent, vx, z, fld)


def mk_field_vec3(self_obj: DevOverlay, parent: Any, vx: float, z: float, fld: Field) -> None:
    """Build a three-component VEC3 entry row and wire up submit + refresh.

    Docs: docs/systems/render.overlay.md
    """
    entries: list[DirectEntry] = []
    for i in range(3):
        e = mk_entry(self_obj, parent, vx + i * 0.15, z, width=4)
        entries.append(e)

    def submit(_: Any = None, f: Field = fld, es: list[DirectEntry] = entries) -> None:
        if f.set is None:
            return
        try:
            vals = tuple(float(e.get()) for e in es)
        except ValueError:
            return
        f.set(vals)

    for e in entries:
        # Commit on Enter AND on click-off (focus out), so an edit is
        # never silently discarded by leaving the box.
        e["command"] = submit
        e["focusOutCommand"] = submit
    self_obj._updaters.append(lambda es=entries, f=fld: refresh_vec3(self_obj, es, f))


def mk_field_scalar(self_obj: DevOverlay, parent: Any, vx: float, z: float, fld: Field) -> None:
    """Build a single-value entry row (FLOAT / INT / STRING) and wire it up.

    Docs: docs/systems/render.overlay.md
    """
    entry = mk_entry(self_obj, parent, vx, z, width=8)

    def submit_scalar(_: Any = None, f: Field = fld, e: DirectEntry = entry) -> None:
        if f.set is None:
            return
        txt = e.get()
        try:
            if f.kind == FieldKind.INT:
                f.set(int(float(txt)))
            elif f.kind == FieldKind.FLOAT:
                f.set(float(txt))
            else:
                f.set(txt)
        except ValueError:
            return

    # Commit on Enter AND on click-off (focus out).
    entry["command"] = submit_scalar
    entry["focusOutCommand"] = submit_scalar
    self_obj._updaters.append(lambda e=entry, f=fld: refresh_scalar(self_obj, e, f))


def mk_entry(self_obj: DevOverlay, parent: Any, x: float, z: float, width: int) -> DirectEntry:
    """Create and register a DirectEntry widget.

    Docs: docs/systems/render.overlay.md
    """
    e = DirectEntry(
        parent=parent,
        scale=_ENTRY_SCALE,
        pos=(x, 0, z),
        width=width,
        numLines=1,
        initialText="",
        text_align=TextNode.ALeft,
        frameColor=(0.15, 0.17, 0.2, 0.9),
        text_fg=(1, 1, 1, 1),
    )
    self_obj._widgets.append(e)
    return e


def mk_buttons(
    self_obj: DevOverlay, parent: Any, x: float, z: float, buttons: list[Button]
) -> None:
    """Create and register DirectButton widgets for a panel button row.

    Docs: docs/systems/render.overlay.md
    """
    bx = x + 0.02
    for b in buttons:
        btn = DirectButton(
            parent=parent,
            text=b.label,
            text_scale=_TEXT_SCALE * 0.9,
            text_align=TextNode.ALeft,
            pos=(bx, 0, z),
            frameColor=(0.2, 0.35, 0.5, 0.95),
            text_fg=(1, 1, 1, 1),
            # FLAT + a thin border: the DirectGui default is a 0.1-unit
            # raised bevel, which on an unscaled button dwarfs the 0.036-tall
            # text and swallows the label. A flat fill sized snugly to the
            # text reads as a clean button.
            relief=DGG.FLAT,
            borderWidth=(0.01, 0.01),
            command=b.on_click,
            pad=(0.02, 0.01),
        )
        self_obj._widgets.append(btn)
        bx += 0.02 + len(b.label) * _TEXT_SCALE * 0.62


# ---------------------------------------------------------------------------
# Live-value refresh helpers (skip widgets the user is editing)
# ---------------------------------------------------------------------------


def checkbox(value: bool) -> str:
    """Return a visual checkbox string.

    Docs: docs/systems/render.overlay.md
    """
    return "[x]" if value else "[ ]"


def is_focused(entry: DirectEntry) -> bool:
    """Return True when a DirectEntry is being actively edited by the user.

    The PGEntry method is get_focus() — the old is_focused() never existed,
    so this used to always raise → always return False → the per-frame
    refresh stomped whatever the user was typing.  With real focus state,
    a focused entry is left untouched until Enter / click-off commits it.

    Docs: docs/systems/render.overlay.md
    """
    try:
        return bool(entry.guiItem.get_focus())
    except Exception:
        return False


def refresh_scalar(self_obj: DevOverlay, entry: DirectEntry, fld: Field) -> None:
    """Update a scalar DirectEntry with the field's current value (unless focused).

    Docs: docs/systems/render.overlay.md
    """
    if is_focused(entry):
        return
    entry.set(_fmt(fld.get()))


def refresh_vec3(self_obj: DevOverlay, entries: list[DirectEntry], fld: Field) -> None:
    """Update three DirectEntry widgets with the field's current Vec3 value.

    Docs: docs/systems/render.overlay.md
    """
    vals = fld.get()
    for e, v in zip(entries, vals, strict=False):
        if is_focused(e):
            continue
        e.set(_fmt(float(v)))
