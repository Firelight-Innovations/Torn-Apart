"""
tests/devtools/test_types.py — tests for fire_engine/devtools/types.py.

Covers Field, Section, Button, Panel (panel model types) and Handle, DragState
(gizmo drag-state types). Exercises construction, field values, read_only flag,
and frozen Handle immutability. Fully headless; no panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.devtools.enums import FieldKind, GizmoMode, HandleType
from fire_engine.devtools.types import (
    Button,
    DragState,
    Field,
    Handle,
    Panel,
    Section,
)

# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


class TestField:
    def test_get_returns_live_value(self):
        store = {"v": 10}
        f = Field("speed", FieldKind.FLOAT, lambda: store["v"])
        assert f.get() == 10
        store["v"] = 99
        assert f.get() == 99

    def test_set_applies_value(self):
        store = {}
        f = Field("n", FieldKind.INT, lambda: store.get("v"), lambda v: store.update(v=v))
        f.set(42)
        assert store["v"] == 42

    def test_read_only_when_no_setter(self):
        f = Field("label", FieldKind.LABEL, lambda: "hello")
        assert f.read_only is True
        assert f.set is None

    def test_not_read_only_with_setter(self):
        f = Field("x", FieldKind.INT, lambda: 0, lambda v: None)
        assert f.read_only is False

    def test_defaults(self):
        f = Field("a", FieldKind.BOOL, lambda: True)
        assert f.step == pytest.approx(0.1)
        assert f.units == ""
        assert f.choices is None

    def test_choices_stored(self):
        choices = ("red", "green", "blue")
        f = Field("colour", FieldKind.ENUM, lambda: "red", choices=choices)
        assert f.choices == choices

    def test_step_and_units_stored(self):
        f = Field("pos", FieldKind.VEC3, lambda: (0, 0, 0), step=0.5, units="m")
        assert f.step == pytest.approx(0.5)
        assert f.units == "m"

    def test_bool_kind(self):
        f = Field("active", FieldKind.BOOL, lambda: False)
        assert f.kind == FieldKind.BOOL


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------


class TestSection:
    def test_holds_title_and_fields(self):
        f = Field("n", FieldKind.INT, lambda: 7)
        s = Section("Props", [f])
        assert s.title == "Props"
        assert s.fields == [f]

    def test_empty_fields(self):
        s = Section("Empty", [])
        assert s.fields == []

    def test_multiple_fields_ordered(self):
        f1 = Field("a", FieldKind.INT, lambda: 1)
        f2 = Field("b", FieldKind.FLOAT, lambda: 2.0)
        s = Section("S", [f1, f2])
        assert s.fields[0] is f1
        assert s.fields[1] is f2


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------


class TestButton:
    def test_label_stored(self):
        b = Button("Launch", lambda: None)
        assert b.label == "Launch"

    def test_on_click_invoked(self):
        fired = []
        b = Button("Go", lambda: fired.append(1))
        b.on_click()
        assert fired == [1]

    def test_on_click_called_multiple_times(self):
        count = {"n": 0}
        b = Button("Inc", lambda: count.update(n=count["n"] + 1))
        b.on_click()
        b.on_click()
        assert count["n"] == 2


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class TestPanel:
    def test_defaults(self):
        p = Panel("tool_id", "Title", [])
        assert p.tool_id == "tool_id"
        assert p.title == "Title"
        assert p.sections == []
        assert p.buttons == []
        assert p.revision == 0

    def test_with_sections_and_buttons(self):
        sec = Section("S", [])
        btn = Button("B", lambda: None)
        p = Panel("t", "T", [sec], buttons=[btn], revision=3)
        assert p.sections == [sec]
        assert p.buttons == [btn]
        assert p.revision == 3

    def test_revision_explicit(self):
        p = Panel("id", "X", [], revision=7)
        assert p.revision == 7


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------


class TestHandle:
    def test_stores_type_and_axis(self):
        h = Handle(HandleType.AXIS, 1)
        assert h.type == HandleType.AXIS
        assert h.axis == 1

    def test_frozen_immutable(self):
        h = Handle(HandleType.RING, 2)
        with pytest.raises((AttributeError, TypeError)):
            h.axis = 99  # type: ignore[misc]

    def test_equality(self):
        assert Handle(HandleType.AXIS, 0) == Handle(HandleType.AXIS, 0)
        assert Handle(HandleType.AXIS, 0) != Handle(HandleType.PLANE, 0)
        assert Handle(HandleType.AXIS, 0) != Handle(HandleType.AXIS, 1)

    def test_all_handle_types(self):
        for ht in HandleType:
            h = Handle(ht, 0)
            assert h.type is ht


# ---------------------------------------------------------------------------
# DragState
# ---------------------------------------------------------------------------


class TestDragState:
    def test_stores_required_fields(self):
        handle = Handle(HandleType.AXIS, 0)
        ds = DragState(
            mode=GizmoMode.TRANSLATE,
            handle=handle,
            pivot=Vec3(1, 2, 3),
            size=1.5,
            start_position=Vec3(0, 0, 0),
            start_rotation=Quat.identity(),
            start_scale=Vec3(1, 1, 1),
        )
        assert ds.mode == GizmoMode.TRANSLATE
        assert ds.handle is handle
        assert ds.pivot.approx_eq(Vec3(1, 2, 3))
        assert ds.size == pytest.approx(1.5)

    def test_optional_fields_default(self):
        ds = DragState(
            mode=GizmoMode.SCALE,
            handle=Handle(HandleType.UNIFORM, 0),
            pivot=Vec3.ZERO,
            size=1.0,
            start_position=Vec3.ZERO,
            start_rotation=Quat.identity(),
            start_scale=Vec3(1, 1, 1),
        )
        assert ds.ref_scalar == pytest.approx(0.0)
        assert ds.ref_point is None
        assert ds.ref_angle == pytest.approx(0.0)
        assert ds.ref_dist == pytest.approx(0.0)

    def test_ref_point_stored_as_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        ds = DragState(
            mode=GizmoMode.TRANSLATE,
            handle=Handle(HandleType.PLANE, 2),
            pivot=Vec3.ZERO,
            size=1.0,
            start_position=Vec3.ZERO,
            start_rotation=Quat.identity(),
            start_scale=Vec3(1, 1, 1),
            ref_point=arr,
        )
        assert ds.ref_point is not None
        assert np.allclose(ds.ref_point, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------


def test_dunder_all():
    import fire_engine.devtools.types as mod

    assert hasattr(mod, "__all__")
    for name in ("Button", "DragState", "Field", "Handle", "Panel", "Section"):
        assert name in mod.__all__, f"{name!r} missing from __all__"
