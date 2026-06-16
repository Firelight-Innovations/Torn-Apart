"""
tests/devtools/test_enums.py — tests for fire_engine/devtools/enums.py.

Covers FieldKind, GizmoMode, and HandleType: member existence, values,
enum identity, and that all expected members are present (no extras silently
dropped). Fully headless; no panda3d imports.
"""

from __future__ import annotations

import pytest

from fire_engine.devtools.enums import FieldKind, GizmoMode, HandleType

# ---------------------------------------------------------------------------
# FieldKind
# ---------------------------------------------------------------------------


class TestFieldKind:
    def test_all_members_present(self):
        expected = {"LABEL", "FLOAT", "INT", "BOOL", "STRING", "VEC3", "ENUM"}
        actual = {m.name for m in FieldKind}
        assert actual == expected

    def test_members_are_unique(self):
        values = [m.value for m in FieldKind]
        assert len(values) == len(set(values))

    def test_label_member(self):
        assert FieldKind.LABEL is FieldKind.LABEL

    def test_float_member(self):
        assert FieldKind.FLOAT is FieldKind.FLOAT

    def test_int_member(self):
        assert FieldKind.INT is FieldKind.INT

    def test_bool_member(self):
        assert FieldKind.BOOL is FieldKind.BOOL

    def test_string_member(self):
        assert FieldKind.STRING is FieldKind.STRING

    def test_vec3_member(self):
        assert FieldKind.VEC3 is FieldKind.VEC3

    def test_enum_member(self):
        assert FieldKind.ENUM is FieldKind.ENUM

    def test_lookup_by_name(self):
        assert FieldKind["FLOAT"] is FieldKind.FLOAT
        assert FieldKind["VEC3"] is FieldKind.VEC3

    def test_inequality_between_kinds(self):
        assert FieldKind.FLOAT != FieldKind.INT
        assert FieldKind.BOOL != FieldKind.LABEL


# ---------------------------------------------------------------------------
# GizmoMode
# ---------------------------------------------------------------------------


class TestGizmoMode:
    def test_all_members_present(self):
        expected = {"TRANSLATE", "ROTATE", "SCALE"}
        assert {m.name for m in GizmoMode} == expected

    def test_string_values(self):
        assert GizmoMode.TRANSLATE.value == "translate"
        assert GizmoMode.ROTATE.value == "rotate"
        assert GizmoMode.SCALE.value == "scale"

    def test_lookup_by_value(self):
        assert GizmoMode("translate") is GizmoMode.TRANSLATE
        assert GizmoMode("rotate") is GizmoMode.ROTATE
        assert GizmoMode("scale") is GizmoMode.SCALE

    def test_identity(self):
        assert GizmoMode.TRANSLATE is GizmoMode.TRANSLATE
        assert GizmoMode.TRANSLATE is not GizmoMode.ROTATE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            GizmoMode("spin")


# ---------------------------------------------------------------------------
# HandleType
# ---------------------------------------------------------------------------


class TestHandleType:
    def test_all_members_present(self):
        expected = {"AXIS", "PLANE", "RING", "UNIFORM"}
        assert {m.name for m in HandleType} == expected

    def test_string_values(self):
        assert HandleType.AXIS.value == "axis"
        assert HandleType.PLANE.value == "plane"
        assert HandleType.RING.value == "ring"
        assert HandleType.UNIFORM.value == "uniform"

    def test_lookup_by_value(self):
        assert HandleType("axis") is HandleType.AXIS
        assert HandleType("uniform") is HandleType.UNIFORM

    def test_identity(self):
        assert HandleType.AXIS is HandleType.AXIS
        assert HandleType.PLANE is not HandleType.RING

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            HandleType("arrow")


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


def test_dunder_all_exports():
    import fire_engine.devtools.enums as mod

    assert hasattr(mod, "__all__")
    for name in ("FieldKind", "GizmoMode", "HandleType"):
        assert name in mod.__all__
