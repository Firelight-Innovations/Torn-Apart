"""
tests/devtools/test_introspect.py — tests for fire_engine/devtools/introspect.py.

Covers describe_object (identity/transform/component sections), is_chunk,
describe_chunk, and _public_slots / _field_for_attr indirectly through the public
API. Fully headless; no panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.math3d import Vec3
from fire_engine.devtools.enums import FieldKind
from fire_engine.devtools.introspect import (
    describe_chunk,
    describe_object,
    is_chunk,
)
from fire_engine.render.component import Component
from fire_engine.render.gameobject import GameObject
from fire_engine.render.registry import ComponentRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


# ---------------------------------------------------------------------------
# Minimal test component
# ---------------------------------------------------------------------------


class _Comp(Component):
    __slots__ = ("count", "flag", "label", "offset", "speed")

    def __init__(self) -> None:
        super().__init__()
        self.speed: float = 5.0
        self.count: int = 3
        self.flag: bool = True
        self.label: str = "hi"
        self.offset: Vec3 = Vec3(1, 2, 3)


# ---------------------------------------------------------------------------
# is_chunk
# ---------------------------------------------------------------------------


class TestIsChunk:
    def test_false_for_arbitrary_object(self):
        assert is_chunk(object()) is False

    def test_false_for_gameobject(self):
        assert is_chunk(GameObject(name="Hero")) is False

    def test_true_for_duck_typed_chunk(self):
        class _FakeChunk:
            materials = np.zeros((32, 32, 32), dtype=np.uint8)
            coord = (0, 0, 0)
            chunk_meters = 16.0

        assert is_chunk(_FakeChunk()) is True

    def test_false_for_partial_duck(self):
        class _PartialChunk:
            materials = np.zeros((2, 2, 2), dtype=np.uint8)
            coord = (0, 0, 0)
            # missing chunk_meters

        assert is_chunk(_PartialChunk()) is False


# ---------------------------------------------------------------------------
# describe_chunk
# ---------------------------------------------------------------------------


class TestDescribeChunk:
    def _make_chunk(self):
        from fire_engine.world.terrain.chunk import Chunk

        c = Chunk((1, 0, -1))
        c.materials[0, 0, 0] = 1
        c.materials[1, 0, 0] = 2
        return c

    def test_section_titles(self):
        chunk = self._make_chunk()
        secs = describe_chunk(chunk)
        assert [s.title for s in secs] == ["Chunk", "Voxels"]

    def test_all_fields_read_only(self):
        chunk = self._make_chunk()
        for s in describe_chunk(chunk):
            for f in s.fields:
                assert f.set is None, f"Field {f.label!r} should be read-only"

    def test_coord_field(self):
        chunk = self._make_chunk()
        rows = {f.label: f for s in describe_chunk(chunk) for f in s.fields}
        assert rows["coord"].get() == "(1, 0, -1)"

    def test_solid_count(self):
        chunk = self._make_chunk()
        rows = {f.label: f for s in describe_chunk(chunk) for f in s.fields}
        assert rows["solid"].get() == 2

    def test_edited_flag(self):
        chunk = self._make_chunk()
        rows = {f.label: f for s in describe_chunk(chunk) for f in s.fields}
        assert rows["edited"].get() is False


# ---------------------------------------------------------------------------
# describe_object — section layout
# ---------------------------------------------------------------------------


class TestDescribeObjectLayout:
    def test_minimum_two_sections(self):
        go = GameObject(name="A")
        secs = describe_object(go)
        assert len(secs) >= 2

    def test_first_section_is_gameobject(self):
        go = GameObject(name="A")
        assert describe_object(go)[0].title == "GameObject"

    def test_second_section_is_transform(self):
        go = GameObject(name="A")
        assert describe_object(go)[1].title == "Transform"

    def test_component_section_appended(self):
        go = GameObject(name="A")
        go.add_component(_Comp)
        titles = [s.title for s in describe_object(go)]
        assert "_Comp" in titles


# ---------------------------------------------------------------------------
# describe_object — identity section
# ---------------------------------------------------------------------------


class TestIdentitySection:
    def _identity_fields(self, go):
        return {
            f.label: f for s in describe_object(go) if s.title == "GameObject" for f in s.fields
        }

    def test_name_field_editable(self):
        go = GameObject(name="Hero")
        fields = self._identity_fields(go)
        assert fields["name"].kind == FieldKind.STRING
        fields["name"].set("Villain")
        assert go.name == "Villain"

    def test_active_field_toggles(self):
        go = GameObject(name="A")
        fields = self._identity_fields(go)
        assert fields["active"].get() is True
        fields["active"].set(False)
        assert go.active_self is False

    def test_id_field_is_read_only(self):
        go = GameObject(name="A")
        fields = self._identity_fields(go)
        assert fields["id"].read_only is True


# ---------------------------------------------------------------------------
# describe_object — transform section
# ---------------------------------------------------------------------------


class TestTransformSection:
    def _transform_fields(self, go):
        return {f.label: f for s in describe_object(go) if s.title == "Transform" for f in s.fields}

    def test_position_field_editable(self):
        go = GameObject(name="A")
        fields = self._transform_fields(go)
        fields["position"].set((4.0, 5.0, 6.0))
        assert go.transform.local_position.approx_eq(Vec3(4, 5, 6))

    def test_rotation_field_in_degrees(self):
        go = GameObject(name="A")
        fields = self._transform_fields(go)
        fields["rotation"].set((90.0, 0.0, 0.0))
        h, _p, _r = fields["rotation"].get()
        assert h == pytest.approx(90.0, abs=1e-3)

    def test_scale_field_editable(self):
        go = GameObject(name="A")
        fields = self._transform_fields(go)
        fields["scale"].set((2.0, 3.0, 4.0))
        assert go.transform.local_scale.approx_eq(Vec3(2, 3, 4))


# ---------------------------------------------------------------------------
# describe_object — component section
# ---------------------------------------------------------------------------


class TestComponentSection:
    def _comp_fields(self, go):
        return {f.label: f for s in describe_object(go) if s.title == "_Comp" for f in s.fields}

    def test_float_field_reflected(self):
        go = GameObject(name="A")
        comp = go.add_component(_Comp)
        fields = self._comp_fields(go)
        assert fields["speed"].kind == FieldKind.FLOAT
        fields["speed"].set(99.0)
        assert comp.speed == pytest.approx(99.0)

    def test_int_field_reflected(self):
        go = GameObject(name="A")
        comp = go.add_component(_Comp)
        fields = self._comp_fields(go)
        assert fields["count"].kind == FieldKind.INT
        fields["count"].set(7)
        assert comp.count == 7

    def test_bool_field_reflected(self):
        go = GameObject(name="A")
        comp = go.add_component(_Comp)
        fields = self._comp_fields(go)
        assert fields["flag"].kind == FieldKind.BOOL
        fields["flag"].set(False)
        assert comp.flag is False

    def test_str_field_reflected(self):
        go = GameObject(name="A")
        comp = go.add_component(_Comp)
        fields = self._comp_fields(go)
        assert fields["label"].kind == FieldKind.STRING
        fields["label"].set("world")
        assert comp.label == "world"

    def test_vec3_field_reflected(self):
        go = GameObject(name="A")
        comp = go.add_component(_Comp)
        fields = self._comp_fields(go)
        assert fields["offset"].kind == FieldKind.VEC3
        fields["offset"].set((7.0, 8.0, 9.0))
        assert comp.offset.approx_eq(Vec3(7, 8, 9))

    def test_enabled_field_always_present(self):
        go = GameObject(name="A")
        go.add_component(_Comp)
        fields = self._comp_fields(go)
        assert "enabled" in fields
        assert fields["enabled"].kind == FieldKind.BOOL
