"""Mirror tests for fire_engine/scene/types.py.

Covers: FieldSpec, ComponentSpec, SceneError, SceneObject (to_dict / from_dict
round-trip, identity-quat and zero-position defaults).

Categories: CORRECTNESS (field values / defaults), ROUND-TRIP (to_dict /
from_dict), and MIGRATION (pre-component saves handled by from_dict).
"""

from __future__ import annotations

import pytest

from fire_engine.scene.types import (
    ComponentSpec,
    FieldSpec,
    SceneError,
    SceneObject,
)

# ---------------------------------------------------------------------------
# FieldSpec
# ---------------------------------------------------------------------------


class TestFieldSpec:
    def test_required_fields_stored(self):
        f = FieldSpec(name="intensity", ui_type="float", default=8.0, min=0.0, max=64.0)
        assert f.name == "intensity"
        assert f.ui_type == "float"
        assert f.default == 8.0
        assert f.min == 0.0
        assert f.max == 64.0

    def test_optional_fields_default_to_none_and_empty(self):
        f = FieldSpec(name="x", ui_type="bool", default=True)
        assert f.min is None
        assert f.max is None
        assert f.choices == ()
        assert f.label is None

    def test_choices_stored_as_given(self):
        f = FieldSpec(
            name="prim",
            ui_type="enum",
            default="cube",
            choices=("cube", "sphere"),
        )
        assert f.choices == ("cube", "sphere")

    def test_label_stored(self):
        f = FieldSpec(name="col", ui_type="color", default=[1.0, 0.0, 0.0], label="Color")
        assert f.label == "Color"

    def test_is_frozen(self):
        f = FieldSpec(name="x", ui_type="float", default=1.0)
        with pytest.raises((TypeError, AttributeError)):
            f.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ComponentSpec
# ---------------------------------------------------------------------------


class TestComponentSpec:
    def test_basic_fields(self):
        spec = ComponentSpec(type="Mesh", label="Mesh", multiple=False)
        assert spec.type == "Mesh"
        assert spec.label == "Mesh"
        assert spec.multiple is False
        assert spec.fields == ()

    def test_fields_stored(self):
        f = FieldSpec(name="prim", ui_type="enum", default="cube", choices=("cube", "sphere"))
        spec = ComponentSpec(type="Mesh", label="Mesh", multiple=False, fields=(f,))
        assert len(spec.fields) == 1
        assert spec.fields[0].name == "prim"

    def test_is_frozen(self):
        spec = ComponentSpec(type="Mesh", label="Mesh", multiple=False)
        with pytest.raises((TypeError, AttributeError)):
            spec.type = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SceneError
# ---------------------------------------------------------------------------


class TestSceneError:
    def test_is_value_error(self):
        err = SceneError("bad op")
        assert isinstance(err, ValueError)

    def test_message_preserved(self):
        msg = "no scene object with id 42"
        assert str(SceneError(msg)) == msg


# ---------------------------------------------------------------------------
# SceneObject — construction and defaults
# ---------------------------------------------------------------------------


class TestSceneObjectDefaults:
    def test_identity_rotation_default(self):
        obj = SceneObject(id=1, name="Cube", kind="cube")
        assert obj.rotation == (1.0, 0.0, 0.0, 0.0)

    def test_zero_position_default(self):
        obj = SceneObject(id=1, name="Cube", kind="cube")
        assert obj.position == (0.0, 0.0, 0.0)

    def test_unit_scale_default(self):
        obj = SceneObject(id=1, name="Cube", kind="cube")
        assert obj.scale == (1.0, 1.0, 1.0)

    def test_parent_default_none(self):
        obj = SceneObject(id=1, name="Cube", kind="cube")
        assert obj.parent is None

    def test_empty_components_default(self):
        obj = SceneObject(id=1, name="Cube", kind="cube")
        assert obj.components == []


# ---------------------------------------------------------------------------
# SceneObject.to_dict
# ---------------------------------------------------------------------------


class TestSceneObjectToDict:
    def test_basic_fields_in_dict(self):
        obj = SceneObject(id=3, name="Ball", kind="sphere", parent=1)
        d = obj.to_dict()
        assert d["id"] == 3
        assert d["name"] == "Ball"
        assert d["kind"] == "sphere"
        assert d["parent"] == 1

    def test_position_as_list(self):
        obj = SceneObject(id=1, name="X", kind="cube", position=(1.0, 2.0, 3.0))
        assert obj.to_dict()["position"] == [1.0, 2.0, 3.0]

    def test_rotation_as_list(self):
        obj = SceneObject(id=1, name="X", kind="cube", rotation=(0.707, 0.0, 0.707, 0.0))
        assert obj.to_dict()["rotation"] == [0.707, 0.0, 0.707, 0.0]

    def test_scale_as_list(self):
        obj = SceneObject(id=1, name="X", kind="cube", scale=(2.0, 1.0, 1.0))
        assert obj.to_dict()["scale"] == [2.0, 1.0, 1.0]

    def test_components_deep_copied(self):
        comp = {"type": "Mesh", "enabled": True, "params": {"primitive": "cube"}}
        obj = SceneObject(id=1, name="X", kind="cube", components=[comp])
        d = obj.to_dict()
        # Mutating the dict's components must NOT mutate the SceneObject's.
        d["components"][0]["params"]["primitive"] = "sphere"
        assert obj.components[0]["params"]["primitive"] == "cube"


# ---------------------------------------------------------------------------
# SceneObject.from_dict — round-trip and migration
# ---------------------------------------------------------------------------


class TestSceneObjectFromDict:
    def _base_dict(self, **overrides):
        d = {
            "id": 5,
            "name": "Light",
            "kind": "light",
            "parent": None,
            "position": [0.0, 1.0, 2.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "components": [{"type": "Light", "enabled": True, "params": {"intensity": 8.0}}],
        }
        d.update(overrides)
        return d

    def test_round_trip_preserves_all_fields(self):
        d = self._base_dict()
        obj = SceneObject.from_dict(d)
        assert obj.id == 5
        assert obj.name == "Light"
        assert obj.kind == "light"
        assert obj.parent is None
        assert obj.position == (0.0, 1.0, 2.0)
        assert obj.rotation == (1.0, 0.0, 0.0, 0.0)
        assert obj.scale == (1.0, 1.0, 1.0)

    def test_round_trip_components_preserved(self):
        d = self._base_dict()
        obj = SceneObject.from_dict(d)
        assert len(obj.components) == 1
        assert obj.components[0]["type"] == "Light"
        assert obj.components[0]["params"]["intensity"] == 8.0

    def test_to_dict_from_dict_round_trip(self):
        """to_dict → from_dict → to_dict must produce the same dict."""
        d = self._base_dict()
        obj = SceneObject.from_dict(d)
        assert obj.to_dict() == d

    def test_migration_no_components_key_synthesises_defaults(self):
        """A pre-component save (no 'components' key) seeds kind defaults."""
        old = {
            "id": 7,
            "name": "Old Cube",
            "kind": "cube",
            "parent": None,
            "position": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        obj = SceneObject.from_dict(old)
        # cube kind → Mesh component seeded
        assert [c["type"] for c in obj.components] == ["Mesh"]

    def test_explicit_empty_components_list_kept_empty(self):
        """An explicit empty components list is NOT migrated (it's intentional)."""
        d = self._base_dict(components=[])
        obj = SceneObject.from_dict(d)
        assert obj.components == []

    def test_parent_int_or_none(self):
        d_root = self._base_dict(parent=None)
        d_child = self._base_dict(parent=2)
        assert SceneObject.from_dict(d_root).parent is None
        assert SceneObject.from_dict(d_child).parent == 2

    def test_components_deep_copied_from_source_dict(self):
        """Mutating source dict after from_dict should not affect the object."""
        d = self._base_dict()
        obj = SceneObject.from_dict(d)
        d["components"][0]["params"]["intensity"] = 99.0
        assert obj.components[0]["params"]["intensity"] == 8.0
