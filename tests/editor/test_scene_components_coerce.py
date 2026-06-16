"""Golden-master characterisation tests for fire_engine.scene.components.

Focus: coerce_params + edges, is_known, default_params aliasing, make_component,
default_components_for_kind, and the FieldSpec/ComponentSpec frozen-dataclass
contracts.  Do NOT fix bugs — pin current behaviour and note suspicions.

Headless only: no panda3d, no fire_engine.world.
"""

from __future__ import annotations

import pytest

from fire_engine.scene.components import (
    COMPONENT_CATALOG,
    coerce_params,
    default_components_for_kind,
    default_params,
    is_known,
    make_component,
)


# ============================================================================ #
# FieldSpec / ComponentSpec — frozen dataclass contracts
# ============================================================================ #
class TestFieldSpecFrozen:
    def test_fieldspec_is_frozen_raises_on_assign(self):
        """FieldSpec must be immutable; any attribute assignment raises."""
        fs = COMPONENT_CATALOG["Light"].fields[1]  # intensity
        with pytest.raises((AttributeError, TypeError)):
            fs.name = "other"  # type: ignore[misc]

    def test_componentspec_is_frozen_raises_on_assign(self):
        spec = COMPONENT_CATALOG["Mesh"]
        with pytest.raises((AttributeError, TypeError)):
            spec.label = "Changed"  # type: ignore[misc]

    def test_fieldspec_attributes_light_intensity(self):
        """Pin exact field metadata for Light.intensity."""
        f = COMPONENT_CATALOG["Light"].fields[1]
        assert f.name == "intensity"
        assert f.ui_type == "float"
        assert f.default == 8.0
        assert f.min == 0.0
        assert f.max == 64.0
        assert f.label == "Intensity"

    def test_fieldspec_attributes_mesh_primitive(self):
        f = COMPONENT_CATALOG["Mesh"].fields[0]
        assert f.name == "primitive"
        assert f.ui_type == "enum"
        assert f.default == "cube"
        assert f.choices == ("cube", "sphere")
        assert f.min is None
        assert f.max is None

    def test_spawnpoint_has_no_fields(self):
        assert COMPONENT_CATALOG["SpawnPoint"].fields == ()

    def test_componentspec_multiple_all_false(self):
        """All built-ins are singletons."""
        for spec in COMPONENT_CATALOG.values():
            assert spec.multiple is False


# ============================================================================ #
# is_known
# ============================================================================ #
class TestIsKnown:
    def test_known_types(self):
        assert is_known("Mesh") is True
        assert is_known("Light") is True
        assert is_known("SpawnPoint") is True

    def test_unknown_type(self):
        assert is_known("Banana") is False

    def test_empty_string(self):
        # Pin: empty string is not a known type.
        assert is_known("") is False

    def test_case_sensitive(self):
        # "mesh" (lowercase) is NOT the same as "Mesh".
        assert is_known("mesh") is False
        assert is_known("light") is False


# ============================================================================ #
# default_params
# ============================================================================ #
class TestDefaultParams:
    def test_returns_correct_keys_for_light(self):
        p = default_params("Light")
        assert set(p.keys()) == {"color", "intensity", "radius"}

    def test_returns_correct_defaults_for_light(self):
        p = default_params("Light")
        assert p["intensity"] == 8.0
        assert p["radius"] == 16.0
        # color is a list copy of (1.0, 0.62, 0.28)
        assert p["color"] == [1.0, 0.62, 0.28]

    def test_returns_correct_defaults_for_mesh(self):
        p = default_params("Mesh")
        assert p == {"primitive": "cube"}

    def test_spawnpoint_returns_empty_dict(self):
        assert default_params("SpawnPoint") == {}

    def test_no_aliasing_between_calls(self):
        """Mutating one call's result must not affect the next call."""
        p1 = default_params("Light")
        p1["color"].append(99)  # mutate the list in place
        p2 = default_params("Light")
        assert len(p2["color"]) == 3  # fresh deep copy — unaffected

    def test_no_aliasing_between_two_returns(self):
        """Two successive calls return equal but distinct objects."""
        p1 = default_params("Light")
        p2 = default_params("Light")
        assert p1 == p2
        assert p1 is not p2
        assert p1["color"] is not p2["color"]  # list is deep-copied

    def test_unknown_type_raises_key_error(self):
        """Pin: unknown type_name raises KeyError (not None, not {})."""
        with pytest.raises(KeyError):
            default_params("Banana")


# ============================================================================ #
# make_component
# ============================================================================ #
class TestMakeComponent:
    def test_shape_of_mesh_component(self):
        c = make_component("Mesh")
        assert set(c.keys()) == {"type", "enabled", "params"}
        assert c["type"] == "Mesh"
        assert c["enabled"] is True
        assert c["params"] == {"primitive": "cube"}

    def test_enabled_default_is_true(self):
        assert make_component("Light")["enabled"] is True

    def test_enabled_false_is_stored(self):
        c = make_component("Light", enabled=False)
        assert c["enabled"] is False

    def test_enabled_truthy_int_coerced_to_bool(self):
        """enabled=1 should be stored as True (bool), not 1 (int)."""
        c = make_component("Light", enabled=1)
        assert c["enabled"] is True

    def test_params_equal_default_params(self):
        c = make_component("Light")
        assert c["params"] == default_params("Light")

    def test_make_spawnpoint_empty_params(self):
        c = make_component("SpawnPoint")
        assert c["params"] == {}

    def test_unknown_type_raises_key_error(self):
        """Pin: make_component propagates KeyError from default_params."""
        with pytest.raises(KeyError):
            make_component("Banana")


# ============================================================================ #
# default_components_for_kind
# ============================================================================ #
class TestDefaultComponentsForKind:
    def test_cube_seeds_mesh_with_cube_primitive(self):
        comps = default_components_for_kind("cube")
        assert [c["type"] for c in comps] == ["Mesh"]
        assert comps[0]["params"]["primitive"] == "cube"

    def test_sphere_seeds_mesh_with_sphere_primitive(self):
        comps = default_components_for_kind("sphere")
        assert [c["type"] for c in comps] == ["Mesh"]
        assert comps[0]["params"]["primitive"] == "sphere"

    def test_light_seeds_light_component(self):
        comps = default_components_for_kind("light")
        assert [c["type"] for c in comps] == ["Light"]

    def test_spawn_seeds_spawnpoint(self):
        comps = default_components_for_kind("spawn")
        assert [c["type"] for c in comps] == ["SpawnPoint"]

    def test_empty_seeds_no_components(self):
        assert default_components_for_kind("empty") == []

    def test_unknown_kind_returns_empty_list(self):
        """Pin: unrecognised kind returns [] (same as 'empty'), not an error."""
        assert default_components_for_kind("whatever_unknown") == []

    def test_kind_is_case_insensitive(self):
        """Pin: kind matching is lowercased — 'CUBE' behaves like 'cube'."""
        comps = default_components_for_kind("CUBE")
        assert [c["type"] for c in comps] == ["Mesh"]
        assert comps[0]["params"]["primitive"] == "cube"


# ============================================================================ #
# coerce_params
# ============================================================================ #
class TestCoerceParams:
    # --- valid round-trips -------------------------------------------------- #
    def test_valid_float_passes_through(self):
        out = coerce_params("Light", {"intensity": 5.0, "radius": 10.0})
        assert out["intensity"] == 5.0
        assert out["radius"] == 10.0

    def test_valid_color_passes_through(self):
        out = coerce_params("Light", {"color": [0.1, 0.2, 0.3]})
        assert out["color"] == [0.1, 0.2, 0.3]

    def test_valid_enum_passes_through(self):
        out = coerce_params("Mesh", {"primitive": "sphere"})
        assert out["primitive"] == "sphere"

    # --- type coercion ------------------------------------------------------- #
    def test_int_where_float_expected_becomes_float(self):
        """Pin: int 5 is coerced to float via float(value)."""
        out = coerce_params("Light", {"intensity": 5})
        # Current behaviour: float(5) == 5.0
        assert out["intensity"] == 5.0
        assert isinstance(out["intensity"], float)

    def test_string_number_where_float_expected_coerced_to_float(self):
        """Pin: '3.5' is coerced by float('3.5') — currently passes.
        SUSPICION: accepting arbitrary strings as numeric params is fragile;
        may be intentional for JSON round-trips, but a non-numeric string
        (e.g. 'abc') would raise ValueError — not caught.
        """
        out = coerce_params("Light", {"intensity": "3.5"})
        assert out["intensity"] == 3.5

    def test_non_numeric_string_where_float_expected_raises(self):
        """Pin: 'abc' cannot be cast to float — raises ValueError.
        SUSPICION BUG: coerce_params lets ValueError propagate uncaught;
        callers expecting a safe coerce_params must handle this themselves.
        """
        with pytest.raises((ValueError, TypeError)):
            coerce_params("Light", {"intensity": "abc"})

    # --- clamp behaviour ----------------------------------------------------- #
    def test_float_above_max_clamped(self):
        out = coerce_params("Light", {"intensity": 9999.0})
        assert out["intensity"] == COMPONENT_CATALOG["Light"].fields[1].max  # 64.0

    def test_float_below_min_clamped(self):
        out = coerce_params("Light", {"radius": -999.0})
        assert out["radius"] == 0.0

    def test_float_at_boundary_not_clamped(self):
        out = coerce_params("Light", {"intensity": 64.0, "radius": 0.0})
        assert out["intensity"] == 64.0
        assert out["radius"] == 0.0

    # --- enum coercion ------------------------------------------------------- #
    def test_invalid_enum_snaps_to_default(self):
        """Pin: invalid enum value returns the field default, not raises."""
        out = coerce_params("Mesh", {"primitive": "triangle"})
        assert out["primitive"] == "cube"  # default

    def test_valid_enum_choice_passes(self):
        assert coerce_params("Mesh", {"primitive": "sphere"})["primitive"] == "sphere"

    # --- extra / unknown keys ------------------------------------------------ #
    def test_unknown_keys_are_dropped(self):
        """Pin: unknown keys silently dropped (not raised, not kept)."""
        out = coerce_params("Light", {"intensity": 5.0, "bogus_key": 42})
        assert "bogus_key" not in out
        assert out["intensity"] == 5.0

    def test_all_unknown_keys_returns_empty(self):
        out = coerce_params("Light", {"x": 1, "y": 2})
        assert out == {}

    # --- missing keys -------------------------------------------------------- #
    def test_missing_keys_not_filled_with_defaults(self):
        """Pin: coerce_params does NOT back-fill missing keys with defaults.
        It only processes what is provided. Callers must merge manually.
        SUSPICION: set_component merges with existing params, so this is
        intentional — but it means coerce_params({}) always returns {}.
        """
        out = coerce_params("Light", {})
        assert out == {}

    # --- unknown type_name --------------------------------------------------- #
    def test_unknown_type_name_returns_empty_dict(self):
        """Pin: unregistered type_name returns {} (not raises KeyError)."""
        out = coerce_params("Banana", {"intensity": 5.0})
        assert out == {}

    def test_empty_type_name_returns_empty_dict(self):
        out = coerce_params("", {"intensity": 5.0})
        assert out == {}

    # --- color / vec3 coercion ----------------------------------------------- #
    def test_color_4tuple_truncated_to_3(self):
        """Pin: color field with 4 values silently truncated to first 3."""
        out = coerce_params("Light", {"color": [0.5, 0.5, 0.5, 1.0]})
        assert out["color"] == [0.5, 0.5, 0.5]

    def test_color_short_list_zero_padded(self):
        """Pin: color list with fewer than 3 values is zero-padded."""
        out = coerce_params("Light", {"color": [0.5, 0.5]})
        assert out["color"] == [0.5, 0.5, 0.0]

    def test_color_int_values_coerced_to_float(self):
        out = coerce_params("Light", {"color": [1, 0, 0]})
        assert out["color"] == [1.0, 0.0, 0.0]
        assert all(isinstance(v, float) for v in out["color"])
