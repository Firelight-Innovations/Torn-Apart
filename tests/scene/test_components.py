"""Mirror tests for fire_engine/scene/components.py.

Covers: COMPONENT_CATALOG shape, is_known, default_params, make_component,
default_components_for_kind, coerce_params, catalog_payload.

Categories: CORRECTNESS (known inputs → known outputs / field values),
ROUND-TRIP (deep-copy isolation so mutations don't alias).
"""

from __future__ import annotations

import pytest

from fire_engine.scene.components import (
    COMPONENT_CATALOG,
    catalog_payload,
    coerce_params,
    default_components_for_kind,
    default_params,
    is_known,
    make_component,
)

# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------


class TestCatalogShape:
    def test_builtin_types_present(self):
        assert set(COMPONENT_CATALOG) == {"Mesh", "Light", "SpawnPoint"}

    def test_mesh_has_primitive_field(self):
        spec = COMPONENT_CATALOG["Mesh"]
        assert len(spec.fields) == 1
        f = spec.fields[0]
        assert f.name == "primitive"
        assert f.ui_type == "enum"
        assert f.default == "cube"
        assert set(f.choices) == {"cube", "sphere"}

    def test_light_has_three_fields(self):
        spec = COMPONENT_CATALOG["Light"]
        names = [f.name for f in spec.fields]
        assert names == ["color", "intensity", "radius"]

    def test_light_intensity_field_bounds(self):
        intensity = next(f for f in COMPONENT_CATALOG["Light"].fields if f.name == "intensity")
        assert intensity.min == 0.0
        assert intensity.max == 64.0

    def test_spawn_point_has_no_fields(self):
        assert COMPONENT_CATALOG["SpawnPoint"].fields == ()

    def test_all_builtins_are_singletons(self):
        for spec in COMPONENT_CATALOG.values():
            assert spec.multiple is False, f"{spec.type} should be singleton"


# ---------------------------------------------------------------------------
# is_known
# ---------------------------------------------------------------------------


class TestIsKnown:
    def test_known_types_return_true(self):
        for t in ("Mesh", "Light", "SpawnPoint"):
            assert is_known(t) is True

    def test_unknown_type_returns_false(self):
        assert is_known("Banana") is False
        assert is_known("mesh") is False  # case-sensitive
        assert is_known("") is False


# ---------------------------------------------------------------------------
# default_params
# ---------------------------------------------------------------------------


class TestDefaultParams:
    def test_mesh_default_params(self):
        p = default_params("Mesh")
        assert p == {"primitive": "cube"}

    def test_light_default_params_keys(self):
        p = default_params("Light")
        assert set(p) == {"color", "intensity", "radius"}

    def test_light_default_intensity(self):
        assert default_params("Light")["intensity"] == 8.0

    def test_light_default_radius(self):
        assert default_params("Light")["radius"] == 16.0

    def test_spawn_point_default_params_empty(self):
        assert default_params("SpawnPoint") == {}

    def test_unknown_type_raises_key_error(self):
        with pytest.raises(KeyError):
            default_params("Banana")

    def test_default_params_are_deep_copied(self):
        """Mutating one call's return must not affect the next."""
        p1 = default_params("Light")
        p1["color"].append(9.0)  # color is a list
        p2 = default_params("Light")
        assert len(p2["color"]) == 3


# ---------------------------------------------------------------------------
# make_component
# ---------------------------------------------------------------------------


class TestMakeComponent:
    def test_structure(self):
        c = make_component("Mesh")
        assert set(c) == {"type", "enabled", "params"}
        assert c["type"] == "Mesh"
        assert c["enabled"] is True

    def test_params_are_defaults(self):
        c = make_component("Mesh")
        assert c["params"] == {"primitive": "cube"}

    def test_enabled_false(self):
        c = make_component("Light", enabled=False)
        assert c["enabled"] is False

    def test_unknown_type_raises_key_error(self):
        with pytest.raises(KeyError):
            make_component("NotAType")

    def test_two_calls_are_independent(self):
        """Mutating one make_component result must not affect the next."""
        c1 = make_component("Light")
        c1["params"]["intensity"] = 999.0
        c2 = make_component("Light")
        assert c2["params"]["intensity"] == 8.0


# ---------------------------------------------------------------------------
# default_components_for_kind
# ---------------------------------------------------------------------------


class TestDefaultComponentsForKind:
    def test_cube_gets_mesh(self):
        comps = default_components_for_kind("cube")
        assert [c["type"] for c in comps] == ["Mesh"]
        assert comps[0]["params"]["primitive"] == "cube"

    def test_sphere_gets_mesh_sphere(self):
        comps = default_components_for_kind("sphere")
        assert [c["type"] for c in comps] == ["Mesh"]
        assert comps[0]["params"]["primitive"] == "sphere"

    def test_light_gets_light(self):
        comps = default_components_for_kind("light")
        assert [c["type"] for c in comps] == ["Light"]

    def test_spawn_gets_spawn_point(self):
        comps = default_components_for_kind("spawn")
        assert [c["type"] for c in comps] == ["SpawnPoint"]

    def test_empty_gets_nothing(self):
        assert default_components_for_kind("empty") == []

    def test_unknown_kind_gets_nothing(self):
        assert default_components_for_kind("banana") == []

    def test_kind_is_case_insensitive(self):
        assert [c["type"] for c in default_components_for_kind("Cube")] == ["Mesh"]
        assert [c["type"] for c in default_components_for_kind("LIGHT")] == ["Light"]

    def test_returns_independent_lists(self):
        """Each call returns a new list (mutations don't alias)."""
        a = default_components_for_kind("light")
        a[0]["params"]["intensity"] = 999.0
        b = default_components_for_kind("light")
        assert b[0]["params"]["intensity"] == 8.0


# ---------------------------------------------------------------------------
# coerce_params
# ---------------------------------------------------------------------------


class TestCoerceParams:
    def test_float_clamped_to_max(self):
        out = coerce_params("Light", {"intensity": 9999.0})
        assert out["intensity"] == 64.0

    def test_float_clamped_to_min(self):
        out = coerce_params("Light", {"radius": -10.0})
        assert out["radius"] == 0.0

    def test_float_in_range_unchanged(self):
        out = coerce_params("Light", {"intensity": 10.0})
        assert out["intensity"] == 10.0

    def test_unknown_key_dropped(self):
        out = coerce_params("Light", {"bogus": 1, "intensity": 5.0})
        assert "bogus" not in out
        assert out["intensity"] == 5.0

    def test_unknown_type_returns_empty(self):
        assert coerce_params("Banana", {"x": 1}) == {}

    def test_enum_valid_choice_kept(self):
        out = coerce_params("Mesh", {"primitive": "sphere"})
        assert out["primitive"] == "sphere"

    def test_enum_invalid_choice_snapped_to_default(self):
        out = coerce_params("Mesh", {"primitive": "pyramid"})
        assert out["primitive"] == "cube"  # default

    def test_color_coerced_to_three_floats(self):
        out = coerce_params("Light", {"color": [0.1, 0.2, 0.3]})
        assert out["color"] == [0.1, 0.2, 0.3]

    def test_color_truncated_to_three_elements(self):
        out = coerce_params("Light", {"color": [0.1, 0.2, 0.3, 0.4]})
        assert out["color"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# catalog_payload
# ---------------------------------------------------------------------------


class TestCatalogPayload:
    def test_has_types_key(self):
        payload = catalog_payload()
        assert "types" in payload

    def test_all_builtins_listed(self):
        payload = catalog_payload()
        types = {t["type"] for t in payload["types"]}
        assert types == {"Mesh", "Light", "SpawnPoint"}

    def test_light_fields_listed_in_order(self):
        payload = catalog_payload()
        light = next(t for t in payload["types"] if t["type"] == "Light")
        assert [f["name"] for f in light["fields"]] == ["color", "intensity", "radius"]

    def test_mesh_field_has_choices(self):
        payload = catalog_payload()
        mesh = next(t for t in payload["types"] if t["type"] == "Mesh")
        choices = mesh["fields"][0]["choices"]
        assert set(choices) == {"cube", "sphere"}

    def test_payload_is_deterministic(self):
        assert catalog_payload() == catalog_payload()

    def test_payload_is_json_serialisable(self):
        """All values must be plain Python primitives (no custom objects)."""
        import json

        json.dumps(catalog_payload())  # must not raise

    def test_field_label_present_in_payload(self):
        payload = catalog_payload()
        intensity = next(
            f
            for t in payload["types"]
            if t["type"] == "Light"
            for f in t["fields"]
            if f["name"] == "intensity"
        )
        assert "label" in intensity

    def test_min_max_present_for_float_fields(self):
        payload = catalog_payload()
        intensity = next(
            f
            for t in payload["types"]
            if t["type"] == "Light"
            for f in t["fields"]
            if f["name"] == "intensity"
        )
        assert intensity["min"] == 0.0
        assert intensity["max"] == 64.0
