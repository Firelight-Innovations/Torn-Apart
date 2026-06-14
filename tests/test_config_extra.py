"""Characterisation / golden-master tests for Config derived properties and edges.

Covers areas NOT already exercised by tests/test_graphics_config.py:
  - Derived properties (chunk_meters, light_cell_meters) and their relationships
  - Units invariants documented in CLAUDE.md and docs/systems/core.md
  - Frozen-dataclass behaviour (FrozenInstanceError, replace() identity)
  - Determinism of load_config() and resolve_graphics_preset()
  - resolve_graphics_preset() edges: empty table, only-overrides, unknown keys
  - load_config() with a minimal TOML file that applies overrides (tmp_path variant)
  - Key default field types and values (world_seed, view_distance_chunks, etc.)

NOTE: This file pins CURRENT behaviour. Do NOT fix bugs here; report suspicions
in comments only.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from fire_engine.core.config import (
    GRAPHICS_PRESETS,
    Config,
    load_config,
    resolve_graphics_preset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default() -> Config:
    """Return a fresh default Config (equivalent to Config())."""
    return Config()


# ===========================================================================
# Derived properties — assert the RELATIONSHIP, not a literal magic number
# ===========================================================================

class TestDerivedProperties:
    def test_chunk_meters_formula(self):
        """chunk_meters == chunk_size * voxel_size (the defining formula)."""
        cfg = _default()
        assert cfg.chunk_meters == float(cfg.chunk_size) * cfg.voxel_size

    def test_light_cell_meters_formula(self):
        """light_cell_meters == voxel_size * light_grid_scale."""
        cfg = _default()
        assert cfg.light_cell_meters == cfg.voxel_size * float(cfg.light_grid_scale)

    def test_chunk_meters_varies_with_chunk_size(self):
        """replace(chunk_size=…) propagates into chunk_meters."""
        cfg = dataclasses.replace(_default(), chunk_size=16)
        assert cfg.chunk_meters == float(16) * cfg.voxel_size

    def test_chunk_meters_varies_with_voxel_size(self):
        """replace(voxel_size=…) propagates into chunk_meters."""
        cfg = dataclasses.replace(_default(), voxel_size=1.0)
        assert cfg.chunk_meters == float(cfg.chunk_size) * 1.0

    def test_light_cell_meters_varies_with_scale(self):
        """replace(light_grid_scale=…) propagates into light_cell_meters."""
        cfg = dataclasses.replace(_default(), light_grid_scale=4)
        assert cfg.light_cell_meters == cfg.voxel_size * float(4)

    def test_derived_properties_are_not_stored_fields(self):
        """chunk_meters and light_cell_meters must NOT be stored fields."""
        field_names = {f.name for f in dataclasses.fields(Config)}
        assert "chunk_meters" not in field_names
        assert "light_cell_meters" not in field_names


# ===========================================================================
# CLAUDE.md / core.md documented invariants at defaults
# ===========================================================================

class TestDocumentedInvariants:
    def test_voxel_size_is_half_meter(self):
        assert _default().voxel_size == 0.5

    def test_chunk_size_is_32(self):
        assert _default().chunk_size == 32

    def test_chunk_meters_is_16(self):
        """CLAUDE.md: 'chunk = 32³ voxels = 16 m'."""
        assert _default().chunk_meters == 16.0

    def test_light_cell_meters_is_1(self):
        """CLAUDE.md: 'light cell = 1 m'."""
        assert _default().light_cell_meters == 1.0

    def test_light_grid_scale_is_2(self):
        """core.md: light_grid_scale default = 2."""
        assert _default().light_grid_scale == 2


# ===========================================================================
# Frozen-dataclass guarantees
# ===========================================================================

class TestFrozenDataclass:
    def test_attribute_assignment_raises(self):
        """Config is frozen; mutation must raise FrozenInstanceError."""
        cfg = _default()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.world_seed = 9999  # type: ignore[misc]

    def test_replace_returns_new_instance(self):
        """dataclasses.replace() must produce a different object."""
        cfg = _default()
        cfg2 = dataclasses.replace(cfg, world_seed=42)
        assert cfg2 is not cfg
        assert cfg2.world_seed == 42
        assert cfg.world_seed != 42  # original unchanged

    def test_replace_leaves_other_fields_untouched(self):
        """Replacing one field must not alter other fields."""
        cfg = _default()
        cfg2 = dataclasses.replace(cfg, world_seed=0)
        assert cfg2.chunk_size == cfg.chunk_size
        assert cfg2.voxel_size == cfg.voxel_size
        assert cfg2.chunk_meters == cfg.chunk_meters

    def test_default_instances_are_equal(self):
        """Two fresh Config() calls should be equal (frozen + same defaults)."""
        assert Config() == Config()


# ===========================================================================
# Determinism of load_config and resolve_graphics_preset
# ===========================================================================

class TestDeterminism:
    def test_load_config_twice_equal(self, tmp_path):
        """load_config() on the same file must produce equal results."""
        toml = tmp_path / "cfg.toml"
        toml.write_text("world_seed = 555\n[graphics]\npreset = \"medium\"\n",
                        encoding="utf-8")
        a = load_config(str(toml))
        b = load_config(str(toml))
        assert a == b

    def test_load_config_missing_twice_equal(self):
        """load_config on a non-existent path must be stable."""
        a = load_config("surely-missing-file-abc123.toml")
        b = load_config("surely-missing-file-abc123.toml")
        assert a == b

    def test_resolve_graphics_preset_same_table_twice_equal(self):
        """resolve_graphics_preset must be deterministic (same input → same output)."""
        table = {"preset": "low", "gfx_bloom_mips": 7}
        assert resolve_graphics_preset(dict(table)) == resolve_graphics_preset(dict(table))

    def test_resolve_graphics_preset_none_twice_equal(self):
        assert resolve_graphics_preset(None) == resolve_graphics_preset(None)


# ===========================================================================
# resolve_graphics_preset edges (not already in test_graphics_config.py)
# ===========================================================================

class TestResolveEdges:
    def test_empty_table_returns_high_preset(self):
        """Empty dict → same as no table → 'high' preset."""
        result = resolve_graphics_preset({})
        assert result["gfx_preset"] == "high"
        for key, value in GRAPHICS_PRESETS["high"].items():
            assert result[key] == value, f"mismatch on {key}"

    def test_only_overrides_no_preset_key(self):
        """Table with only gfx_* keys (no 'preset') → base is 'high', overrides win."""
        result = resolve_graphics_preset({"gfx_cloud_steps": 999})
        assert result["gfx_preset"] == "high"          # default base
        assert result["gfx_cloud_steps"] == 999        # override wins
        # Remaining preset keys still from "high"
        assert result["gfx_bloom_mips"] == GRAPHICS_PRESETS["high"]["gfx_bloom_mips"]

    def test_unknown_extra_keys_pass_through(self):
        """Unknown keys in the table are forwarded into the resolved dict unchanged.

        Suspected behaviour: arbitrary keys just flow through (resolve_graphics_preset
        does not filter them out).  Pin this — if the signature ever strips unknowns,
        this test will catch the change.
        """
        result = resolve_graphics_preset({"gfx_future_knob": "some_value"})
        # Pin: unknown key IS present in output (current pass-through behaviour)
        assert "gfx_future_knob" in result
        assert result["gfx_future_knob"] == "some_value"

    def test_preset_key_consumed_not_forwarded(self):
        """'preset' itself must not appear as a key in the resolved dict."""
        result = resolve_graphics_preset({"preset": "low"})
        assert "preset" not in result

    def test_unknown_preset_name_emits_warning(self):
        """An invalid preset name must emit a warning (never raise).

        NOTE (suspected behaviour): resolve_graphics_preset lowercases the
        preset name before validation, so the warning text contains the
        lowercased form ('extreme') not the original casing ('EXTREME').
        This is pinned as current behaviour.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_graphics_preset({"preset": "EXTREME"})
        assert result["gfx_preset"] == "high"
        # Pin current behaviour: warning uses the LOWERCASED form of the input.
        assert any("extreme" in str(w.message) for w in caught), \
            "expected warning mentioning the (lowercased) bad preset name"

    def test_override_beats_invalid_preset_fallback(self):
        """Even when the preset is invalid (→ 'high'), explicit overrides still win."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = resolve_graphics_preset({"preset": "bogus", "gfx_cloud_steps": 1})
        assert result["gfx_cloud_steps"] == 1

    def test_none_table_returns_high_preset(self):
        """None argument (no [graphics] table at all) → 'high'."""
        result = resolve_graphics_preset(None)
        assert result["gfx_preset"] == "high"


# ===========================================================================
# load_config with a TOML file that applies targeted overrides
# (complements test_graphics_config.py's test_load_config_end_to_end)
# ===========================================================================

class TestLoadConfigToml:
    def test_top_level_overrides(self, tmp_path):
        """Top-level keys (not under a table) override the matching Config field."""
        toml = tmp_path / "config.toml"
        toml.write_text(
            "world_seed = 42\n"
            "view_distance_chunks = 10\n"
            "fixed_dt = 0.01\n",
            encoding="utf-8",
        )
        cfg = load_config(str(toml))
        assert cfg.world_seed == 42
        assert cfg.view_distance_chunks == 10
        assert cfg.fixed_dt == pytest.approx(0.01)
        # Unset fields stay at defaults
        assert cfg.chunk_size == 32
        assert cfg.voxel_size == 0.5

    def test_debug_table_overrides(self, tmp_path):
        """Keys under [debug] flatten into Config correctly."""
        toml = tmp_path / "config.toml"
        toml.write_text(
            "[debug]\n"
            "show_fps = false\n"
            "show_chunk_borders = true\n",
            encoding="utf-8",
        )
        cfg = load_config(str(toml))
        assert cfg.show_fps is False
        assert cfg.show_chunk_borders is True

    def test_sky_table_overrides(self, tmp_path):
        """Keys under [sky] flatten into Config correctly."""
        toml = tmp_path / "config.toml"
        toml.write_text(
            "[sky]\n"
            "sky_star_count = 1000\n"
            "sky_cloud_altitude_m = 200.0\n",
            encoding="utf-8",
        )
        cfg = load_config(str(toml))
        assert cfg.sky_star_count == 1000
        assert cfg.sky_cloud_altitude_m == pytest.approx(200.0)

    def test_unknown_toml_keys_ignored(self, tmp_path):
        """Extra TOML keys not in Config must be silently dropped (no error)."""
        toml = tmp_path / "config.toml"
        toml.write_text(
            "totally_unknown_key = 999\n"
            "world_seed = 77\n",
            encoding="utf-8",
        )
        cfg = load_config(str(toml))
        assert cfg.world_seed == 77
        assert not hasattr(cfg, "totally_unknown_key")

    def test_derived_properties_correct_after_toml_load(self, tmp_path):
        """chunk_meters and light_cell_meters still satisfy the formula after a TOML load."""
        toml = tmp_path / "config.toml"
        # Write a config that doesn't touch chunk_size / voxel_size
        toml.write_text("world_seed = 1\n", encoding="utf-8")
        cfg = load_config(str(toml))
        assert cfg.chunk_meters == float(cfg.chunk_size) * cfg.voxel_size
        assert cfg.light_cell_meters == cfg.voxel_size * float(cfg.light_grid_scale)

    def test_missing_path_returns_default_config(self):
        """Missing file → Config() defaults (verified via a few key fields)."""
        cfg = load_config("nonexistent-totally-missing.toml")
        default = Config()
        # Must be equal to a fresh default (already ensured by frozen equality)
        assert cfg == default


# ===========================================================================
# Key field default types and values
# ===========================================================================

class TestDefaultFieldTypes:
    def test_world_seed_is_int(self):
        cfg = _default()
        assert isinstance(cfg.world_seed, int)
        assert cfg.world_seed == 1337

    def test_view_distance_chunks_is_positive_int(self):
        cfg = _default()
        assert isinstance(cfg.view_distance_chunks, int)
        assert cfg.view_distance_chunks > 0

    def test_fixed_dt_is_float(self):
        cfg = _default()
        assert isinstance(cfg.fixed_dt, float)
        assert cfg.fixed_dt == pytest.approx(0.02)

    def test_show_fps_is_bool(self):
        assert isinstance(_default().show_fps, bool)

    def test_mesh_style_is_str(self):
        cfg = _default()
        assert isinstance(cfg.mesh_style, str)
        assert cfg.mesh_style == "faceted"

    def test_sky_star_count_is_int(self):
        cfg = _default()
        assert isinstance(cfg.sky_star_count, int)
        assert cfg.sky_star_count == 2500

    def test_world_size_m_is_float(self):
        cfg = _default()
        assert isinstance(cfg.world_size_m, float)
        assert cfg.world_size_m == pytest.approx(1000.0)

    def test_gfx_preset_default_is_high(self):
        assert _default().gfx_preset == "high"

    def test_lighting_backend_default(self):
        assert _default().lighting_backend == "gpu"
