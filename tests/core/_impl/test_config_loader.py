"""
tests/core/_impl/test_config_loader.py — Mirror test for
fire_engine/core/_impl/config_loader.py.

Covers:
- GRAPHICS_PRESETS dict structure and content for all four presets
- resolve_graphics_preset: preset selection, overrides, unknown preset fallback
- load_config: TOML parsing, table flattening, graphics preset expansion,
  missing file fallback
- Determinism: same input always yields same output
"""

from __future__ import annotations

import warnings

import pytest

from fire_engine.core._impl.config_loader import (
    GRAPHICS_PRESETS,
    load_config,
    resolve_graphics_preset,
)
from fire_engine.core.config import Config

# ---------------------------------------------------------------------------
# GRAPHICS_PRESETS structure
# ---------------------------------------------------------------------------


class TestGraphicsPresets:
    KNOWN_PRESETS = ("off", "low", "medium", "high")

    def test_all_presets_present(self):
        for name in self.KNOWN_PRESETS:
            assert name in GRAPHICS_PRESETS, f"preset {name!r} missing"

    def test_all_values_are_dicts(self):
        for name, d in GRAPHICS_PRESETS.items():
            assert isinstance(d, dict), f"preset {name!r} value is not a dict"

    def test_high_preset_has_expected_keys(self):
        high = GRAPHICS_PRESETS["high"]
        for key in ("gfx_post_process", "gfx_bloom", "gfx_fxaa", "gfx_clouds"):
            assert key in high, f"'high' preset missing key {key!r}"

    def test_off_preset_disables_post_process(self):
        assert GRAPHICS_PRESETS["off"]["gfx_post_process"] is False

    def test_off_preset_disables_bloom(self):
        assert GRAPHICS_PRESETS["off"]["gfx_bloom"] is False

    def test_high_preset_enables_bloom(self):
        assert GRAPHICS_PRESETS["high"]["gfx_bloom"] is True

    def test_low_cloud_resolution_is_half(self):
        """Low preset uses half-resolution cloud rendering."""
        assert GRAPHICS_PRESETS["low"]["gfx_cloud_resolution_scale"] == pytest.approx(0.5)

    def test_high_cloud_steps_greater_than_low(self):
        high_steps = GRAPHICS_PRESETS["high"]["gfx_cloud_steps"]
        low_steps = GRAPHICS_PRESETS["low"]["gfx_cloud_steps"]
        assert high_steps > low_steps


# ---------------------------------------------------------------------------
# resolve_graphics_preset
# ---------------------------------------------------------------------------


class TestResolveGraphicsPreset:
    def test_none_returns_high(self):
        result = resolve_graphics_preset(None)
        assert result["gfx_preset"] == "high"

    def test_empty_dict_returns_high(self):
        result = resolve_graphics_preset({})
        assert result["gfx_preset"] == "high"

    def test_explicit_preset_key(self):
        for preset in ("off", "low", "medium", "high"):
            result = resolve_graphics_preset({"preset": preset})
            assert result["gfx_preset"] == preset

    def test_preset_key_not_forwarded(self):
        """'preset' itself must not appear as a gfx_ key."""
        result = resolve_graphics_preset({"preset": "low"})
        assert "preset" not in result

    def test_override_beats_preset(self):
        """Explicit gfx_* key overrides the preset's default."""
        result = resolve_graphics_preset({"preset": "low", "gfx_bloom_mips": 99})
        assert result["gfx_bloom_mips"] == 99

    def test_unknown_preset_falls_back_to_high(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_graphics_preset({"preset": "ultra_max"})
        assert result["gfx_preset"] == "high"
        assert any("ultra_max" in str(w.message) for w in caught)

    def test_unknown_preset_is_case_insensitive(self):
        """Preset name is lowercased before lookup; 'HIGH' resolves to 'high'."""
        result = resolve_graphics_preset({"preset": "HIGH"})
        assert result["gfx_preset"] == "high"

    def test_determinism(self):
        """Same table always yields equal result."""
        table = {"preset": "medium", "gfx_cloud_steps": 64}
        a = resolve_graphics_preset(dict(table))
        b = resolve_graphics_preset(dict(table))
        assert a == b

    def test_off_preset_correct_values(self):
        result = resolve_graphics_preset({"preset": "off"})
        assert result["gfx_post_process"] is False
        assert result["gfx_bloom"] is False
        assert result["gfx_rain_mode"] == "off"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_file_returns_default(self):
        """load_config on a non-existent file returns the default Config."""
        cfg = load_config("nonexistent-config-xyz123.toml")
        assert cfg == Config()

    def test_missing_file_deterministic(self):
        """Two calls on a missing file yield equal configs."""
        a = load_config("nonexistent-abc.toml")
        b = load_config("nonexistent-abc.toml")
        assert a == b

    def test_top_level_overrides(self, tmp_path):
        """Top-level TOML keys flatten directly into Config fields."""
        toml = tmp_path / "c.toml"
        toml.write_text("world_seed = 7777\n", encoding="utf-8")
        cfg = load_config(str(toml))
        assert cfg.world_seed == 7777

    def test_debug_table(self, tmp_path):
        """[debug] table keys flatten into Config."""
        toml = tmp_path / "c.toml"
        toml.write_text("[debug]\nshow_fps = false\n", encoding="utf-8")
        cfg = load_config(str(toml))
        assert cfg.show_fps is False

    def test_sky_table(self, tmp_path):
        """[sky] table keys flatten into Config."""
        toml = tmp_path / "c.toml"
        toml.write_text("[sky]\nsky_star_count = 500\n", encoding="utf-8")
        cfg = load_config(str(toml))
        assert cfg.sky_star_count == 500

    def test_graphics_preset_expansion(self, tmp_path):
        """[graphics] table with preset='off' disables bloom."""
        toml = tmp_path / "c.toml"
        toml.write_text('[graphics]\npreset = "off"\n', encoding="utf-8")
        cfg = load_config(str(toml))
        assert cfg.gfx_preset == "off"
        assert cfg.gfx_bloom is False

    def test_graphics_override_within_preset(self, tmp_path):
        """[graphics] override field wins over preset default."""
        toml = tmp_path / "c.toml"
        toml.write_text('[graphics]\npreset = "low"\ngfx_cloud_steps = 200\n', encoding="utf-8")
        cfg = load_config(str(toml))
        assert cfg.gfx_cloud_steps == 200

    def test_unknown_keys_ignored(self, tmp_path):
        """TOML keys not in Config are silently dropped — no AttributeError."""
        toml = tmp_path / "c.toml"
        toml.write_text("completely_unknown = 1\nworld_seed = 5\n", encoding="utf-8")
        cfg = load_config(str(toml))
        assert cfg.world_seed == 5
        assert not hasattr(cfg, "completely_unknown")

    def test_returns_frozen_config(self, tmp_path):
        """load_config always returns a frozen Config instance."""
        import dataclasses

        toml = tmp_path / "c.toml"
        toml.write_text("world_seed = 1\n", encoding="utf-8")
        cfg = load_config(str(toml))
        assert isinstance(cfg, Config)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.world_seed = 0  # type: ignore[misc]

    def test_determinism_same_file(self, tmp_path):
        """load_config called twice on the same file yields equal results."""
        toml = tmp_path / "c.toml"
        toml.write_text('world_seed = 42\n[graphics]\npreset = "medium"\n', encoding="utf-8")
        assert load_config(str(toml)) == load_config(str(toml))
