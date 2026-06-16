"""
tests/core/test_config.py — Mirror test for fire_engine/core/config.py.

Covers:
- Config dataclass construction and frozen semantics
- Derived properties chunk_meters and light_cell_meters
- Re-exported symbols (load_config, resolve_graphics_preset, GRAPHICS_PRESETS)
- Default field values match CLAUDE.md documented invariants
- Config equality and field types
"""

from __future__ import annotations

import dataclasses

import pytest

from fire_engine.core.config import (
    GRAPHICS_PRESETS,
    Config,
    load_config,
    resolve_graphics_preset,
)


class TestConfigConstruction:
    def test_default_construction(self):
        """Config() constructs with documented defaults."""
        cfg = Config()
        assert cfg.world_seed == 1337
        assert cfg.chunk_size == 32
        assert cfg.voxel_size == 0.5

    def test_frozen_raises_on_set(self):
        """Frozen dataclass must raise FrozenInstanceError on mutation."""
        cfg = Config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.world_seed = 0  # type: ignore[misc]

    def test_equality(self):
        """Two default instances must be equal."""
        assert Config() == Config()

    def test_replace_produces_new_instance(self):
        """dataclasses.replace creates a distinct instance with updated field."""
        cfg = Config()
        cfg2 = dataclasses.replace(cfg, world_seed=99)
        assert cfg2.world_seed == 99
        assert cfg.world_seed == 1337
        assert cfg2 is not cfg


class TestDerivedProperties:
    def test_chunk_meters(self):
        """chunk_meters == chunk_size * voxel_size (CLAUDE.md: 32*0.5 = 16 m)."""
        cfg = Config()
        assert cfg.chunk_meters == float(cfg.chunk_size) * cfg.voxel_size
        assert cfg.chunk_meters == 16.0

    def test_light_cell_meters(self):
        """light_cell_meters == voxel_size * light_grid_scale (default = 1.0 m)."""
        cfg = Config()
        assert cfg.light_cell_meters == cfg.voxel_size * float(cfg.light_grid_scale)
        assert cfg.light_cell_meters == 1.0

    def test_chunk_meters_with_custom_chunk_size(self):
        cfg = dataclasses.replace(Config(), chunk_size=64)
        assert cfg.chunk_meters == 64 * 0.5

    def test_light_cell_meters_with_custom_scale(self):
        cfg = dataclasses.replace(Config(), light_grid_scale=4)
        assert cfg.light_cell_meters == 0.5 * 4


class TestReExportedSymbols:
    def test_graphics_presets_present(self):
        """GRAPHICS_PRESETS dict is re-exported and contains expected preset names."""
        assert isinstance(GRAPHICS_PRESETS, dict)
        for name in ("off", "low", "medium", "high"):
            assert name in GRAPHICS_PRESETS

    def test_load_config_callable(self):
        """load_config is callable and returns a Config for a missing path."""
        cfg = load_config("definitely-does-not-exist-abc.toml")
        assert isinstance(cfg, Config)
        assert cfg == Config()

    def test_resolve_graphics_preset_callable(self):
        """resolve_graphics_preset returns a dict with gfx_preset key."""
        result = resolve_graphics_preset(None)
        assert isinstance(result, dict)
        assert "gfx_preset" in result

    def test_all_public_names_present(self):
        """__all__ must export Config, load_config, resolve_graphics_preset, GRAPHICS_PRESETS."""
        import fire_engine.core.config as mod

        for name in ("Config", "load_config", "resolve_graphics_preset", "GRAPHICS_PRESETS"):
            assert hasattr(mod, name), f"missing: {name}"


class TestDocumentedDefaults:
    def test_voxel_size(self):
        assert Config().voxel_size == 0.5

    def test_chunk_size(self):
        assert Config().chunk_size == 32

    def test_world_seed_type(self):
        assert isinstance(Config().world_seed, int)

    def test_mesh_style(self):
        assert Config().mesh_style == "faceted"

    def test_gfx_preset(self):
        assert Config().gfx_preset == "high"

    def test_lighting_backend(self):
        assert Config().lighting_backend == "gpu"

    def test_profiler_disabled_by_default(self):
        assert Config().profiler_enabled is False
