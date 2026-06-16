"""Graphics-quality config: presets, overrides, and headless guarantees.

Covers the ``[graphics]`` table added for the HDR post-processing + volumetric
cloud pipeline: preset expansion (off/low/medium/high), explicit per-field
overrides, deterministic resolution, invalid-preset fallback, end-to-end
``load_config`` flattening, and the hard rule that ``core.config`` stays
headless (no panda3d).
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from fire_engine.core.config import (
    GRAPHICS_PRESETS,
    Config,
    load_config,
    resolve_graphics_preset,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_equal_high_preset():
    """The Config dataclass defaults must mirror the "high" preset exactly."""
    cfg = Config()
    for key, value in GRAPHICS_PRESETS["high"].items():
        assert getattr(cfg, key) == value, key
    assert cfg.gfx_preset == "high"


def test_resolve_is_deterministic():
    """Same table in → identical dict out (no hidden state / ordering)."""
    table = {"preset": "medium", "gfx_bloom_strength": 0.1}
    assert resolve_graphics_preset(dict(table)) == resolve_graphics_preset(dict(table))


@pytest.mark.parametrize("preset", ["off", "low", "medium", "high"])
def test_each_preset_resolves_to_its_table(preset):
    resolved = resolve_graphics_preset({"preset": preset})
    assert resolved["gfx_preset"] == preset
    for key, value in GRAPHICS_PRESETS[preset].items():
        assert resolved[key] == value


def test_off_preset_disables_post():
    resolved = resolve_graphics_preset({"preset": "off"})
    assert resolved["gfx_post_process"] is False
    assert resolved["gfx_clouds"] is False
    assert resolved["gfx_hdr_format"] == "rgba8"


def test_low_preset_uses_half_res_clouds():
    resolved = resolve_graphics_preset({"preset": "low"})
    assert resolved["gfx_cloud_resolution_scale"] == 0.5
    assert resolved["gfx_cloud_steps"] == 32
    assert resolved["gfx_fxaa"] is False


def test_foliage_refine_preset_wiring():
    """Foliage shadow refinement: off on iGPU-relief presets, on above."""
    assert resolve_graphics_preset({"preset": "off"})["gfx_foliage_shadow_refine"] is False
    assert resolve_graphics_preset({"preset": "low"})["gfx_foliage_shadow_refine"] is False
    assert resolve_graphics_preset({"preset": "medium"})["gfx_foliage_shadow_refine"] is True
    assert resolve_graphics_preset({"preset": "high"})["gfx_foliage_shadow_refine"] is True
    # Explicit override still wins.
    assert (
        resolve_graphics_preset({"preset": "low", "gfx_foliage_shadow_refine": True})[
            "gfx_foliage_shadow_refine"
        ]
        is True
    )


def test_no_table_defaults_to_high():
    assert resolve_graphics_preset(None)["gfx_preset"] == "high"
    assert resolve_graphics_preset({})["gfx_preset"] == "high"


def test_explicit_override_beats_preset():
    """A gfx_* key in the table wins over the preset's value."""
    resolved = resolve_graphics_preset({"preset": "low", "gfx_fxaa": True, "gfx_cloud_steps": 99})
    assert resolved["gfx_fxaa"] is True
    assert resolved["gfx_cloud_steps"] == 99
    # untouched preset values still apply
    assert resolved["gfx_cloud_resolution_scale"] == 0.5


def test_invalid_preset_falls_back_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_graphics_preset({"preset": "ultra-mega"})
    assert resolved["gfx_preset"] == "high"
    assert any("ultra-mega" in str(w.message) for w in caught)


def test_load_config_end_to_end(tmp_path):
    """A [graphics] table in a real TOML file flows through load_config."""
    toml = tmp_path / "config.toml"
    toml.write_text('world_seed = 7\n[graphics]\npreset = "low"\ngfx_cloud_steps = 40\n')
    cfg = load_config(str(toml))
    assert cfg.world_seed == 7  # other tables still load
    assert cfg.gfx_preset == "low"
    assert cfg.gfx_cloud_resolution_scale == 0.5  # from the preset
    assert cfg.gfx_cloud_steps == 40  # explicit override
    assert cfg.gfx_post_process is True


def test_load_config_missing_file_is_high():
    cfg = load_config("does-not-exist-anywhere.toml")
    assert cfg.gfx_preset == "high"
    assert cfg.gfx_post_process is True


def test_core_config_imports_no_panda3d():
    """core.config must stay headless (CLAUDE.md hard rule 1)."""
    probe = (
        "import sys; import fire_engine.core.config as c; "
        "c.load_config('does-not-exist.toml'); "
        "leaked=[m for m in sys.modules if m=='panda3d' or m.startswith('panda3d.')]; "
        "print('LEAK' if leaked else 'clean')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "clean" in proc.stdout
