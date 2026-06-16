"""
tests/procedural/textures/sky/test_night_sky.py
— Tests for fire_engine/procedural/textures/sky/night_sky.py.

Covers shape, dtype, alpha-as-luminance, determinism, star_count param.
Extracted from tests/test_procedural.py (TestNightSkyTexture).
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_night_sky(seed: int = 1337, **params) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.sky.night_sky import NightSkyDef

    set_world_seed(seed)
    register(NightSkyDef())
    clear_cache()
    return get("night_sky", **params)


class TestNightSkyShape:
    def test_shape_and_dtype(self):
        arr = _fresh_night_sky()
        assert arr.shape == (512, 1024, 4), f"Expected (512,1024,4), got {arr.shape}"
        assert arr.dtype == np.uint8

    def test_alpha_is_luminance_mask(self):
        """Alpha varies — dark sky has low alpha, stars have high alpha."""
        arr = _fresh_night_sky()
        assert arr[..., 3].max() > 128, "bright stars should give high alpha"
        assert arr[..., 3].min() < 64, "empty sky should give low alpha"


class TestNightSkyDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_night_sky(seed=42).copy()
        arr2 = _fresh_night_sky(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_night_sky(seed=1).copy()
        arr_b = _fresh_night_sky(seed=2)
        assert not np.array_equal(arr_a, arr_b)


class TestNightSkyParams:
    def test_star_count_param_accepted(self):
        """star_count is accepted and changes the output (separate cache slot)."""
        arr_default = _fresh_night_sky()
        arr_custom = _fresh_night_sky(star_count=500)
        assert arr_custom is not arr_default
        assert arr_custom.shape == arr_default.shape
        assert not np.array_equal(arr_custom, arr_default)


class TestNightSkyRegistration:
    def test_name_is_night_sky(self):
        from fire_engine.procedural.textures.sky.night_sky import NightSkyDef

        assert NightSkyDef.name == "night_sky"

    def test_registered_via_get(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("night_sky")
        assert arr.shape == (512, 1024, 4)
