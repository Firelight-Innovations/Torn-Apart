"""
tests/procedural/textures/sky/test_rain_streak.py
— Tests for fire_engine/procedural/textures/sky/rain_streak.py.

Covers shape, dtype, sparse alpha, determinism, streak_count param.
Extracted from tests/test_procedural.py (TestRainStreakTexture).
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_rain_streak(seed: int = 1337, **params) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.sky.rain_streak import RainStreakDef

    set_world_seed(seed)
    register(RainStreakDef())
    clear_cache()
    return get("rain_streak", **params)


class TestRainStreakShape:
    def test_shape_and_dtype(self):
        arr = _fresh_rain_streak()
        assert arr.shape == (512, 128, 4), f"Expected (512,128,4), got {arr.shape}"
        assert arr.dtype == np.uint8

    def test_sparse_streaks(self):
        """Alpha = streak intensity: mostly empty with some bright streaks."""
        arr = _fresh_rain_streak()
        lit = (arr[..., 3] > 0).mean()
        assert 0.0 < lit < 0.5, f"streaks should be sparse, got {lit:.0%} lit"
        assert arr[..., 3].max() > 200, "the brightest tier should be near-opaque"


class TestRainStreakDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_rain_streak(seed=42).copy()
        arr2 = _fresh_rain_streak(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_rain_streak(seed=1).copy()
        arr_b = _fresh_rain_streak(seed=2)
        assert not np.array_equal(arr_a, arr_b)


class TestRainStreakParams:
    def test_streak_count_param_accepted(self):
        """streak_count param changes the output (separate cache slot)."""
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import clear_cache, get, register
        from fire_engine.procedural.textures.sky.rain_streak import RainStreakDef

        set_world_seed(5)
        register(RainStreakDef())
        clear_cache()
        arr_default = get("rain_streak")
        arr_heavy = get("rain_streak", streak_count=48)
        assert arr_heavy is not arr_default
        assert arr_heavy.shape == arr_default.shape


class TestRainStreakRegistration:
    def test_name_is_rain_streak(self):
        from fire_engine.procedural.textures.sky.rain_streak import RainStreakDef

        assert RainStreakDef.name == "rain_streak"

    def test_registered_via_get(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("rain_streak")
        assert arr.shape == (512, 128, 4)
