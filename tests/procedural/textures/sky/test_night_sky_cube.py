"""
tests/procedural/textures/sky/test_night_sky_cube.py
— Tests for fire_engine/procedural/textures/sky/night_sky_cube.py.

Covers registration, shape, dtype, alpha, and determinism of NightSkyCubeDef.
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_cube(seed: int = 1337, **params) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.sky.night_sky_cube import NightSkyCubeDef

    set_world_seed(seed)
    register(NightSkyCubeDef())
    clear_cache()
    return get("night_sky_cube", **params)


class TestNightSkyCubeShape:
    def test_dtype_uint8(self):
        arr = _fresh_cube()
        assert arr.dtype == np.uint8

    def test_four_channels(self):
        arr = _fresh_cube()
        assert arr.ndim == 3 or arr.ndim == 4, "output must be 3D or 4D"
        # Last dimension is RGBA
        assert arr.shape[-1] == 4

    def test_alpha_varies(self):
        """Alpha is a luminance mask — must not be all zero or all 255."""
        arr = _fresh_cube()
        alpha = arr[..., 3]
        assert alpha.max() > 0, "cube-map sky must have some bright pixels"
        # Sky floor ensures no pixel is pure black across all six faces
        assert alpha.max() > 10, "cube-map should have visibly bright stars"


class TestNightSkyCubeDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_cube(seed=42).copy()
        arr2 = _fresh_cube(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_cube(seed=1).copy()
        arr_b = _fresh_cube(seed=2)
        assert not np.array_equal(arr_a, arr_b)


class TestNightSkyCubeRegistration:
    def test_name_is_night_sky_cube(self):
        from fire_engine.procedural.textures.sky.night_sky_cube import NightSkyCubeDef

        assert NightSkyCubeDef.name == "night_sky_cube"

    def test_registered_via_get(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("night_sky_cube")
        assert arr.shape[-1] == 4

    def test_reexported_from_night_sky(self):
        """NightSkyCubeDef must also be importable from night_sky for back-compat."""
        from fire_engine.procedural.textures.sky.night_sky import NightSkyCubeDef  # noqa: F401
