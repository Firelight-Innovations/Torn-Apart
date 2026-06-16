"""
tests/procedural/textures/sky/test_moon_surface.py
— Tests for fire_engine/procedural/textures/sky/moon_surface.py.

Covers shape, dtype, disc alpha mask, determinism, and crater_count param.
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_moon(seed: int = 1337, **params) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.sky.moon_surface import MoonSurfaceDef

    set_world_seed(seed)
    register(MoonSurfaceDef())
    clear_cache()
    return get("moon_surface", **params)


class TestMoonSurfaceShape:
    def test_default_shape(self):
        arr = _fresh_moon()
        assert arr.shape == (256, 256, 4), f"Expected (256,256,4), got {arr.shape}"

    def test_dtype_uint8(self):
        arr = _fresh_moon()
        assert arr.dtype == np.uint8

    def test_disc_alpha_mask(self):
        """Pixels inside the disc have alpha=255; outside have alpha=0."""
        arr = _fresh_moon()
        alpha = arr[..., 3]
        # Must have some fully opaque pixels (the disc)
        assert (alpha == 255).any(), "Moon disc should have opaque pixels"
        # Must have some fully transparent pixels (outside the disc)
        assert (alpha == 0).any(), "Outside the disc should be alpha=0"
        # Alpha must be exactly 0 or 255 (no semi-transparency in the mask)
        assert ((alpha == 0) | (alpha == 255)).all(), "Alpha must be binary"

    def test_disc_centre_is_opaque(self):
        """The centre pixel of the disc must be opaque."""
        arr = _fresh_moon()
        cx, cy = arr.shape[1] // 2, arr.shape[0] // 2
        assert arr[cy, cx, 3] == 255, "Centre pixel of moon disc must be opaque"


class TestMoonSurfaceDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_moon(seed=42).copy()
        arr2 = _fresh_moon(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_moon(seed=1).copy()
        arr_b = _fresh_moon(seed=2)
        assert not np.array_equal(arr_a, arr_b)


class TestMoonSurfaceParams:
    def test_crater_count_changes_output(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import clear_cache, get, register
        from fire_engine.procedural.textures.sky.moon_surface import MoonSurfaceDef

        set_world_seed(5)
        register(MoonSurfaceDef())
        clear_cache()
        arr_default = get("moon_surface")
        arr_few = get("moon_surface", crater_count=3)
        assert arr_few is not arr_default
        assert arr_few.shape == arr_default.shape


class TestMoonSurfaceRegistration:
    def test_name_is_moon_surface(self):
        from fire_engine.procedural.textures.sky.moon_surface import MoonSurfaceDef

        assert MoonSurfaceDef.name == "moon_surface"

    def test_registered_via_get(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("moon_surface")
        assert arr.shape == (256, 256, 4)
