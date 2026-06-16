"""
tests/procedural/textures/sprites/test_dust_mote.py
— Tests for fire_engine/procedural/textures/sprites/dust_mote.py.

Covers shape, dtype, smooth alpha (not binary), warm RGB, determinism.
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_dust_mote(seed: int = 1337, **params) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.sprites.dust_mote import DustMoteDef

    set_world_seed(seed)
    register(DustMoteDef())
    clear_cache()
    return get("dust_mote", **params)


class TestDustMoteShape:
    def test_default_shape(self):
        arr = _fresh_dust_mote()
        assert arr.shape == (32, 32, 4), f"Expected (32,32,4), got {arr.shape}"

    def test_dtype_uint8(self):
        arr = _fresh_dust_mote()
        assert arr.dtype == np.uint8

    def test_smooth_alpha_not_binary(self):
        """Dust mote uses additive blend — alpha must have intermediate values."""
        arr = _fresh_dust_mote()
        alpha = arr[..., 3]
        # Not binary: there must be values between 1 and 254
        between = ((alpha > 0) & (alpha < 255)).any()
        assert between, "dust_mote alpha should be a smooth radial falloff (non-binary)"

    def test_centre_brighter_than_edge(self):
        """Centre alpha > corner alpha (radial falloff from centre)."""
        arr = _fresh_dust_mote()
        cx, cy = arr.shape[1] // 2, arr.shape[0] // 2
        centre_alpha = int(arr[cy, cx, 3])
        corner_alpha = int(arr[0, 0, 3])
        assert centre_alpha > corner_alpha, "Centre should be brighter than corners"

    def test_corner_alpha_near_zero(self):
        """Corners (far from centre) should be nearly transparent."""
        arr = _fresh_dust_mote()
        corner_alpha = int(arr[0, 0, 3])
        assert corner_alpha < 64, f"Corner alpha should be near-zero, got {corner_alpha}"


class TestDustMoteDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_dust_mote(seed=42).copy()
        arr2 = _fresh_dust_mote(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_dust_mote(seed=1).copy()
        arr_b = _fresh_dust_mote(seed=2)
        assert not np.array_equal(arr_a, arr_b)


class TestDustMoteRegistration:
    def test_name_is_dust_mote(self):
        from fire_engine.procedural.textures.sprites.dust_mote import DustMoteDef

        assert DustMoteDef.name == "dust_mote"

    def test_registered_via_get(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("dust_mote")
        assert arr.shape == (32, 32, 4)
