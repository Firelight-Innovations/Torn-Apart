"""
tests/procedural/textures/ground/test_wasteland_ground.py
— Tests for fire_engine/procedural/textures/ground/wasteland_ground.py.

Covers shape, dtype, alpha, determinism, and custom params.
Extracted from tests/test_procedural.py (TestRegistryRoundTrip + TestDeterminism).
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_wasteland(seed: int = 1337) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.ground.wasteland_ground import WastelandGroundDef

    set_world_seed(seed)
    register(WastelandGroundDef())  # evicts old cache entries for this name
    clear_cache()
    return get("wasteland_ground")


class TestWastelandGroundShape:
    def test_default_shape(self):
        arr = _fresh_wasteland()
        assert arr.shape == (256, 256, 4), f"Expected (256,256,4), got {arr.shape}"

    def test_default_dtype(self):
        arr = _fresh_wasteland()
        assert arr.dtype == np.uint8

    def test_alpha_all_opaque(self):
        arr = _fresh_wasteland()
        assert (arr[..., 3] == 255).all(), "Alpha channel should be all 255"

    def test_rgb_in_range(self):
        arr = _fresh_wasteland()
        assert arr[..., :3].min() >= 0
        assert arr[..., :3].max() <= 255

    def test_custom_size(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import clear_cache, get, register
        from fire_engine.procedural.textures.ground.wasteland_ground import WastelandGroundDef

        set_world_seed(1)
        register(WastelandGroundDef())
        clear_cache()
        arr = get("wasteland_ground", width=64, height=32)
        assert arr.shape == (32, 64, 4)
        assert arr.dtype == np.uint8


class TestWastelandGroundDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_wasteland(seed=42).copy()
        arr2 = _fresh_wasteland(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_wasteland(seed=1).copy()
        arr_b = _fresh_wasteland(seed=2)
        assert not np.array_equal(arr_a, arr_b)

    def test_three_runs_identical(self):
        results = [_fresh_wasteland(seed=999).copy() for _ in range(3)]
        assert np.array_equal(results[0], results[1])
        assert np.array_equal(results[1], results[2])


class TestWastelandGroundRegistration:
    def test_name_is_wasteland_ground(self):
        from fire_engine.procedural.textures.ground.wasteland_ground import WastelandGroundDef

        assert WastelandGroundDef.name == "wasteland_ground"

    def test_registered_via_get(self):
        """WastelandGroundDef auto-registers at import; accessible via get()."""
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("wasteland_ground")
        assert arr.shape[2] == 4
