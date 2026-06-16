"""
tests/procedural/textures/ground/test_grass_ground.py
— Tests for fire_engine/procedural/textures/ground/grass_ground.py.

Covers shape, dtype, alpha, palette size, determinism, and custom params.
Extracted from tests/test_procedural.py (TestGrassGroundTexture).
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_grass(seed: int = 1337) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.ground.grass_ground import GrassGroundDef

    set_world_seed(seed)
    register(GrassGroundDef())
    clear_cache()
    return get("grass_ground")


class TestGrassGroundShape:
    def test_default_shape(self):
        arr = _fresh_grass()
        assert arr.shape == (64, 64, 4), f"Expected (64,64,4), got {arr.shape}"

    def test_default_dtype(self):
        arr = _fresh_grass()
        assert arr.dtype == np.uint8

    def test_alpha_all_opaque(self):
        arr = _fresh_grass()
        assert (arr[..., 3] == 255).all(), "Alpha must be 255 everywhere"

    def test_custom_size(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import clear_cache, get, register
        from fire_engine.procedural.textures.ground.grass_ground import GrassGroundDef

        set_world_seed(1)
        register(GrassGroundDef())
        clear_cache()
        arr = get("grass_ground", width=128, height=128)
        assert arr.shape == (128, 128, 4)
        assert arr.dtype == np.uint8


class TestGrassGroundPalette:
    def test_small_palette(self):
        """Unique RGB colours should be <= 16 (posterised palette)."""
        arr = _fresh_grass()
        rgb = arr[..., :3].reshape(-1, 3)
        unique_colours = len(np.unique(rgb, axis=0))
        assert unique_colours <= 16, (
            f"Expected a small pixel-art palette (<=16 colours), got {unique_colours}"
        )


class TestGrassGroundDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_grass(seed=42).copy()
        arr2 = _fresh_grass(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_grass(seed=10).copy()
        arr_b = _fresh_grass(seed=20)
        assert not np.array_equal(arr_a, arr_b)


class TestGrassGroundRegistration:
    def test_name_is_grass_ground(self):
        from fire_engine.procedural.textures.ground.grass_ground import GrassGroundDef

        assert GrassGroundDef.name == "grass_ground"

    def test_registered_via_get(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("grass_ground")
        assert arr.shape == (64, 64, 4)
