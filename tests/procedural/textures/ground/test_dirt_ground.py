"""
tests/procedural/textures/ground/test_dirt_ground.py
— Tests for fire_engine/procedural/textures/ground/dirt_ground.py.

Covers shape, dtype, alpha, palette size, determinism, palette/thresholds exports.
Extracted from tests/test_procedural.py (TestDirtGroundTexture).
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh_dirt(seed: int = 1337) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.ground.dirt_ground import DirtGroundDef

    set_world_seed(seed)
    register(DirtGroundDef())
    clear_cache()
    return get("dirt_ground")


class TestDirtGroundShape:
    def test_default_shape(self):
        arr = _fresh_dirt()
        assert arr.shape == (64, 64, 4), f"Expected (64,64,4), got {arr.shape}"

    def test_default_dtype(self):
        arr = _fresh_dirt()
        assert arr.dtype == np.uint8

    def test_alpha_all_opaque(self):
        arr = _fresh_dirt()
        assert (arr[..., 3] == 255).all(), "Alpha must be 255 everywhere"

    def test_custom_size(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import clear_cache, get, register
        from fire_engine.procedural.textures.ground.dirt_ground import DirtGroundDef

        set_world_seed(1)
        register(DirtGroundDef())
        clear_cache()
        arr = get("dirt_ground", width=128, height=128)
        assert arr.shape == (128, 128, 4)
        assert arr.dtype == np.uint8


class TestDirtGroundPalette:
    def test_small_palette(self):
        """Unique RGB colours should be <= 16 (posterised palette)."""
        arr = _fresh_dirt()
        rgb = arr[..., :3].reshape(-1, 3)
        unique_colours = len(np.unique(rgb, axis=0))
        assert unique_colours <= 16, (
            f"Expected a small pixel-art palette (<=16 colours), got {unique_colours}"
        )

    def test_palette_export_shape(self):
        from fire_engine.procedural.textures.ground.dirt_ground import DIRT_PALETTE

        assert DIRT_PALETTE.dtype == np.uint8
        assert DIRT_PALETTE.ndim == 2
        assert DIRT_PALETTE.shape[1] == 3

    def test_thresholds_export(self):
        from fire_engine.procedural.textures.ground.dirt_ground import (
            DIRT_PALETTE,
            DIRT_THRESHOLDS,
        )

        assert DIRT_THRESHOLDS.ndim == 1
        # thresholds has one fewer entry than palette colours
        assert len(DIRT_THRESHOLDS) == len(DIRT_PALETTE) - 1
        # thresholds are ascending
        assert (np.diff(DIRT_THRESHOLDS) > 0).all(), "DIRT_THRESHOLDS must be ascending"


class TestDirtGroundDeterminism:
    def test_same_seed_byte_identical(self):
        arr1 = _fresh_dirt(seed=42).copy()
        arr2 = _fresh_dirt(seed=42)
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        arr_a = _fresh_dirt(seed=10).copy()
        arr_b = _fresh_dirt(seed=20)
        assert not np.array_equal(arr_a, arr_b)


class TestDirtGroundRegistration:
    def test_name_is_dirt_ground(self):
        from fire_engine.procedural.textures.ground.dirt_ground import DirtGroundDef

        assert DirtGroundDef.name == "dirt_ground"

    def test_registered_via_get(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import clear_cache, get

        set_world_seed(0)
        clear_cache()
        arr = get("dirt_ground")
        assert arr.shape == (64, 64, 4)
