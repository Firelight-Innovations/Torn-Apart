"""
tests/procedural/textures/ground/test_roof_shingle.py
— Tests for fire_engine/procedural/textures/ground/roof_shingle.py.

Covers shape, dtype, alpha, small posterised palette, determinism, palette /
threshold exports, and registration. Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np


def _fresh(seed: int = 1337) -> np.ndarray:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.ground.roof_shingle import RoofShingleDef

    set_world_seed(seed)
    register(RoofShingleDef())
    clear_cache()
    return get("roof_shingle")


def test_shape_dtype_alpha():
    arr = _fresh()
    assert arr.shape == (64, 64, 4)
    assert arr.dtype == np.uint8
    assert (arr[..., 3] == 255).all()


def test_custom_size():
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.registry import clear_cache, get, register
    from fire_engine.procedural.textures.ground.roof_shingle import RoofShingleDef

    set_world_seed(1)
    register(RoofShingleDef())
    clear_cache()
    arr = get("roof_shingle", width=128, height=96)
    assert arr.shape == (96, 128, 4)


def test_small_palette():
    arr = _fresh()
    rgb = arr[..., :3].reshape(-1, 3)
    assert len(np.unique(rgb, axis=0)) <= 16


def test_palette_and_thresholds_exports():
    from fire_engine.procedural.textures.ground.roof_shingle import (
        ROOF_SHINGLE_PALETTE,
        ROOF_SHINGLE_THRESHOLDS,
    )

    assert ROOF_SHINGLE_PALETTE.dtype == np.uint8
    assert ROOF_SHINGLE_PALETTE.shape[1] == 3
    assert len(ROOF_SHINGLE_THRESHOLDS) == len(ROOF_SHINGLE_PALETTE) - 1
    assert (np.diff(ROOF_SHINGLE_THRESHOLDS) > 0).all()


def test_determinism():
    a = _fresh(seed=42).copy()
    b = _fresh(seed=42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _fresh(seed=7))


def test_registered_name():
    from fire_engine.procedural.textures.ground.roof_shingle import RoofShingleDef

    assert RoofShingleDef.name == "roof_shingle"
