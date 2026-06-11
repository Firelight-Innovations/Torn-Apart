"""
tests/test_ground_lut.py — world-space procedural ground palette LUT.

The GPU terrain shader colours the ground from a world-space noise value
indexed into a posterised palette LUT (one row per material).  These tests pin
the LUT contract: it must match the texture defs' ``_posterise`` ramp exactly so
the procedural ground and the baked previews agree, be deterministic, and reject
malformed palette/threshold pairs.  Headless — nothing imports panda3d.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.procedural.textures.ground_lut import build_ground_lut
from fire_engine.procedural.textures.grass_ground import (
    GRASS_PALETTE, GRASS_THRESHOLDS, _posterise as _grass_posterise)
from fire_engine.procedural.textures.dirt_ground import (
    DIRT_PALETTE, DIRT_THRESHOLDS)
from fire_engine.terrain.generation import MATERIAL_DIRT, MATERIAL_GRASS


def _entries():
    return {
        MATERIAL_DIRT:  (DIRT_PALETTE,  DIRT_THRESHOLDS),
        MATERIAL_GRASS: (GRASS_PALETTE, GRASS_THRESHOLDS),
    }


def test_shape_dtype_and_alpha():
    lut = build_ground_lut(_entries())
    # rows = max material id + 1 (air row 0 unused), 256 buckets, RGBA.
    assert lut.shape == (MATERIAL_GRASS + 1, 256, 4)
    assert lut.dtype == np.uint8
    assert np.all(lut[..., 3] == 255)          # fully opaque
    assert np.all(lut[0, :, :3] == 0)          # unused air row left black


def test_custom_level_count():
    lut = build_ground_lut(_entries(), levels=64)
    assert lut.shape == (MATERIAL_GRASS + 1, 64, 4)


def test_matches_posterise_ramp():
    """LUT row must equal _posterise evaluated on the same bucket centres."""
    lut = build_ground_lut(_entries())
    ramp = (np.arange(256, dtype=np.float32) + 0.5) / 256.0
    # _posterise takes (H, W); feed the ramp as a single row.
    expected = _grass_posterise(ramp[None, :], GRASS_PALETTE, GRASS_THRESHOLDS)[0]
    np.testing.assert_array_equal(lut[MATERIAL_GRASS, :, :3], expected)


def test_endpoints_are_palette_extremes():
    lut = build_ground_lut(_entries())
    # Bucket 0 -> darkest palette colour; last bucket -> lightest.
    np.testing.assert_array_equal(lut[MATERIAL_GRASS, 0, :3], GRASS_PALETTE[0])
    np.testing.assert_array_equal(lut[MATERIAL_GRASS, -1, :3], GRASS_PALETTE[-1])


def test_deterministic():
    a = build_ground_lut(_entries())
    b = build_ground_lut(_entries())
    np.testing.assert_array_equal(a, b)


def test_rejects_empty():
    with pytest.raises(ValueError):
        build_ground_lut({})


def test_rejects_threshold_palette_mismatch():
    bad_palette = np.array([(0, 0, 0), (255, 255, 255)], dtype=np.uint8)  # 2 colours
    bad_thresholds = np.array([0.3, 0.6], dtype=np.float32)               # needs 1
    with pytest.raises(ValueError):
        build_ground_lut({1: (bad_palette, bad_thresholds)})


def test_rejects_bad_palette_shape():
    with pytest.raises(ValueError):
        build_ground_lut({1: (np.zeros((4,), np.uint8), np.zeros((3,), np.float32))})
