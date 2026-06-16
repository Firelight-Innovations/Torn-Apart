"""
tests/test_plaster_wall.py — the "plaster_wall" building texture def: shape,
dtype, opacity, palette membership, and determinism.

Constructs the def directly (rather than going through the shared registry +
reset_registry) so it never perturbs the global registry other tests rely on.
"""

import numpy as np

from fire_engine.core.rng import for_domain
from fire_engine.procedural.textures.ground.plaster_wall import (
    PLASTER_PALETTE,
    PlasterWallDef,
)


def _gen(seed_keys=("texture", "plaster_wall")) -> np.ndarray:
    return PlasterWallDef().generate(for_domain(*seed_keys))


class TestPlasterWall:
    def test_shape_dtype_opaque(self):
        arr = _gen()
        assert arr.shape == (64, 64, 4)
        assert arr.dtype == np.uint8
        assert np.all(arr[..., 3] == 255)

    def test_custom_size(self):
        arr = PlasterWallDef().generate(for_domain("t"), width=32, height=16)
        assert arr.shape == (16, 32, 4)

    def test_colours_are_from_palette(self):
        rgb = _gen()[..., :3].reshape(-1, 3)
        palette = {tuple(c) for c in PLASTER_PALETTE.tolist()}
        assert {tuple(c) for c in np.unique(rgb, axis=0).tolist()} <= palette

    def test_determinism_same_seed(self):
        a = PlasterWallDef().generate(for_domain("texture", "plaster_wall"))
        b = PlasterWallDef().generate(for_domain("texture", "plaster_wall"))
        assert np.array_equal(a, b)

    def test_uses_multiple_palette_entries(self):
        # The noise should exercise more than one bucket (not a flat fill).
        rgb = _gen()[..., :3].reshape(-1, 3)
        assert np.unique(rgb, axis=0).shape[0] >= 3
