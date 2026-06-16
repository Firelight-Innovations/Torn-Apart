"""
tests/procedural/textures/sprites/test_leaf_sprite.py
— Tests for fire_engine/procedural/textures/sprites/leaf_sprite.py.

Covers shape, dtype, alpha, atlas variants, determinism.
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.rng import for_domain, set_world_seed

_SHAPE = (32, 96, 4)
_CELL_W = 32
_N_VARIANTS = 3


def _gen(seed: int, **params) -> np.ndarray:
    set_world_seed(seed)
    from fire_engine.procedural.textures.sprites.leaf_sprite import LeafSpriteDef

    return LeafSpriteDef().generate(for_domain("procedural", "leaf_sprite"), **params)


class TestLeafSpriteShape:
    def test_shape(self):
        arr = _gen(1337)
        assert arr.shape == _SHAPE, f"Expected {_SHAPE}, got {arr.shape}"

    def test_dtype(self):
        arr = _gen(1337)
        assert arr.dtype == np.uint8

    def test_has_opaque_pixels(self):
        """Each leaf cell should contain opaque pixels (the leaf body)."""
        arr = _gen(1337)
        assert (arr[..., 3] == 255).any(), "leaf_sprite must have opaque pixels"

    def test_has_transparent_pixels(self):
        """Each leaf is not a solid block — some transparency required."""
        arr = _gen(1337)
        assert (arr[..., 3] == 0).any(), "leaf_sprite must have transparent pixels"

    def test_variants_distinct(self):
        """Three distinct hue variants — adjacent cells must not be identical."""
        arr = _gen(1337)
        cells = [arr[:, k * _CELL_W : (k + 1) * _CELL_W] for k in range(_N_VARIANTS)]
        for k in range(_N_VARIANTS - 1):
            assert not np.array_equal(cells[k], cells[k + 1]), (
                f"Leaf cells {k} and {k + 1} should be distinct hue variants"
            )


class TestLeafSpriteDeterminism:
    def test_same_seed_identical(self):
        arr1 = _gen(1337)
        arr2 = _gen(1337)
        assert np.array_equal(arr1, arr2)

    def test_different_seed_differs(self):
        arr1 = _gen(1337)
        arr2 = _gen(9999)
        assert not np.array_equal(arr1, arr2)


class TestLeafSpriteRegistration:
    def test_name_is_leaf_sprite(self):
        from fire_engine.procedural.textures.sprites.leaf_sprite import LeafSpriteDef

        assert LeafSpriteDef.name == "leaf_sprite"

    def test_registered_via_get(self):
        from fire_engine.procedural import clear_cache, get
        from fire_engine.procedural.registry import register
        from fire_engine.procedural.textures.sprites.leaf_sprite import LeafSpriteDef

        set_world_seed(1337)
        register(LeafSpriteDef())
        clear_cache()
        arr = get("leaf_sprite")
        assert arr.shape == _SHAPE
