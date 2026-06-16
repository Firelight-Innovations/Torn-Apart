"""
tests/procedural/textures/sprites/test_flower_sprite.py
— Tests for fire_engine/procedural/textures/sprites/flower_sprite.py.

Covers shape, dtype, binary alpha, atlas variants, base row, determinism.
Extracted from tests/test_flora.py (TestFloraSpriteTextures).
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.rng import for_domain, set_world_seed

_SHAPE = (32, 128, 4)
_CELL_W = 32
_N_VARIANTS = 4


def _gen(seed: int, **params) -> np.ndarray:
    set_world_seed(seed)
    from fire_engine.procedural.textures.sprites.flower_sprite import FlowerSpriteDef

    return FlowerSpriteDef().generate(for_domain("procedural", "flower_sprite"), **params)


class TestFlowerSpriteShape:
    def test_shape(self):
        arr = _gen(1337)
        assert arr.shape == _SHAPE, f"Expected {_SHAPE}, got {arr.shape}"

    def test_dtype(self):
        arr = _gen(1337)
        assert arr.dtype == np.uint8

    def test_binary_alpha(self):
        """Discard-rendered cutouts: alpha is exactly 0 or 255."""
        arr = _gen(1337)
        assert ((arr[..., 3] == 0) | (arr[..., 3] == 255)).all(), (
            "flower_sprite alpha must be binary (0 or 255)"
        )

    def test_variants_distinct(self):
        """Atlas cells must not all be identical."""
        arr = _gen(1337)
        cells = [arr[:, k * _CELL_W : (k + 1) * _CELL_W] for k in range(_N_VARIANTS)]
        for k in range(_N_VARIANTS - 1):
            assert not np.array_equal(cells[k], cells[k + 1]), (
                f"Cells {k} and {k + 1} should be distinct"
            )

    def test_bases_on_bottom_row(self):
        """Stems touch the bottom image row so plants stand on the ground."""
        arr = _gen(1337)
        assert (arr[-1, :, 3] == 255).any(), "Bottom row must have some opaque pixels"


class TestFlowerSpriteDeterminism:
    def test_same_seed_identical(self):
        arr1 = _gen(1337)
        arr2 = _gen(1337)
        assert np.array_equal(arr1, arr2)

    def test_different_seed_differs(self):
        arr1 = _gen(1337)
        arr2 = _gen(9999)
        assert not np.array_equal(arr1, arr2)


class TestFlowerSpriteRegistration:
    def test_name_is_flower_sprite(self):
        from fire_engine.procedural.textures.sprites.flower_sprite import FlowerSpriteDef

        assert FlowerSpriteDef.name == "flower_sprite"

    def test_registered_via_get(self):
        # Ensure def is registered (reset_registry in other tests may evict it)
        from fire_engine.procedural import clear_cache, get
        from fire_engine.procedural.registry import register
        from fire_engine.procedural.textures.sprites.flower_sprite import FlowerSpriteDef

        set_world_seed(1337)
        register(FlowerSpriteDef())
        clear_cache()
        assert get("flower_sprite").shape == _SHAPE
