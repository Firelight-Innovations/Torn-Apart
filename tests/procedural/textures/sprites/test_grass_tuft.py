"""
tests/procedural/textures/sprites/test_grass_tuft.py
— Tests for fire_engine/procedural/textures/sprites/grass_tuft.py.

Covers shape, dtype, binary alpha, bases on bottom row, determinism.
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.rng import for_domain, set_world_seed


def _gen(seed: int, **params) -> np.ndarray:
    set_world_seed(seed)
    from fire_engine.procedural.textures.sprites.grass_tuft import GrassTuftDef

    return GrassTuftDef().generate(for_domain("procedural", "grass_tuft"), **params)


class TestGrassTuftShape:
    def test_default_shape(self):
        arr = _gen(1337)
        assert arr.shape == (32, 32, 4), f"Expected (32,32,4), got {arr.shape}"

    def test_dtype(self):
        arr = _gen(1337)
        assert arr.dtype == np.uint8

    def test_binary_alpha(self):
        """Alpha cutout: alpha is exactly 0 or 255 only."""
        arr = _gen(1337)
        assert ((arr[..., 3] == 0) | (arr[..., 3] == 255)).all(), (
            "grass_tuft alpha must be binary (0 or 255)"
        )

    def test_bases_on_bottom_row(self):
        """Blade bases sit on the bottom image row (V=0 after upload flip)."""
        arr = _gen(1337)
        assert (arr[-1, :, 3] == 255).any(), "Bottom row must have opaque pixels (blade bases)"

    def test_some_transparent_pixels(self):
        """The texture is not a solid block — there must be transparent pixels."""
        arr = _gen(1337)
        assert (arr[..., 3] == 0).any(), "grass_tuft must have some transparent pixels"


class TestGrassTuftDeterminism:
    def test_same_seed_identical(self):
        arr1 = _gen(1337)
        arr2 = _gen(1337)
        assert np.array_equal(arr1, arr2)

    def test_different_seed_differs(self):
        arr1 = _gen(1337)
        arr2 = _gen(9999)
        assert not np.array_equal(arr1, arr2)


class TestGrassTuftRegistration:
    def test_name_is_grass_tuft(self):
        from fire_engine.procedural.textures.sprites.grass_tuft import GrassTuftDef

        assert GrassTuftDef.name == "grass_tuft"

    def test_registered_via_get(self):
        from fire_engine.procedural import clear_cache, get
        from fire_engine.procedural.registry import register
        from fire_engine.procedural.textures.sprites.grass_tuft import GrassTuftDef

        set_world_seed(1337)
        register(GrassTuftDef())
        clear_cache()
        arr = get("grass_tuft")
        assert arr.shape == (32, 32, 4)
