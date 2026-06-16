"""
tests/procedural/textures/test_base.py — Tests for procedural/textures/base.py.

Covers ProceduralTextureDef, value_noise, pixel_noise.
Extracted verbatim from tests/test_procedural.py (TestValueNoise, TestPixelNoise).
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# posterise helper
# ---------------------------------------------------------------------------


class TestPosterise:
    def _palette(self):
        return np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=np.uint8)

    def test_buckets_by_threshold(self):
        from fire_engine.procedural.textures.base import posterise

        field = np.array([[0.0, 0.5, 0.99]], dtype=np.float32)
        out = posterise(field, self._palette(), np.array([0.33, 0.66], dtype=np.float32))
        assert out.shape == (1, 3, 3)
        assert np.array_equal(out[0, 0], [0, 0, 0])  # below first threshold
        assert np.array_equal(out[0, 1], [128, 128, 128])  # middle bucket
        assert np.array_equal(out[0, 2], [255, 255, 255])  # top bucket

    def test_output_is_uint8_rgb(self):
        from fire_engine.procedural.textures.base import posterise

        field = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        out = posterise(field, self._palette(), np.array([0.33, 0.66], dtype=np.float32))
        assert out.dtype == np.uint8
        assert out.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# value_noise helper
# ---------------------------------------------------------------------------


class TestValueNoise:
    def test_shape_and_dtype(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import value_noise

        set_world_seed(0)
        rng = for_domain("test", "noise")
        h = value_noise(rng, (64, 32), octaves=3)
        assert h.shape == (64, 32)
        assert h.dtype == np.float32

    def test_range(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import value_noise

        set_world_seed(0)
        rng = for_domain("test", "range")
        h = value_noise(rng, (128, 128), octaves=4)
        assert float(h.min()) >= 0.0, f"min={h.min()} below 0"
        assert float(h.max()) <= 1.0, f"max={h.max()} above 1"

    def test_deterministic(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import value_noise

        set_world_seed(7)
        h1 = value_noise(for_domain("test", "det"), (32, 32))
        set_world_seed(7)
        h2 = value_noise(for_domain("test", "det"), (32, 32))
        assert np.array_equal(h1, h2)

    def test_different_seeds_differ(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import value_noise

        set_world_seed(1)
        h1 = value_noise(for_domain("test", "diff"), (32, 32))
        set_world_seed(2)
        h2 = value_noise(for_domain("test", "diff"), (32, 32))
        assert not np.array_equal(h1, h2)


# ---------------------------------------------------------------------------
# pixel_noise helper
# ---------------------------------------------------------------------------


class TestPixelNoise:
    def test_shape_and_dtype(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import pixel_noise

        set_world_seed(0)
        rng = for_domain("test", "pixel_noise_shape")
        pn = pixel_noise(rng, (64, 32), octaves=3)
        assert pn.shape == (64, 32)
        assert pn.dtype == np.float32

    def test_range(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import pixel_noise

        set_world_seed(0)
        rng = for_domain("test", "pixel_noise_range")
        pn = pixel_noise(rng, (128, 128), octaves=3)
        assert float(pn.min()) >= 0.0, f"min={pn.min()} below 0"
        assert float(pn.max()) <= 1.0, f"max={pn.max()} above 1"

    def test_deterministic(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import pixel_noise

        set_world_seed(7)
        pn1 = pixel_noise(for_domain("test", "pn_det"), (32, 32))
        set_world_seed(7)
        pn2 = pixel_noise(for_domain("test", "pn_det"), (32, 32))
        assert np.array_equal(pn1, pn2), "pixel_noise must be deterministic for same seed"

    def test_different_seeds_differ(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import pixel_noise

        set_world_seed(1)
        pn1 = pixel_noise(for_domain("test", "pn_diff"), (32, 32))
        set_world_seed(2)
        pn2 = pixel_noise(for_domain("test", "pn_diff"), (32, 32))
        assert not np.array_equal(pn1, pn2)

    def test_blocks_are_constant(self):
        """Nearest-neighbour: adjacent pixels in same coarse cell must be equal."""
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import pixel_noise

        set_world_seed(5)
        pn = pixel_noise(for_domain("test", "pn_blocks"), (8, 8), octaves=1, base_freq=2)
        assert (
            np.array_equal(pn[0], pn[1])
            and np.array_equal(pn[1], pn[2])
            and np.array_equal(pn[2], pn[3])
        ), "Rows 0-3 should be identical (same coarse-grid row, nn upsample)"


# ---------------------------------------------------------------------------
# ProceduralTextureDef — domain subclass
# ---------------------------------------------------------------------------


class TestProceduralTextureDef:
    def test_is_subclass_of_procedural_def(self):
        from fire_engine.procedural.defs import ProceduralDef
        from fire_engine.procedural.textures.base import ProceduralTextureDef

        assert issubclass(ProceduralTextureDef, ProceduralDef)

    def test_concrete_subclass_generate_returns_rgba_array(self):
        from fire_engine.core.rng import for_domain, set_world_seed
        from fire_engine.procedural.textures.base import ProceduralTextureDef

        set_world_seed(0)

        class _TinyDef(ProceduralTextureDef):
            name = "_base_tiny"

            def generate(self, rng, **params):
                out = np.zeros((4, 4, 4), dtype=np.uint8)
                out[..., 3] = 255
                return out

        obj = _TinyDef()
        rng = for_domain("test", "_base_tiny")
        result = obj.generate(rng)
        assert result.shape == (4, 4, 4)
        assert result.dtype == np.uint8
        assert (result[..., 3] == 255).all()

    def test_exports_value_noise_and_pixel_noise(self):
        """Both helpers must be importable from the base module."""
        from fire_engine.procedural.textures.base import pixel_noise, value_noise

        assert callable(value_noise)
        assert callable(pixel_noise)
