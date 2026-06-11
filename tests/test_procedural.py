"""
tests/test_procedural.py — Test suite for Phase 2: Procedural Registry + Textures.

Test categories
---------------
1. Registry round-trip  — register a def, get it back in the right shape/dtype.
2. Determinism (same seed, fresh registry)  — byte-identical results.
3. Different seed  — different bytes.
4. Shape/dtype/alpha invariants  — (256,256,4), uint8, alpha=255.
5. Cache identity  — two ``get`` calls with the same params return the same object.
6. Custom params  — ``get("wasteland_ground", width=128, height=64)`` works.
7. Unknown name  — ``get("does_not_exist")`` raises KeyError.
8. register_def decorator  — registers a minimal def at decoration time.

No panda3d imports anywhere in this file — pure headless.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry():
    """
    Helper: reset the registry and cache, re-register built-in defs by
    re-importing the procedural package.
    """
    # We need to reset first, then re-trigger auto-registration.
    # The cleanest way without module-reload tricks is to call reset_registry
    # and then re-register the built-in defs directly.
    from torn_apart.procedural.registry import reset_registry, register
    from torn_apart.procedural.textures.wasteland_ground import WastelandGroundDef
    from torn_apart.procedural.textures.night_sky import (
        NightSkyCubeDef,
        NightSkyDef,
    )
    from torn_apart.procedural.textures.rain_streak import RainStreakDef
    from torn_apart.procedural.textures.grass_ground import GrassGroundDef
    from torn_apart.procedural.textures.dirt_ground import DirtGroundDef
    from torn_apart.procedural.textures.moon_surface import MoonSurfaceDef
    from torn_apart.procedural.textures.grass_tuft import GrassTuftDef
    reset_registry()
    register(WastelandGroundDef())
    register(NightSkyDef())
    register(NightSkyCubeDef())
    register(RainStreakDef())
    register(GrassGroundDef())
    register(DirtGroundDef())
    register(MoonSurfaceDef())
    register(GrassTuftDef())


# ---------------------------------------------------------------------------
# 1 & 4 — Registry round-trip + shape/dtype/alpha
# ---------------------------------------------------------------------------

class TestRegistryRoundTrip:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        set_world_seed(1337)
        _fresh_registry()

    def test_wasteland_ground_shape(self):
        from torn_apart.procedural.registry import get
        arr = get("wasteland_ground")
        assert arr.shape == (256, 256, 4), (
            f"Expected (256,256,4), got {arr.shape}"
        )

    def test_wasteland_ground_dtype(self):
        from torn_apart.procedural.registry import get
        arr = get("wasteland_ground")
        assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"

    def test_alpha_channel_all_opaque(self):
        from torn_apart.procedural.registry import get
        arr = get("wasteland_ground")
        assert (arr[..., 3] == 255).all(), (
            "Alpha channel should be all 255 (fully opaque)"
        )

    def test_rgb_values_in_range(self):
        from torn_apart.procedural.registry import get
        arr = get("wasteland_ground")
        assert arr[..., :3].min() >= 0
        assert arr[..., :3].max() <= 255


# ---------------------------------------------------------------------------
# 2 — Determinism: same seed, two fresh caches → byte-identical
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_byte_identical(self):
        """Two get() calls on fresh registries with same seed → identical arrays."""
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(42)
        _fresh_registry()
        arr1 = get("wasteland_ground")

        set_world_seed(42)
        _fresh_registry()
        arr2 = get("wasteland_ground")

        assert np.array_equal(arr1, arr2), (
            "Same seed must produce byte-identical output across fresh registries"
        )

    def test_determinism_across_three_runs(self):
        """Consistency check: three independent generations with seed 9999."""
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        results = []
        for _ in range(3):
            set_world_seed(9999)
            _fresh_registry()
            results.append(get("wasteland_ground").copy())

        assert np.array_equal(results[0], results[1])
        assert np.array_equal(results[1], results[2])


# ---------------------------------------------------------------------------
# 3 — Different seed → different bytes
# ---------------------------------------------------------------------------

class TestDifferentSeed:
    def test_different_seeds_produce_different_output(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(1)
        _fresh_registry()
        arr_a = get("wasteland_ground")

        set_world_seed(2)
        _fresh_registry()
        arr_b = get("wasteland_ground")

        assert not np.array_equal(arr_a, arr_b), (
            "Different world seeds must produce different output"
        )


# ---------------------------------------------------------------------------
# 5 — Cache identity
# ---------------------------------------------------------------------------

class TestCacheIdentity:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        set_world_seed(1337)
        _fresh_registry()

    def test_same_call_returns_same_object(self):
        from torn_apart.procedural.registry import get
        arr1 = get("wasteland_ground")
        arr2 = get("wasteland_ground")
        assert arr1 is arr2, (
            "Two get() calls with identical args must return the same cached object"
        )

    def test_clear_cache_breaks_identity(self):
        from torn_apart.procedural.registry import get, clear_cache
        arr1 = get("wasteland_ground")
        clear_cache()
        arr2 = get("wasteland_ground")
        assert arr1 is not arr2, (
            "After clear_cache(), get() must return a freshly generated object"
        )

    def test_different_params_different_cache_slot(self):
        from torn_apart.procedural.registry import get
        arr_default = get("wasteland_ground")
        arr_custom  = get("wasteland_ground", width=128, height=128)
        assert arr_default is not arr_custom
        assert arr_custom.shape == (128, 128, 4)


# ---------------------------------------------------------------------------
# 6 — Custom params
# ---------------------------------------------------------------------------

class TestCustomParams:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        set_world_seed(77)
        _fresh_registry()

    def test_custom_width_height(self):
        from torn_apart.procedural.registry import get
        arr = get("wasteland_ground", width=64, height=32)
        assert arr.shape == (32, 64, 4)
        assert arr.dtype == np.uint8

    def test_custom_size_deterministic(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(5)
        _fresh_registry()
        a1 = get("wasteland_ground", width=64, height=64)

        set_world_seed(5)
        _fresh_registry()
        a2 = get("wasteland_ground", width=64, height=64)

        assert np.array_equal(a1, a2)


# ---------------------------------------------------------------------------
# 7 — Unknown name raises KeyError
# ---------------------------------------------------------------------------

class TestUnknownName:
    def test_unknown_name_raises_key_error(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get
        set_world_seed(0)
        _fresh_registry()
        with pytest.raises(KeyError, match="does_not_exist"):
            get("does_not_exist")


# ---------------------------------------------------------------------------
# 8 — register_def decorator
# ---------------------------------------------------------------------------

class TestRegisterDefDecorator:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import reset_registry
        set_world_seed(42)
        reset_registry()

    def test_decorator_registers_at_import(self):
        """A class decorated with @register_def is immediately retrievable."""
        from torn_apart.procedural.defs import ProceduralDef, register_def
        from torn_apart.procedural.registry import get

        @register_def
        class MinimalDef(ProceduralDef):
            name = "_test_minimal"

            def generate(self, rng, **params):
                return np.array([42], dtype=np.uint8)

        result = get("_test_minimal")
        assert isinstance(result, np.ndarray)
        assert result[0] == 42

    def test_custom_def_deterministic(self):
        """Custom def uses for_domain RNG → same seed = same output."""
        from torn_apart.procedural.defs import ProceduralDef, register_def
        from torn_apart.procedural.textures.base import value_noise
        from torn_apart.procedural.registry import get, reset_registry
        from torn_apart.core.rng import set_world_seed

        @register_def
        class NoiseDef(ProceduralDef):
            name = "_test_noise"

            def generate(self, rng, **params):
                return value_noise(rng, (16, 16))

        set_world_seed(11)
        reset_registry()
        from torn_apart.procedural.defs import register_def as _rd  # re-apply
        # Re-register after reset
        from torn_apart.procedural.registry import register
        register(NoiseDef())
        a = get("_test_noise").copy()

        set_world_seed(11)
        reset_registry()
        register(NoiseDef())
        b = get("_test_noise").copy()

        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# value_noise helper
# ---------------------------------------------------------------------------

class TestValueNoise:
    def test_shape_and_dtype(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import value_noise
        set_world_seed(0)
        rng = for_domain("test", "noise")
        h = value_noise(rng, (64, 32), octaves=3)
        assert h.shape == (64, 32)
        assert h.dtype == np.float32

    def test_range(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import value_noise
        set_world_seed(0)
        rng = for_domain("test", "range")
        h = value_noise(rng, (128, 128), octaves=4)
        assert float(h.min()) >= 0.0, f"min={h.min()} below 0"
        assert float(h.max()) <= 1.0, f"max={h.max()} above 1"

    def test_deterministic(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import value_noise
        set_world_seed(7)
        h1 = value_noise(for_domain("test","det"), (32, 32))
        set_world_seed(7)
        h2 = value_noise(for_domain("test","det"), (32, 32))
        assert np.array_equal(h1, h2)

    def test_different_seeds_differ(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import value_noise
        set_world_seed(1)
        h1 = value_noise(for_domain("test","diff"), (32, 32))
        set_world_seed(2)
        h2 = value_noise(for_domain("test","diff"), (32, 32))
        assert not np.array_equal(h1, h2)


# ---------------------------------------------------------------------------
# "night_sky" texture (sky feature)
# ---------------------------------------------------------------------------

class TestNightSkyTexture:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        set_world_seed(1337)
        _fresh_registry()

    def test_shape_and_dtype(self):
        from torn_apart.procedural.registry import get
        arr = get("night_sky")
        assert arr.shape == (512, 1024, 4), f"Expected (512,1024,4), got {arr.shape}"
        assert arr.dtype == np.uint8

    def test_alpha_is_luminance_mask(self):
        """Alpha varies (additive-blend mask): dark sky → low, stars → high."""
        from torn_apart.procedural.registry import get
        arr = get("night_sky")
        assert arr[..., 3].max() > 128, "bright stars should give high alpha"
        assert arr[..., 3].min() < 64, "empty sky should give low alpha"

    def test_same_seed_byte_identical(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(42)
        _fresh_registry()
        arr1 = get("night_sky").copy()

        set_world_seed(42)
        _fresh_registry()
        arr2 = get("night_sky")

        assert np.array_equal(arr1, arr2), (
            "Same seed must regenerate night_sky byte-identically"
        )

    def test_different_seeds_differ(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(1)
        _fresh_registry()
        arr_a = get("night_sky").copy()

        set_world_seed(2)
        _fresh_registry()
        arr_b = get("night_sky")

        assert not np.array_equal(arr_a, arr_b)

    def test_star_count_param(self):
        """star_count is accepted and changes the output (separate cache slot)."""
        from torn_apart.procedural.registry import get
        default = get("night_sky")
        custom = get("night_sky", star_count=500)
        assert custom is not default
        assert custom.shape == default.shape
        assert not np.array_equal(custom, default)


# ---------------------------------------------------------------------------
# "rain_streak" texture (sky feature)
# ---------------------------------------------------------------------------

class TestRainStreakTexture:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        set_world_seed(1337)
        _fresh_registry()

    def test_shape_and_dtype(self):
        from torn_apart.procedural.registry import get
        arr = get("rain_streak")
        assert arr.shape == (512, 128, 4), f"Expected (512,128,4), got {arr.shape}"
        assert arr.dtype == np.uint8

    def test_sparse_streaks(self):
        """Alpha = streak intensity: mostly empty with some bright streaks."""
        from torn_apart.procedural.registry import get
        arr = get("rain_streak")
        lit = (arr[..., 3] > 0).mean()
        assert 0.0 < lit < 0.5, f"streaks should be sparse, got {lit:.0%} lit"
        assert arr[..., 3].max() > 200, "the brightest tier should be near-opaque"

    def test_same_seed_byte_identical(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(42)
        _fresh_registry()
        arr1 = get("rain_streak").copy()

        set_world_seed(42)
        _fresh_registry()
        arr2 = get("rain_streak")

        assert np.array_equal(arr1, arr2), (
            "Same seed must regenerate rain_streak byte-identically"
        )

    def test_different_seeds_differ(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(1)
        _fresh_registry()
        arr_a = get("rain_streak").copy()

        set_world_seed(2)
        _fresh_registry()
        arr_b = get("rain_streak")

        assert not np.array_equal(arr_a, arr_b)


# ---------------------------------------------------------------------------
# pixel_noise helper
# ---------------------------------------------------------------------------

class TestPixelNoise:
    def test_shape_and_dtype(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import pixel_noise
        set_world_seed(0)
        rng = for_domain("test", "pixel_noise_shape")
        pn = pixel_noise(rng, (64, 32), octaves=3)
        assert pn.shape == (64, 32)
        assert pn.dtype == np.float32

    def test_range(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import pixel_noise
        set_world_seed(0)
        rng = for_domain("test", "pixel_noise_range")
        pn = pixel_noise(rng, (128, 128), octaves=3)
        assert float(pn.min()) >= 0.0, f"min={pn.min()} below 0"
        assert float(pn.max()) <= 1.0, f"max={pn.max()} above 1"

    def test_deterministic(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import pixel_noise
        set_world_seed(7)
        pn1 = pixel_noise(for_domain("test", "pn_det"), (32, 32))
        set_world_seed(7)
        pn2 = pixel_noise(for_domain("test", "pn_det"), (32, 32))
        assert np.array_equal(pn1, pn2), "pixel_noise must be deterministic for same seed"

    def test_different_seeds_differ(self):
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import pixel_noise
        set_world_seed(1)
        pn1 = pixel_noise(for_domain("test", "pn_diff"), (32, 32))
        set_world_seed(2)
        pn2 = pixel_noise(for_domain("test", "pn_diff"), (32, 32))
        assert not np.array_equal(pn1, pn2)

    def test_blocks_are_constant(self):
        """Nearest-neighbour: adjacent pixels in same coarse cell must be equal."""
        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import pixel_noise
        set_world_seed(5)
        # 8×8 output with base_freq=2 → 2-cell grid → 4px-wide blocks
        pn = pixel_noise(for_domain("test", "pn_blocks"), (8, 8), octaves=1, base_freq=2)
        # The first 4 rows should all be identical (they map to the same coarse row)
        assert np.array_equal(pn[0], pn[1]) and np.array_equal(pn[1], pn[2]) and \
               np.array_equal(pn[2], pn[3]), \
               "Rows 0-3 should be identical (same coarse-grid row, nn upsample)"


# ---------------------------------------------------------------------------
# "grass_ground" texture
# ---------------------------------------------------------------------------

class TestGrassGroundTexture:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        set_world_seed(1337)
        _fresh_registry()

    def test_shape_and_dtype(self):
        from torn_apart.procedural.registry import get
        arr = get("grass_ground")
        assert arr.shape == (64, 64, 4), f"Expected (64,64,4), got {arr.shape}"
        assert arr.dtype == np.uint8

    def test_alpha_all_opaque(self):
        from torn_apart.procedural.registry import get
        arr = get("grass_ground")
        assert (arr[..., 3] == 255).all(), "Alpha must be 255 everywhere"

    def test_same_seed_byte_identical(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(42)
        _fresh_registry()
        arr1 = get("grass_ground").copy()

        set_world_seed(42)
        _fresh_registry()
        arr2 = get("grass_ground")

        assert np.array_equal(arr1, arr2), "Same seed must produce byte-identical grass_ground"

    def test_different_seeds_differ(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(10)
        _fresh_registry()
        arr_a = get("grass_ground").copy()

        set_world_seed(20)
        _fresh_registry()
        arr_b = get("grass_ground")

        assert not np.array_equal(arr_a, arr_b)

    def test_palette_size_small(self):
        """Unique RGB colours should be <= 16 (posterised palette)."""
        from torn_apart.procedural.registry import get
        arr = get("grass_ground")
        rgb = arr[..., :3].reshape(-1, 3)
        unique_colours = len(np.unique(rgb, axis=0))
        assert unique_colours <= 16, (
            f"Expected a small pixel-art palette (<=16 colours), got {unique_colours}"
        )

    def test_custom_size(self):
        from torn_apart.procedural.registry import get
        arr = get("grass_ground", width=128, height=128)
        assert arr.shape == (128, 128, 4)
        assert arr.dtype == np.uint8


# ---------------------------------------------------------------------------
# "dirt_ground" texture
# ---------------------------------------------------------------------------

class TestDirtGroundTexture:
    def setup_method(self):
        from torn_apart.core.rng import set_world_seed
        set_world_seed(1337)
        _fresh_registry()

    def test_shape_and_dtype(self):
        from torn_apart.procedural.registry import get
        arr = get("dirt_ground")
        assert arr.shape == (64, 64, 4), f"Expected (64,64,4), got {arr.shape}"
        assert arr.dtype == np.uint8

    def test_alpha_all_opaque(self):
        from torn_apart.procedural.registry import get
        arr = get("dirt_ground")
        assert (arr[..., 3] == 255).all(), "Alpha must be 255 everywhere"

    def test_same_seed_byte_identical(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(42)
        _fresh_registry()
        arr1 = get("dirt_ground").copy()

        set_world_seed(42)
        _fresh_registry()
        arr2 = get("dirt_ground")

        assert np.array_equal(arr1, arr2), "Same seed must produce byte-identical dirt_ground"

    def test_different_seeds_differ(self):
        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural.registry import get

        set_world_seed(10)
        _fresh_registry()
        arr_a = get("dirt_ground").copy()

        set_world_seed(20)
        _fresh_registry()
        arr_b = get("dirt_ground")

        assert not np.array_equal(arr_a, arr_b)

    def test_palette_size_small(self):
        """Unique RGB colours should be <= 16 (posterised palette)."""
        from torn_apart.procedural.registry import get
        arr = get("dirt_ground")
        rgb = arr[..., :3].reshape(-1, 3)
        unique_colours = len(np.unique(rgb, axis=0))
        assert unique_colours <= 16, (
            f"Expected a small pixel-art palette (<=16 colours), got {unique_colours}"
        )

    def test_custom_size(self):
        from torn_apart.procedural.registry import get
        arr = get("dirt_ground", width=128, height=128)
        assert arr.shape == (128, 128, 4)
        assert arr.dtype == np.uint8
