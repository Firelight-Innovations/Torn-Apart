"""
tests/world/terrain/test_generation.py — Generation determinism, flatness, bounds.
Headless: no panda3d imports.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.generation import generate_chunk


@pytest.fixture
def cfg():
    return load_config()


class TestFlatGeneration:
    def test_same_coord_byte_identical(self, cfg):
        a = generate_chunk((2, -3, 1), cfg)
        b = generate_chunk((2, -3, 1), cfg)
        assert np.array_equal(a, b)
        assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()

    def test_shape_dtype(self, cfg):
        m = generate_chunk((0, 0, 0), cfg)
        assert m.shape == (32, 32, 32)
        assert m.dtype == np.uint8

    def test_seed_independent(self, cfg):
        """Baseline terrain is flat/authored: it does NOT depend on world_seed."""
        set_world_seed(1)
        a = generate_chunk((0, 0, 0), cfg)
        set_world_seed(99999)
        b = generate_chunk((0, 0, 0), cfg)
        assert np.array_equal(a, b)

    def test_ground_chunk_is_flat(self, cfg):
        """A ground-straddling chunk is solid below ground_height_m, air above.

        Every (x,y) column must be a single solid→air run (no holes/overhangs):
        once air begins going up it never returns to solid.
        """
        m = generate_chunk((0, 0, 0), cfg)  # world z ∈ [0, 16)
        vs = cfg.voxel_size
        ground = cfg.ground_height_m
        zc = (np.arange(32) + 0.5) * vs  # voxel-centre world Z
        expected_col = zc < ground  # (32,) bool over z
        solid = m > 0
        # Within this chunk's footprint (near origin) every column equals the
        # same flat profile.
        assert np.all(solid == expected_col[None, None, :])
        # Sanity: there IS both solid and air in a straddling chunk.
        assert solid.any() and (~solid).any()

    def test_below_ground_fully_solid(self, cfg):
        """A chunk entirely below the ground surface is fully solid (in bounds)."""
        m = generate_chunk((0, 0, -1), cfg)  # world z ∈ [-16, 0), all < 8
        assert np.all(m > 0)

    def test_above_ground_fully_air(self, cfg):
        """A chunk entirely above the ground surface is empty air."""
        m = generate_chunk((0, 0, 2), cfg)  # world z ∈ [32, 48), all > 8
        assert np.all(m == 0)

    def test_outside_world_footprint_is_air(self, cfg):
        """Beyond the world_size_m footprint there is no ground, only air."""
        half_chunks = int((cfg.world_size_m * 0.5) // cfg.chunk_meters) + 2
        m = generate_chunk((half_chunks, 0, -1), cfg)  # well past +X half-extent
        assert np.all(m == 0)

    def test_footprint_is_centred_on_origin(self, cfg):
        """Both sides of the origin are ground; symmetry confirms centring."""
        east = generate_chunk((0, 0, -1), cfg)  # x ∈ [0, 16)
        west = generate_chunk((-1, 0, -1), cfg)  # x ∈ [-16, 0)
        assert np.all(east > 0) and np.all(west > 0)

    def test_surface_height_is_flat_constant(self, cfg):
        from fire_engine.world.terrain.generation import surface_height

        wx = np.array([0.0, 8.0, -300.0])[:, None]
        wy = np.array([0.0, -50.0])[None, :]
        surf = surface_height(wx, wy, cfg)
        assert surf.shape == (3, 2)
        assert np.all(surf == cfg.ground_height_m)
