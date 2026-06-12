"""
tests/test_tree_placement.py — CPU tree/bush placement (zones/tree_placement).

Pins the jittered-grid guarantees (determinism, in-bounds, minimum
spacing), terrain-surface Z decode + sentinel drop, density/cap math,
species-mix parsing/assignment, and the data-texture block layout that
tree.vert / tree_impostor.vert texelFetch (the placement ↔ GLSL contract).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.config import Config, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.zones import (
    HEIGHT_SENTINEL,
    ZoneVolume,
    bake_tree_instances,
    instances_data_block,
    species_mix_from_params,
)

MIX = [("tree_gnarled_oak", 3.0), ("tree_dead", 1.0)]
POOLS = {"tree_gnarled_oak": 8, "tree_dead": 6}


def _flat_chunks(cfg, coords):
    from fire_engine.terrain.generation import generate_chunk

    class _Chunk:
        def __init__(self, materials):
            self.materials = materials

    return {c: _Chunk(generate_chunk(c, cfg)) for c in coords}


@pytest.fixture()
def setup():
    set_world_seed(1337)
    cfg = load_config()
    vol = ZoneVolume(1, "trees", (14.0, -5.0, 6.0), (34.0, 15.0, 14.0),
                     params={"density": 0.05})
    chunks = _flat_chunks(cfg, [(cx, cy, 0)
                                for cx in (0, 1, 2) for cy in (-1, 0)])
    return cfg, vol, chunks


class TestPlacement:
    def test_deterministic(self, setup):
        cfg, vol, chunks = setup
        a = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        b = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        for f in ("x", "y", "z", "yaw", "scale", "phase", "tint",
                  "species_idx", "variant"):
            assert np.array_equal(getattr(a, f), getattr(b, f))

    def test_positions_in_bounds(self, setup):
        cfg, vol, chunks = setup
        inst = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        assert inst.count > 0
        assert (inst.x >= vol.min_corner[0]).all() \
            and (inst.x < vol.max_corner[0]).all()
        assert (inst.y >= vol.min_corner[1]).all() \
            and (inst.y < vol.max_corner[1]).all()

    def test_minimum_spacing(self, setup):
        """The no-overlap invariant: pairwise distance ≥ 0.3 × cell edge."""
        cfg, vol, chunks = setup
        inst = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        density = float(vol.params["density"])
        cell = max(cfg.tree_min_spacing_m, 1.0 / math.sqrt(density))
        pts = np.stack([inst.x, inst.y], axis=1)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        d[np.diag_indices(inst.count)] = np.inf
        assert d.min() >= 0.3 * cell - 1e-4

    def test_z_matches_height_field(self, setup):
        """Flat demo ground tops at z=8 inside the volume's [6, 14] window."""
        cfg, vol, chunks = setup
        inst = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        # R = round((8-6)/8*254) = 64 → z = 6 + 64/254*8 ≈ 8.016 (quantised).
        assert np.allclose(inst.z, 6.0 + 64.0 / 254.0 * 8.0, atol=1e-4)

    def test_sentinel_drops_instances(self, setup):
        cfg, vol, chunks = setup
        inst = bake_tree_instances(vol, cfg, {}, MIX, POOLS, "trees")
        assert inst.count == 0                  # nothing loaded → no ground

    def test_density_and_cap(self, setup):
        cfg, vol, chunks = setup
        # Expected ≈ density × area; jitter grid keeps within ±40%.
        inst = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        expect = 0.05 * vol.area_xy_m2
        assert 0.5 * expect <= inst.count <= 1.6 * expect
        # Zero density → empty.
        v0 = ZoneVolume(2, "trees", vol.min_corner, vol.max_corner,
                        params={"density": 0.0})
        assert bake_tree_instances(v0, cfg, chunks, MIX, POOLS,
                                   "trees").count == 0
        # A tiny cap truncates deterministically.
        small = Config(tree_max_instances=5)
        capped = bake_tree_instances(vol, small, chunks, MIX, POOLS, "trees")
        assert capped.count == 5

    def test_species_and_variant_assignment(self, setup):
        cfg, vol, chunks = setup
        big = ZoneVolume(3, "trees", (0.0, 0.0, 6.0), (64.0, 32.0, 14.0),
                         params={"density": 0.3})
        chunks = _flat_chunks(cfg, [(cx, cy, 0)
                                    for cx in range(0, 4)
                                    for cy in range(0, 2)])
        inst = bake_tree_instances(big, cfg, chunks, MIX, POOLS, "trees")
        assert inst.count > 100
        assert inst.species_names == ("tree_gnarled_oak", "tree_dead")
        assert set(np.unique(inst.species_idx)) <= {0, 1}
        # 3:1 weighting lands near 75 % oaks.
        oak_frac = float((inst.species_idx == 0).mean())
        assert 0.6 < oak_frac < 0.9
        # Variants stay inside each species' pool and cover it.
        for s, name in enumerate(inst.species_names):
            v = inst.variant[inst.species_idx == s]
            assert (v >= 0).all() and (v < POOLS[name]).all()
        assert len(np.unique(inst.variant[inst.species_idx == 0])) >= 4

    def test_bush_kind_uses_bush_config(self, setup):
        cfg, _, chunks = setup
        vol = ZoneVolume(4, "bushes", (14.0, -5.0, 6.0), (34.0, 15.0, 11.0))
        inst = bake_tree_instances(vol, cfg, chunks,
                                   [("bush_scrub", 1.0)], {"bush_scrub": 6},
                                   "bushes")
        # config bush density 0.08 × 400 m² ≈ 32 (jitter tolerance).
        assert 10 <= inst.count <= 60
        # Bush scale jitter range [0.7, 1.3).
        assert (inst.scale >= 0.7).all() and (inst.scale < 1.31).all()


class TestSpeciesMix:
    def test_default(self):
        assert species_mix_from_params({}, "tree_gnarled_oak") \
            == [("tree_gnarled_oak", 1.0)]

    def test_single_species_param(self):
        assert species_mix_from_params({"species": "tree_dead"}, "x") \
            == [("tree_dead", 1.0)]

    def test_weighted_mix(self):
        mix = species_mix_from_params(
            {"species_mix": "tree_gnarled_oak:3, tree_dead:1"}, "x")
        assert mix == [("tree_gnarled_oak", 3.0), ("tree_dead", 1.0)]

    def test_weightless_entry_defaults_to_one(self):
        assert species_mix_from_params({"species_mix": "a:2,b"}, "x") \
            == [("a", 2.0), ("b", 1.0)]

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            species_mix_from_params({"species_mix": " , "}, "x")


class TestDataBlock:
    def test_layout_pin(self, setup):
        """texel(0,i) = (x,y,z,yaw); texel(1,i) = (scale,phase,tint,variant).

        tree.vert / tree_impostor.vert texelFetch EXACTLY this — edit both
        or neither (the data-texture mirror discipline).
        """
        cfg, vol, chunks = setup
        inst = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        block = instances_data_block(inst)
        assert block.dtype == np.float32
        assert block.shape == (inst.count, 2, 4)
        assert np.array_equal(block[:, 0, 0], inst.x)
        assert np.array_equal(block[:, 0, 1], inst.y)
        assert np.array_equal(block[:, 0, 2], inst.z)
        assert np.array_equal(block[:, 0, 3], inst.yaw)
        assert np.array_equal(block[:, 1, 0], inst.scale)
        assert np.array_equal(block[:, 1, 1], inst.phase)
        assert np.array_equal(block[:, 1, 2], inst.tint)
        assert np.array_equal(block[:, 1, 3],
                              inst.variant.astype(np.float32))

    def test_mask_selects(self, setup):
        cfg, vol, chunks = setup
        inst = bake_tree_instances(vol, cfg, chunks, MIX, POOLS, "trees")
        mask = inst.species_idx == 0
        block = instances_data_block(inst, mask)
        assert block.shape[0] == int(mask.sum())
        if block.shape[0]:
            assert np.array_equal(block[:, 0, 0], inst.x[mask])
