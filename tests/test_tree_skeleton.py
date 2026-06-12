"""
tests/test_tree_skeleton.py — SkeletonBuilder / validate_skeleton invariants.

Covers the headless core of the 3-D flora pipeline (procedural/flora/):
determinism, structural connectivity (the "floating canopy" bug class),
radius taper, sway monotonicity, pitch-set adherence and the bush path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora import (
    SkeletonBuilder,
    leaf_clusters_at_tips,
    validate_skeleton,
)


def _grow_oak(rng):
    """Reference recipe used across these tests (mirrors gnarled oak)."""
    sb = SkeletonBuilder(rng)
    trunk = sb.trunk(height_m=5.5, base_radius_m=0.28, segments=4,
                     wobble_m=0.35)
    limbs = sb.branches(trunk, count=(3, 5), t_range=(0.35, 0.95),
                        pitch_set=(math.radians(80), math.radians(95)),
                        length_ratio=(0.5, 0.8),
                        length_scale_by_height=(1.0, 0.45),
                        radius_ratio=0.5, upturn_rad=math.radians(18),
                        segments=2)
    twigs = sb.branches(limbs, count=(1, 3), pitch_set=(math.radians(85),),
                        length_ratio=(0.4, 0.6), radius_ratio=0.5,
                        upturn_rad=math.radians(25))
    return sb.skeleton(), trunk, limbs, twigs


class TestDeterminism:
    def test_same_rng_state_byte_identical(self):
        set_world_seed(42)
        sk1, *_ = _grow_oak(for_domain("test", "tree"))
        set_world_seed(42)
        sk2, *_ = _grow_oak(for_domain("test", "tree"))
        assert np.array_equal(sk1.start, sk2.start)
        assert np.array_equal(sk1.end, sk2.end)
        assert np.array_equal(sk1.parent, sk2.parent)
        assert np.array_equal(sk1.radius_start, sk2.radius_start)
        assert np.array_equal(sk1.sway, sk2.sway)

    def test_different_seed_differs(self):
        set_world_seed(42)
        sk1, *_ = _grow_oak(for_domain("test", "tree"))
        set_world_seed(43)
        sk2, *_ = _grow_oak(for_domain("test", "tree"))
        assert sk1.n_segments != sk2.n_segments \
            or not np.array_equal(sk1.end, sk2.end)


class TestStructure:
    def setup_method(self):
        set_world_seed(7)
        self.sk, self.trunk, self.limbs, self.twigs = \
            _grow_oak(for_domain("test", "tree"))

    def test_validates_clean(self):
        validate_skeleton(self.sk)

    def test_base_at_origin(self):
        root = int(self.trunk[0])
        assert np.allclose(self.sk.start[root], 0.0)
        assert self.sk.parent[root] == -1

    def test_corrupted_parent_link_fails(self):
        """The floating-canopy regression: a detached start must be caught."""
        sk = self.sk
        bad = int(self.twigs[0])
        sk.start[bad] += np.array([1.0, 1.0, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="floating"):
            validate_skeleton(sk)

    def test_depth_levels(self):
        assert (self.sk.depth[self.trunk] == 0).all()
        assert (self.sk.depth[self.limbs] == 1).all()
        assert (self.sk.depth[self.twigs] == 2).all()

    def test_radius_taper(self):
        sk = self.sk
        assert (sk.radius_end <= sk.radius_start + 1e-5).all()
        child = np.nonzero(sk.parent >= 0)[0]
        p = sk.parent[child]
        assert (sk.radius_start[child]
                <= np.maximum(sk.radius_start[p], sk.radius_end[p])
                + 1e-5).all()

    def test_sway_range_and_monotone(self):
        sk = self.sk
        assert (sk.sway >= 0.0).all() and (sk.sway <= 1.0).all()
        child = np.nonzero(sk.parent >= 0)[0]
        assert (sk.sway[child] + 1e-6 >= sk.sway[sk.parent[child]]).all()
        # Trunk base barely sways; some tip reaches full weight.
        assert sk.sway[int(self.trunk[0])] < 0.5
        assert np.isclose(sk.sway.max(), 1.0)

    def test_pitch_set_respected(self):
        """First sub-segment directions honour pitch_set within jitter."""
        set_world_seed(11)
        sb = SkeletonBuilder(for_domain("test", "pitch"))
        trunk = sb.trunk(height_m=4.0, base_radius_m=0.2, segments=1,
                         wobble_m=0.0)
        limbs = sb.branches(trunk, count=(8, 8),
                            pitch_set=(math.radians(90),),
                            pitch_jitter_rad=0.05, segments=1)
        sk = sb.skeleton()
        axis = np.array([0.0, 0.0, 1.0])
        d = sk.end[limbs] - sk.start[limbs]
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip(d @ axis, -1.0, 1.0)))
        assert (np.abs(ang - 90.0) < 4.0).all()


class TestBushPath:
    def test_stub_trunk_bush(self):
        set_world_seed(5)
        sb = SkeletonBuilder(for_domain("test", "bush"))
        stub = sb.trunk(height_m=0.15, base_radius_m=0.06, segments=1)
        stems = sb.branches(stub, count=(4, 7), t_range=(0.6, 1.0),
                            pitch_set=(math.radians(50), math.radians(70)),
                            yaw_mode="random", length_m=(0.5, 0.9),
                            radius_ratio=0.7, segments=2)
        sk = sb.skeleton()
        validate_skeleton(sk)
        assert stems.size >= 8                  # 4 stems × 2 sub-segments
        assert sk.end[:, 2].max() < 1.5         # bushes stay low

    def test_yaw_mode_rejects_unknown(self):
        set_world_seed(5)
        sb = SkeletonBuilder(for_domain("test", "bush"))
        trunk = sb.trunk(height_m=1.0, base_radius_m=0.1)
        with pytest.raises(ValueError, match="yaw_mode"):
            sb.branches(trunk, yaw_mode="sideways")


class TestLeafClusters:
    def test_clusters_on_tips_only(self):
        set_world_seed(9)
        rng = for_domain("test", "leaves")
        sk, trunk, limbs, twigs = _grow_oak(rng)
        ids = np.concatenate([limbs, twigs])
        leaves = leaf_clusters_at_tips(sk, ids, rng, radius_m=(0.6, 1.1),
                                       offset_frac=0.35)
        assert leaves.n_clusters > 0
        # Every cluster sits near SOME tip end point (≤ radius × offset √3).
        tips = sk.tip_ids(ids)
        d = np.linalg.norm(leaves.center[:, None, :]
                           - sk.end[tips][None, :, :], axis=2).min(axis=1)
        assert (d <= leaves.radius * 0.35 * math.sqrt(3.0) + 1e-5).all()
        assert (leaves.sway >= 0.85).all() and (leaves.sway <= 1.0).all()

    def test_empty_per_tip_gives_empty(self):
        set_world_seed(9)
        rng = for_domain("test", "leaves")
        sk, _, limbs, twigs = _grow_oak(rng)
        leaves = leaf_clusters_at_tips(sk, np.concatenate([limbs, twigs]),
                                       rng, per_tip=(0, 0))
        assert leaves.n_clusters == 0
