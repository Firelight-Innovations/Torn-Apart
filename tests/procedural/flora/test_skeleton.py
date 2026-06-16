"""
tests/procedural/flora/test_skeleton.py
— Tests for fire_engine/procedural/flora/skeleton.py.

Covers SkeletonBuilder, trunk(), branches(), skeleton(), and validate_skeleton.
Headless — no panda3d imports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 42) -> np.random.Generator:
    set_world_seed(seed)
    return for_domain("test", "skeleton", seed)


def _simple_tree(rng: np.random.Generator) -> TreeSkeleton:
    """Grow a minimal trunk + limbs + twigs skeleton."""
    sb = SkeletonBuilder(rng)
    trunk = sb.trunk(height_m=5.0, base_radius_m=0.25, segments=4, wobble_m=0.2)
    limbs = sb.branches(
        trunk,
        count=(2, 4),
        t_range=(0.4, 0.9),
        pitch_set=(math.radians(85),),
        length_ratio=(0.4, 0.6),
        radius_ratio=0.5,
    )
    sb.branches(
        limbs,
        count=(1, 3),
        t_range=(0.5, 0.9),
        pitch_set=(math.radians(80),),
        length_ratio=(0.4, 0.7),
        radius_ratio=0.5,
    )
    return sb.skeleton()


# ---------------------------------------------------------------------------
# TreeSkeleton struct
# ---------------------------------------------------------------------------


class TestTreeSkeletonFields:
    def test_field_dtypes(self):
        sk = _simple_tree(_rng())
        assert sk.parent.dtype == np.int32
        assert sk.start.dtype == np.float32
        assert sk.end.dtype == np.float32
        assert sk.radius_start.dtype == np.float32
        assert sk.radius_end.dtype == np.float32
        assert sk.depth.dtype == np.int32
        assert sk.sway.dtype == np.float32

    def test_field_shapes_consistent(self):
        sk = _simple_tree(_rng())
        S = sk.n_segments
        assert sk.parent.shape == (S,)
        assert sk.start.shape == (S, 3)
        assert sk.end.shape == (S, 3)
        assert sk.radius_start.shape == (S,)
        assert sk.radius_end.shape == (S,)
        assert sk.depth.shape == (S,)
        assert sk.sway.shape == (S,)

    def test_n_segments_positive(self):
        sk = _simple_tree(_rng())
        assert sk.n_segments > 0

    def test_trunk_has_depth_zero(self):
        sb = SkeletonBuilder(_rng())
        trunk = sb.trunk(height_m=4.0, base_radius_m=0.2, segments=3)
        sk = sb.skeleton()
        trunk_segs = sk.depth[np.array(trunk)]
        assert (trunk_segs == 0).all(), "Trunk segments must have depth=0"

    def test_branches_have_depth_one(self):
        sb = SkeletonBuilder(_rng())
        trunk = sb.trunk(height_m=4.0, base_radius_m=0.2, segments=3)
        limbs = sb.branches(trunk, count=(2, 4), t_range=(0.4, 0.8), pitch_set=(math.radians(80),))
        sk = sb.skeleton()
        limb_segs = sk.depth[np.array(limbs)]
        assert (limb_segs == 1).all(), "First-level branches must have depth=1"


class TestTreeSkeletonProperties:
    def test_sway_range(self):
        sk = _simple_tree(_rng())
        assert (sk.sway >= 0.0).all() and (sk.sway <= 1.0).all()

    def test_radius_tapers(self):
        sk = _simple_tree(_rng())
        assert (sk.radius_end <= sk.radius_start + 1e-5).all(), (
            "radius_end must not exceed radius_start (tapers toward tip)"
        )

    def test_sway_start_root_is_zero(self):
        sk = _simple_tree(_rng())
        sw_start = sk.sway_start()
        root_ids = np.where(sk.parent < 0)[0]
        assert (sw_start[root_ids] == 0.0).all(), "Root segment starts should have sway=0"

    def test_tip_ids_are_leaves(self):
        sk = _simple_tree(_rng())
        tips = sk.tip_ids()
        # No tip is a parent of another segment
        parent_set = set(sk.parent[sk.parent >= 0].tolist())
        for t in tips.tolist():
            assert t not in parent_set, f"Tip {t} should not be a parent of another segment"


# ---------------------------------------------------------------------------
# SkeletonBuilder.trunk
# ---------------------------------------------------------------------------


class TestTrunk:
    def test_trunk_starts_at_origin(self):
        sb = SkeletonBuilder(_rng())
        trunk = sb.trunk(height_m=5.0, base_radius_m=0.3, segments=4)
        sk = sb.skeleton()
        root_id = trunk[0]
        # Trunk root start must be at (or near) the origin
        assert np.linalg.norm(sk.start[root_id]) < 1e-3, "Trunk root must start at origin"

    def test_trunk_height_approximately_correct(self):
        sb = SkeletonBuilder(_rng())
        trunk = sb.trunk(height_m=6.0, base_radius_m=0.3, segments=4, wobble_m=0.0)
        sk = sb.skeleton()
        tip_id = trunk[-1]
        tip_z = float(sk.end[tip_id, 2])
        assert abs(tip_z - 6.0) < 0.5, f"Trunk tip Z should be ~6m, got {tip_z}"

    def test_trunk_root_has_no_parent(self):
        sb = SkeletonBuilder(_rng())
        trunk = sb.trunk(height_m=4.0, base_radius_m=0.2)
        sk = sb.skeleton()
        root_id = trunk[0]
        assert sk.parent[root_id] == -1, "Trunk root must have parent=-1"


# ---------------------------------------------------------------------------
# validate_skeleton
# ---------------------------------------------------------------------------


class TestValidateSkeleton:
    def test_valid_skeleton_does_not_raise(self):
        from fire_engine.procedural.flora.skeleton import validate_skeleton

        sk = _simple_tree(_rng())
        validate_skeleton(sk)  # must not raise

    def test_floating_branch_raises(self):
        from fire_engine.procedural.flora.skeleton import validate_skeleton

        sk = _simple_tree(_rng())
        # Force a floating branch by teleporting a non-root segment
        child_ids = np.where(sk.parent >= 0)[0]
        if len(child_ids) == 0:
            pytest.skip("no child segments to corrupt")
        bad = int(child_ids[0])
        sk.start[bad] += np.array([50.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="floating branch"):
            validate_skeleton(sk)

    def test_sway_outside_range_raises(self):
        from fire_engine.procedural.flora.skeleton import validate_skeleton

        sk = _simple_tree(_rng())
        sk.sway[0] = -0.1
        with pytest.raises(ValueError, match="sway"):
            validate_skeleton(sk)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestSkeletonDeterminism:
    def test_same_seed_identical_n_segments(self):
        sk1 = _simple_tree(_rng(1))
        sk2 = _simple_tree(_rng(1))
        assert sk1.n_segments == sk2.n_segments

    def test_same_seed_identical_positions(self):
        sk1 = _simple_tree(_rng(1))
        sk2 = _simple_tree(_rng(1))
        assert np.array_equal(sk1.start, sk2.start)
        assert np.array_equal(sk1.end, sk2.end)

    def test_different_seed_different_positions(self):
        sk1 = _simple_tree(_rng(1))
        sk2 = _simple_tree(_rng(99))
        # May differ in structure or positions
        if sk1.n_segments == sk2.n_segments:
            assert not np.array_equal(sk1.start, sk2.start), (
                "Different seeds should produce different skeletons"
            )
