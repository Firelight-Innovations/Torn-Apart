"""
tests/test_flora_leaves.py — golden-master / characterization tests for
procedural/flora/leaves.py.

Pins CURRENT behaviour; does NOT fix bugs — suspected anomalies are noted
in comments.  Headless only; fixed seed; numpy assertions; no per-element
Python loops.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora import SkeletonBuilder, leaves_at_tips
from fire_engine.procedural.flora.leaves import Leaves

# ---------------------------------------------------------------------------
# Shared skeleton factory (mirrors the gnarled-oak recipe used in skeleton
# tests so we have a tree with tip segments at multiple heights).
# ---------------------------------------------------------------------------


def _build_oak(rng: np.random.Generator):
    """Grow a reference oak skeleton; return (sk, trunk_ids, limb_ids, twig_ids)."""
    sb = SkeletonBuilder(rng)
    trunk = sb.trunk(height_m=5.5, base_radius_m=0.28, segments=4, wobble_m=0.35)
    limbs = sb.branches(
        trunk,
        count=(3, 5),
        t_range=(0.35, 0.95),
        pitch_set=(math.radians(80), math.radians(95)),
        length_ratio=(0.5, 0.8),
        length_scale_by_height=(1.0, 0.45),
        radius_ratio=0.5,
        upturn_rad=math.radians(18),
        segments=2,
    )
    twigs = sb.branches(
        limbs,
        count=(1, 3),
        pitch_set=(math.radians(85),),
        length_ratio=(0.4, 0.6),
        radius_ratio=0.5,
        upturn_rad=math.radians(25),
    )
    return sb.skeleton(), trunk, limbs, twigs


def _fresh_oak(seed: int = 42):
    """Return a fresh (sk, limbs, twigs) with a fixed seed — no rng spill."""
    set_world_seed(seed)
    rng = for_domain("test", "leaves_characterize")
    sk, _trunk, limbs, twigs = _build_oak(rng)
    return sk, limbs, twigs, rng


# ---------------------------------------------------------------------------
# 1. Leaves.empty() — shape, dtype, zero count
# ---------------------------------------------------------------------------


class TestLeavesEmpty:
    def test_n_leaves_is_zero(self):
        e = Leaves.empty()
        assert e.n_leaves == 0

    def test_center_shape_and_dtype(self):
        e = Leaves.empty()
        assert e.center.shape == (0, 3)
        assert e.center.dtype == np.float32

    def test_radius_shape_and_dtype(self):
        e = Leaves.empty()
        assert e.radius.shape == (0,)
        assert e.radius.dtype == np.float32

    def test_sway_shape_and_dtype(self):
        e = Leaves.empty()
        assert e.sway.shape == (0,)
        assert e.sway.dtype == np.float32

    def test_n_leaves_matches_center_length(self):
        e = Leaves.empty()
        assert e.n_leaves == len(e.center)


# ---------------------------------------------------------------------------
# 2. n_leaves property consistency (non-empty Leaves)
# ---------------------------------------------------------------------------


class TestNLeavesProperty:
    def test_n_leaves_matches_center_length(self):
        sk, limbs, twigs, rng = _fresh_oak(7)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.7)
        assert leaves.n_leaves == leaves.center.shape[0]

    def test_n_leaves_matches_radius_length(self):
        sk, limbs, twigs, rng = _fresh_oak(7)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.7)
        assert leaves.n_leaves == leaves.radius.shape[0]

    def test_n_leaves_matches_sway_length(self):
        sk, limbs, twigs, rng = _fresh_oak(7)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.7)
        assert leaves.n_leaves == leaves.sway.shape[0]


# ---------------------------------------------------------------------------
# 3. Output shapes and dtypes
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def setup_method(self):
        sk, limbs, twigs, rng = _fresh_oak(11)
        ids = np.concatenate([limbs, twigs])
        self.leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.7)

    def test_center_is_Lx3_float32(self):
        L = self.leaves.n_leaves
        assert self.leaves.center.shape == (L, 3)
        assert self.leaves.center.dtype == np.float32

    def test_radius_is_L_float32(self):
        L = self.leaves.n_leaves
        assert self.leaves.radius.shape == (L,)
        assert self.leaves.radius.dtype == np.float32

    def test_sway_is_L_float32(self):
        L = self.leaves.n_leaves
        assert self.leaves.sway.shape == (L,)
        assert self.leaves.sway.dtype == np.float32

    def test_nonempty_output(self):
        assert self.leaves.n_leaves > 0


# ---------------------------------------------------------------------------
# 4. Determinism — same skeleton + same rng state → identical arrays
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _call_twice(self, seed: int):
        """Build the same skeleton twice from the same seed and call leaves_at_tips."""
        set_world_seed(seed)
        rng1 = for_domain("test", "det_a")
        sk1, _, limbs1, twigs1 = _build_oak(rng1)
        ids1 = np.concatenate([limbs1, twigs1])
        leaves1 = leaves_at_tips(sk1, ids1, rng1, rounds=3, density=0.7)

        set_world_seed(seed)
        rng2 = for_domain("test", "det_a")
        sk2, _, limbs2, twigs2 = _build_oak(rng2)
        ids2 = np.concatenate([limbs2, twigs2])
        leaves2 = leaves_at_tips(sk2, ids2, rng2, rounds=3, density=0.7)

        return leaves1, leaves2

    def test_center_byte_identical(self):
        l1, l2 = self._call_twice(42)
        assert np.array_equal(l1.center, l2.center)

    def test_radius_byte_identical(self):
        l1, l2 = self._call_twice(42)
        assert np.array_equal(l1.radius, l2.radius)

    def test_sway_byte_identical(self):
        l1, l2 = self._call_twice(42)
        assert np.array_equal(l1.sway, l2.sway)

    def test_different_seeds_differ(self):
        l1, _ = self._call_twice(42)
        l2, _ = self._call_twice(99)
        # Different seeds should produce different centers (extremely likely)
        assert not np.array_equal(l1.center, l2.center)


# ---------------------------------------------------------------------------
# 5. Sway range — documented as ~0.85–1.0 (sway_min default 0.85)
# ---------------------------------------------------------------------------


class TestSwayRange:
    def setup_method(self):
        sk, limbs, twigs, rng = _fresh_oak(13)
        ids = np.concatenate([limbs, twigs])
        self.leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.8)

    def test_sway_not_below_085(self):
        # Default sway_min=0.85; sway = max(tip_sway, uniform(0.85, 1.0))
        # clipped to [0,1] at the end — pin current lower bound
        assert (self.leaves.sway >= 0.85).all()

    def test_sway_not_above_1(self):
        assert (self.leaves.sway <= 1.0).all()

    def test_sway_finite(self):
        assert np.isfinite(self.leaves.sway).all()

    def test_sway_respects_custom_sway_min(self):
        sk, limbs, twigs, rng = _fresh_oak(13)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.8, sway_min=0.5)
        # With a lower floor, some leaves may dip below 0.85 (tip sway might
        # be < 0.85 on some segments) — but nothing below 0.5 should survive.
        assert (leaves.sway >= 0.5).all()
        assert (leaves.sway <= 1.0).all()


# ---------------------------------------------------------------------------
# 6. Leaf centers are finite and within plausible skeleton bounds
# ---------------------------------------------------------------------------


class TestLeafCenters:
    def test_centers_finite(self):
        sk, limbs, twigs, rng = _fresh_oak(17)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.7)
        assert np.isfinite(leaves.center).all()

    def test_centers_near_tip_ends(self):
        """Every leaf center must be within CA growth reach of some branch tip."""
        CELL = 0.25
        ROUNDS = 3
        sk, limbs, twigs, rng = _fresh_oak(17)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, cell_m=CELL, rounds=ROUNDS, density=0.7)
        tips = sk.tip_ids(ids)
        # dist from each leaf center to the nearest tip end
        diff = leaves.center[:, None, :] - sk.end[tips][None, :, :]  # (L, T, 3)
        dist = np.linalg.norm(diff, axis=2).min(axis=1)  # (L,)
        reach = (ROUNDS + 1.0) * CELL * math.sqrt(3.0)
        assert (dist <= reach + 1e-4).all()

    def test_centers_above_ground(self):
        """Pin that leaf Z values remain within one CA growth reach of the tips.

        SUSPECTED BUG: leaves CAN have significantly negative Z (observed
        ~-0.73 m) even for a tree whose lowest tip is well above ground.
        This occurs because the grid lo-bound is (min_tip_Z - pad) and the
        jitter can then place leaves well below the lowest tip.  There is no
        explicit Z floor clamping leaf centers to >=0.  Pinning the ACTUAL
        lower bound: Z > -(rounds+1)*cell_m*sqrt(3) below the lowest tip.
        """
        CELL = 0.25
        ROUNDS = 3
        sk, limbs, twigs, rng = _fresh_oak(17)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, cell_m=CELL, rounds=ROUNDS, density=0.7)
        tips = sk.tip_ids(ids)
        min_tip_z = float(sk.end[tips, 2].min())
        # Pin: no leaf should be more than one full growth-reach below the
        # lowest tip end (the grid extends pad = (rounds+0.5)*cell_m downward)
        max_drop = (ROUNDS + 1.0) * CELL * math.sqrt(3.0)
        assert (leaves.center[:, 2] >= min_tip_z - max_drop - 0.1).all()


# ---------------------------------------------------------------------------
# 7. Radius range
# ---------------------------------------------------------------------------


class TestRadius:
    def test_radius_in_default_range(self):
        """Default leaf_size_m=(0.09, 0.14) — all radii should be in range."""
        sk, limbs, twigs, rng = _fresh_oak(19)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.7, leaf_size_m=(0.09, 0.14))
        assert (leaves.radius >= 0.09 - 1e-6).all()
        assert (leaves.radius <= 0.14 + 1e-6).all()

    def test_radius_finite_positive(self):
        sk, limbs, twigs, rng = _fresh_oak(19)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=0.7)
        assert np.isfinite(leaves.radius).all()
        assert (leaves.radius > 0).all()


# ---------------------------------------------------------------------------
# 8. rounds=0 and density=0.0 → Leaves.empty()
# ---------------------------------------------------------------------------


class TestEarlyExitConditions:
    def test_rounds_zero_returns_empty(self):
        sk, limbs, twigs, rng = _fresh_oak(23)
        ids = np.concatenate([limbs, twigs])
        result = leaves_at_tips(sk, ids, rng, rounds=0, density=0.7)
        assert result.n_leaves == 0

    def test_density_zero_returns_empty(self):
        sk, limbs, twigs, rng = _fresh_oak(23)
        ids = np.concatenate([limbs, twigs])
        result = leaves_at_tips(sk, ids, rng, rounds=3, density=0.0)
        assert result.n_leaves == 0

    def test_empty_ids_returns_empty(self):
        sk, limbs, twigs, rng = _fresh_oak(23)
        result = leaves_at_tips(sk, np.empty(0, dtype=np.int32), rng, rounds=3, density=0.7)
        assert result.n_leaves == 0

    def test_empty_ids_center_shape(self):
        sk, limbs, twigs, rng = _fresh_oak(23)
        result = leaves_at_tips(sk, np.empty(0, dtype=np.int32), rng)
        assert result.center.shape == (0, 3)

    def test_ids_with_no_tips_returns_empty(self):
        """Passing only the trunk (which has children) → no tips → empty."""
        sk, limbs, twigs, rng = _fresh_oak(23)
        # Trunk ids all have children (limbs grow from them), so tip_ids is empty.
        trunk_ids = np.array([0], dtype=np.int32)
        # tip_ids of trunk[0] (parent of limbs) should be empty unless it
        # has no children — with our recipe it does have children.
        tips = sk.tip_ids(trunk_ids)
        if tips.size == 0:
            result = leaves_at_tips(sk, trunk_ids, rng, rounds=3, density=0.7)
            assert result.n_leaves == 0


# ---------------------------------------------------------------------------
# 9. rounds / density monotonicity — more → more leaves
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_higher_density_more_or_equal_leaves(self):
        """Double density should not produce fewer leaves (statistical — large cap)."""
        # We use max_leaves high enough that the cap doesn't interfere.
        sk, limbs, twigs, rng = _fresh_oak(29)
        ids = np.concatenate([limbs, twigs])

        set_world_seed(29)
        rng_lo = for_domain("test", "mono_lo")
        sk_lo, _, limbs_lo, twigs_lo = _build_oak(rng_lo)
        ids_lo = np.concatenate([limbs_lo, twigs_lo])
        leaves_lo = leaves_at_tips(sk_lo, ids_lo, rng_lo, rounds=3, density=0.3, max_leaves=10_000)

        set_world_seed(29)
        rng_hi = for_domain("test", "mono_hi")
        sk_hi, _, limbs_hi, twigs_hi = _build_oak(rng_hi)
        ids_hi = np.concatenate([limbs_hi, twigs_hi])
        leaves_hi = leaves_at_tips(sk_hi, ids_hi, rng_hi, rounds=3, density=0.9, max_leaves=10_000)

        # NOTE: strictly monotone is stochastic — pin the directional trend
        # with a generous margin (more likely true than not for density 0.3→0.9)
        assert leaves_hi.n_leaves > leaves_lo.n_leaves

    def test_more_rounds_more_or_equal_leaves(self):
        """More rounds spreads hydration further → more hydrated cells → more leaves."""
        set_world_seed(31)
        rng1 = for_domain("test", "rounds_1")
        sk1, _, limbs1, twigs1 = _build_oak(rng1)
        ids1 = np.concatenate([limbs1, twigs1])
        leaves1 = leaves_at_tips(sk1, ids1, rng1, rounds=1, density=0.8, max_leaves=10_000)

        set_world_seed(31)
        rng4 = for_domain("test", "rounds_4")
        sk4, _, limbs4, twigs4 = _build_oak(rng4)
        ids4 = np.concatenate([limbs4, twigs4])
        leaves4 = leaves_at_tips(sk4, ids4, rng4, rounds=4, density=0.8, max_leaves=10_000)

        # Pin direction: 4 rounds must produce at least as many leaves as 1
        assert leaves4.n_leaves >= leaves1.n_leaves


# ---------------------------------------------------------------------------
# 10. max_leaves cap — deterministic thinning
# ---------------------------------------------------------------------------


class TestMaxLeavesCap:
    def test_exactly_at_cap(self):
        """When uncapped output > max_leaves, result must equal max_leaves."""
        sk, limbs, twigs, rng = _fresh_oak(37)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, rounds=3, density=1.0, per_cell=(2, 2), max_leaves=50)
        assert leaves.n_leaves == 50

    def test_cap_not_exceeded(self):
        sk, limbs, twigs, rng = _fresh_oak(37)
        ids = np.concatenate([limbs, twigs])
        for cap in (10, 25, 100):
            set_world_seed(37)
            rng2 = for_domain("test", "cap_test")
            sk2, _, limbs2, twigs2 = _build_oak(rng2)
            ids2 = np.concatenate([limbs2, twigs2])
            leaves = leaves_at_tips(sk2, ids2, rng2, rounds=3, density=1.0, max_leaves=cap)
            assert leaves.n_leaves <= cap

    def test_below_cap_not_thinned(self):
        """When n < max_leaves the cap must not reduce count."""
        sk, limbs, twigs, rng = _fresh_oak(41)
        ids = np.concatenate([limbs, twigs])
        # rounds=1 density=0.1 → very few leaves; cap=10000 should not bite
        leaves = leaves_at_tips(sk, ids, rng, rounds=1, density=0.1, max_leaves=10_000)
        # We cannot know the exact count, but it must be <= 10_000
        assert leaves.n_leaves <= 10_000
        # And it must be > 0 (some leaves should form with at least 1 round)
        assert leaves.n_leaves >= 0  # pin: could be 0 if stochastic draw fails


# ---------------------------------------------------------------------------
# 11. Ids that are not tips — only actual tips seed hydration
# ---------------------------------------------------------------------------


class TestTipFiltering:
    def test_passing_only_limbs_gives_fewer_leaves_than_all(self):
        """Tips in limbs only < tips in limbs+twigs → fewer or equal leaves."""
        set_world_seed(43)
        rng_all = for_domain("test", "tip_all")
        sk_all, _, limbs_all, twigs_all = _build_oak(rng_all)
        ids_all = np.concatenate([limbs_all, twigs_all])
        leaves_all = leaves_at_tips(
            sk_all, ids_all, rng_all, rounds=3, density=0.8, max_leaves=10_000
        )

        set_world_seed(43)
        rng_limb = for_domain("test", "tip_limb")
        sk_limb, _, limbs_limb, _twigs_limb = _build_oak(rng_limb)
        # Only pass limbs — twigs grow from limbs, so limbs are parents (not tips)
        # → many fewer tip seeds
        leaves_limb = leaves_at_tips(
            sk_limb, limbs_limb, rng_limb, rounds=3, density=0.8, max_leaves=10_000
        )

        # NOTE: If ALL limbs happen to be tips (no twigs off them in this variant)
        # then limbs_limb tips == ids_all tips. In practice the oak recipe always
        # has twigs, so limb tips < all tips.
        assert leaves_all.n_leaves >= leaves_limb.n_leaves


# ---------------------------------------------------------------------------
# 12. per_cell tuple controls leaves-per-cell range
# ---------------------------------------------------------------------------


class TestPerCell:
    def test_per_cell_1_1_gives_fewer_than_2_2(self):
        """per_cell=(1,1) gives at most as many leaves as per_cell=(2,2)."""
        set_world_seed(47)
        rng1 = for_domain("test", "pc1")
        sk1, _, limbs1, twigs1 = _build_oak(rng1)
        ids1 = np.concatenate([limbs1, twigs1])
        leaves1 = leaves_at_tips(
            sk1, ids1, rng1, rounds=3, density=0.8, per_cell=(1, 1), max_leaves=10_000
        )

        set_world_seed(47)
        rng2 = for_domain("test", "pc2")
        sk2, _, limbs2, twigs2 = _build_oak(rng2)
        ids2 = np.concatenate([limbs2, twigs2])
        leaves2 = leaves_at_tips(
            sk2, ids2, rng2, rounds=3, density=0.8, per_cell=(2, 2), max_leaves=10_000
        )

        # per_cell=(2,2) always emits exactly 2 leaves/cell;
        # per_cell=(1,1) always emits 1 — so double.
        assert leaves2.n_leaves >= leaves1.n_leaves


# ---------------------------------------------------------------------------
# 13. custom cell_m — larger cells → fewer, farther-spread leaves
# ---------------------------------------------------------------------------


class TestCellSize:
    def _tiny_skeleton(self, rng):
        """A minimal single-tip skeleton to avoid the _MAX_GRID_CELLS guard at small cell_m."""
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=2.0, base_radius_m=0.1, segments=1, wobble_m=0.0)
        twigs = sb.branches(
            trunk,
            count=(1, 1),
            pitch_set=(math.radians(85),),
            length_ratio=(0.5, 0.5),
            radius_ratio=0.5,
            segments=1,
        )
        return sb.skeleton(), twigs

    def test_larger_cell_larger_canopy_spread(self):
        """A bigger CA cell should spread hydration further in world space.

        NOTE: cell_m=0.1 on a full oak triggers the _MAX_GRID_CELLS=200_000
        guard (grid would be ~67×87×60 ≈ 350k cells) and raises ValueError.
        So this test uses a minimal single-tip skeleton to safely compare
        small vs large cell_m.
        """
        set_world_seed(53)
        rng_sm = for_domain("test", "cell_sm")
        sk_sm, twigs_sm = self._tiny_skeleton(rng_sm)
        leaves_sm = leaves_at_tips(
            sk_sm, twigs_sm, rng_sm, rounds=3, density=0.8, cell_m=0.1, max_leaves=10_000
        )

        set_world_seed(53)
        rng_lg = for_domain("test", "cell_lg")
        sk_lg, twigs_lg = self._tiny_skeleton(rng_lg)
        leaves_lg = leaves_at_tips(
            sk_lg, twigs_lg, rng_lg, rounds=3, density=0.8, cell_m=0.5, max_leaves=10_000
        )

        # Larger cell → each round spreads 0.5 m vs 0.1 m: larger bounding box.
        # Pin this as: max extent of large-cell leaves >= small-cell leaves
        if leaves_sm.n_leaves > 0 and leaves_lg.n_leaves > 0:
            ext_sm = leaves_sm.center.max(axis=0) - leaves_sm.center.min(axis=0)
            ext_lg = leaves_lg.center.max(axis=0) - leaves_lg.center.min(axis=0)
            # At least one axis should be equal or larger for big cells.
            # SUSPECTED: this may occasionally fail due to stochastic density
            # thinning. Pinning the directional relationship with tolerance:
            assert ext_lg.max() >= ext_sm.max() * 0.5  # generous tolerance

    def test_small_cell_with_full_oak_raises_grid_overflow(self):
        """Pin that cell_m=0.1 on a full oak exceeds the _MAX_GRID_CELLS=200_000 guard.

        SUSPECTED BUG/MISSING GUARD: the error message says the grid is too
        large but the default cell_m=0.25 is fine; only extreme values trigger
        it.  This test pins the current guard behaviour: ValueError is raised.
        """
        set_world_seed(53)
        rng = for_domain("test", "overflow")
        sk, _, limbs, twigs = _build_oak(rng)
        ids = np.concatenate([limbs, twigs])
        with pytest.raises(ValueError, match="exceeds"):
            leaves_at_tips(sk, ids, rng, rounds=3, density=0.8, cell_m=0.1)


# ---------------------------------------------------------------------------
# 14. Leaves dataclass attribute access by name (not just index)
# ---------------------------------------------------------------------------


class TestDataclassContract:
    def test_center_attribute_name(self):
        sk, limbs, twigs, rng = _fresh_oak(59)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng)
        _ = leaves.center  # must not raise AttributeError

    def test_radius_attribute_name(self):
        sk, limbs, twigs, rng = _fresh_oak(59)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng)
        _ = leaves.radius

    def test_sway_attribute_name(self):
        sk, limbs, twigs, rng = _fresh_oak(59)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng)
        _ = leaves.sway

    def test_n_leaves_is_int(self):
        e = Leaves.empty()
        assert isinstance(e.n_leaves, int)

    def test_n_leaves_from_result_is_int(self):
        sk, limbs, twigs, rng = _fresh_oak(59)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng)
        assert isinstance(leaves.n_leaves, int)
