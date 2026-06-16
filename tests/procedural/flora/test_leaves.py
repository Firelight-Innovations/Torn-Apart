"""
tests/procedural/flora/test_leaves.py — tests for procedural/flora/leaves.py.

Covers the **along-wood** leaf placement: every leaf is anchored to a real
branch segment (anti-floating invariant), the count scales with branch
structure, determinism, empty cases and the thinning cap.  Headless only;
fixed seed; numpy assertions; no per-element Python loops.
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora import SkeletonBuilder, leaves_at_tips
from fire_engine.procedural.flora.leaves import Leaves

# ---------------------------------------------------------------------------
# Shared skeleton factories.
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
    """Return a fresh (sk, limbs, twigs, rng) with a fixed seed — no rng spill."""
    set_world_seed(seed)
    rng = for_domain("test", "leaves")
    sk, _trunk, limbs, twigs = _build_oak(rng)
    return sk, limbs, twigs, rng


def _nearest_segment_distance(centers: np.ndarray, sk, seg_ids: np.ndarray) -> np.ndarray:
    """Distance from each leaf center to the NEAREST point on any leaf-bearing segment.

    Vectorized point-to-segment distance: (L, S) closest-point clamp, then min
    over segments.  No per-leaf Python loop.
    """
    seg = np.unique(np.asarray(seg_ids, dtype=np.int64).ravel())
    a = sk.start[seg].astype(np.float64)  # (S, 3)
    b = sk.end[seg].astype(np.float64)
    ab = b - a  # (S, 3)
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)  # (S,)
    p = centers.astype(np.float64)[:, None, :]  # (L, 1, 3)
    ap = p - a[None, :, :]  # (L, S, 3)
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / denom[None, :], 0.0, 1.0)  # (L, S)
    closest = a[None, :, :] + ab[None, :, :] * t[:, :, None]  # (L, S, 3)
    d = np.linalg.norm(p - closest, axis=2)  # (L, S)
    return d.min(axis=1)  # (L,)


# ---------------------------------------------------------------------------
# 1. Leaves.empty() — shape, dtype, zero count
# ---------------------------------------------------------------------------


class TestLeavesEmpty:
    def test_n_leaves_is_zero(self):
        assert Leaves.empty().n_leaves == 0

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


# ---------------------------------------------------------------------------
# 2. n_leaves property consistency / output shapes & dtypes
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def setup_method(self):
        sk, limbs, twigs, rng = _fresh_oak(11)
        ids = np.concatenate([limbs, twigs])
        self.leaves = leaves_at_tips(sk, ids, rng, density=0.7)

    def test_nonempty_output(self):
        assert self.leaves.n_leaves > 0

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

    def test_n_leaves_matches_all_arrays(self):
        L = self.leaves.n_leaves
        assert self.leaves.center.shape[0] == L
        assert self.leaves.radius.shape[0] == L
        assert self.leaves.sway.shape[0] == L


# ---------------------------------------------------------------------------
# 3. Determinism — same skeleton + same rng state → identical arrays
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _call_twice(self, seed: int):
        set_world_seed(seed)
        rng1 = for_domain("test", "det")
        sk1, _, limbs1, twigs1 = _build_oak(rng1)
        leaves1 = leaves_at_tips(sk1, np.concatenate([limbs1, twigs1]), rng1, density=0.7)

        set_world_seed(seed)
        rng2 = for_domain("test", "det")
        sk2, _, limbs2, twigs2 = _build_oak(rng2)
        leaves2 = leaves_at_tips(sk2, np.concatenate([limbs2, twigs2]), rng2, density=0.7)
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
        assert not np.array_equal(l1.center, l2.center)


# ---------------------------------------------------------------------------
# 4. Anti-floating invariant — EVERY leaf hugs the wood it grows on.
#    This is the core guarantee of the rework.
# ---------------------------------------------------------------------------


class TestAttachmentInvariant:
    def test_every_leaf_near_a_segment_small_skeleton(self):
        """On a known small skeleton every leaf center lies within the
        allowed offset of the nearest point on some leaf-bearing segment."""
        set_world_seed(5)
        rng = for_domain("test", "attach_small")
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=2.0, base_radius_m=0.12, segments=2, wobble_m=0.0)
        twigs = sb.branches(
            trunk,
            count=(2, 3),
            pitch_set=(math.radians(80),),
            length_ratio=(0.4, 0.6),
            radius_ratio=0.5,
            segments=1,
        )
        sk = sb.skeleton()
        leaf_size = (0.09, 0.14)
        leaves = leaves_at_tips(sk, twigs, rng, density=0.8, leaf_size_m=leaf_size, max_leaves=2000)
        assert leaves.n_leaves > 0
        dist = _nearest_segment_distance(leaves.center, sk, twigs)
        # max_offset default = segment_radius(t) + 1.5*max(leaf_size).
        # Bound: thickest twig radius + 1.5*max(leaf) + tiny epsilon.
        seg = np.unique(twigs.astype(np.int64))
        max_seg_r = float(max(sk.radius_start[seg].max(), sk.radius_end[seg].max()))
        bound = max_seg_r + 1.5 * max(leaf_size) + 1e-4
        assert (dist <= bound).all(), f"max leaf dist {dist.max():.4f} > bound {bound:.4f}"

    def test_full_oak_leaves_hug_wood(self):
        sk, limbs, twigs, rng = _fresh_oak(17)
        ids = np.concatenate([limbs, twigs])
        leaf_size = (0.12, 0.18)
        leaves = leaves_at_tips(sk, ids, rng, density=0.85, leaf_size_m=leaf_size, max_leaves=2000)
        dist = _nearest_segment_distance(leaves.center, sk, ids)
        seg = np.unique(ids.astype(np.int64))
        max_seg_r = float(max(sk.radius_start[seg].max(), sk.radius_end[seg].max()))
        bound = max_seg_r + 1.5 * max(leaf_size) + 1e-4
        assert (dist <= bound).all()
        # And no leaf floats far: the worst-case offset is small (< ~0.4 m),
        # nothing like the old ~0.8 m CA blob reach.
        assert dist.max() < 0.45

    def test_leaf_base_anchors_on_bark(self):
        # max_offset_m is deprecated: leaves now anchor their BASE exactly on
        # the bark.  out_dir is unit, and base = center - out_dir*radius sits
        # within a segment radius of the wood (it is literally on the surface).
        sk, limbs, twigs, rng = _fresh_oak(23)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(
            sk, ids, rng, density=0.8, leaf_size_m=(0.05, 0.07), max_leaves=2000
        )
        n = np.linalg.norm(leaves.out_dir, axis=1)
        assert np.allclose(n, 1.0, atol=1e-4)
        base = leaves.center - leaves.out_dir * leaves.radius[:, None]
        dist = _nearest_segment_distance(base, sk, ids)
        seg = np.unique(ids.astype(np.int64))
        max_seg_r = float(max(sk.radius_start[seg].max(), sk.radius_end[seg].max()))
        assert (dist <= max_seg_r + 1e-4).all()


# ---------------------------------------------------------------------------
# 5. Count scales with branch structure (more twigs → more leaves)
# ---------------------------------------------------------------------------


class TestCountScaling:
    def test_higher_density_more_leaves(self):
        set_world_seed(29)
        rng_lo = for_domain("test", "mono_lo")
        sk_lo, _, limbs_lo, twigs_lo = _build_oak(rng_lo)
        lo = leaves_at_tips(
            sk_lo, np.concatenate([limbs_lo, twigs_lo]), rng_lo, density=0.3, max_leaves=10_000
        )
        set_world_seed(29)
        rng_hi = for_domain("test", "mono_hi")
        sk_hi, _, limbs_hi, twigs_hi = _build_oak(rng_hi)
        hi = leaves_at_tips(
            sk_hi, np.concatenate([limbs_hi, twigs_hi]), rng_hi, density=0.9, max_leaves=10_000
        )
        assert hi.n_leaves > lo.n_leaves

    def test_higher_leaves_per_m_more_leaves(self):
        sk, limbs, twigs, rng1 = _fresh_oak(31)
        ids = np.concatenate([limbs, twigs])
        sparse = leaves_at_tips(sk, ids, rng1, density=0.8, leaves_per_m=20, max_leaves=10_000)
        # fresh rng for a fair count comparison
        _, _, _, rng2 = _fresh_oak(31)
        dense = leaves_at_tips(sk, ids, rng2, density=0.8, leaves_per_m=120, max_leaves=10_000)
        assert dense.n_leaves > sparse.n_leaves

    def test_more_segments_more_leaves(self):
        """A tree with more twigs grows more leaves at fixed density."""
        set_world_seed(33)
        rng_few = for_domain("test", "few")
        sk_few, _, limbs_few, twigs_few = _build_oak(rng_few)
        few = leaves_at_tips(
            sk_few,
            np.concatenate([limbs_few, twigs_few]),
            rng_few,
            density=0.7,
            max_leaves=10_000,
        )
        # Bushy tree: more twigs per limb → longer total branch length.
        set_world_seed(33)
        rng_many = for_domain("test", "many")
        sb = SkeletonBuilder(rng_many)
        trunk = sb.trunk(height_m=5.5, base_radius_m=0.28, segments=4, wobble_m=0.35)
        limbs = sb.branches(trunk, count=(4, 6), length_ratio=(0.5, 0.8), segments=2)
        twigs = sb.branches(limbs, count=(3, 5), length_ratio=(0.4, 0.6))
        sk_many = sb.skeleton()
        many = leaves_at_tips(
            sk_many, np.concatenate([limbs, twigs]), rng_many, density=0.7, max_leaves=10_000
        )
        assert many.n_leaves > few.n_leaves

    def test_default_oak_count_is_substantial(self):
        """Default oak call yields a full canopy (hundreds of leaves)."""
        sk, limbs, twigs, rng = _fresh_oak(37)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, density=0.85, max_leaves=2000)
        assert leaves.n_leaves >= 100


# ---------------------------------------------------------------------------
# 6. Sway range
# ---------------------------------------------------------------------------


class TestSwayRange:
    def setup_method(self):
        sk, limbs, twigs, rng = _fresh_oak(13)
        self.leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng, density=0.8)

    def test_sway_tracks_branch_no_floor(self):
        # Sway now TRACKS the host branch (no 0.85 floor) so leaves stay glued
        # to the wood as it bends: inner-limb leaves sway less than tip leaves.
        assert (self.leaves.sway >= 0.0).all()
        assert self.leaves.sway.min() < 0.85

    def test_sway_not_above_1(self):
        assert (self.leaves.sway <= 1.0).all()

    def test_sway_finite(self):
        assert np.isfinite(self.leaves.sway).all()

    def test_sway_min_is_deprecated_no_floor(self):
        # sway_min is accepted but ignored — a high value does NOT floor sway.
        sk, limbs, twigs, rng = _fresh_oak(13)
        leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng, density=0.8, sway_min=0.95)
        assert leaves.sway.min() < 0.95
        assert (leaves.sway >= 0.0).all() and (leaves.sway <= 1.0).all()


# ---------------------------------------------------------------------------
# 7. Radius range / finiteness
# ---------------------------------------------------------------------------


class TestRadius:
    def test_radius_in_default_range(self):
        sk, limbs, twigs, rng = _fresh_oak(19)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, density=0.7, leaf_size_m=(0.09, 0.14))
        assert (leaves.radius >= 0.09 - 1e-6).all()
        assert (leaves.radius <= 0.14 + 1e-6).all()

    def test_radius_finite_positive(self):
        sk, limbs, twigs, rng = _fresh_oak(19)
        leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng, density=0.7)
        assert np.isfinite(leaves.radius).all()
        assert (leaves.radius > 0).all()

    def test_centers_finite(self):
        sk, limbs, twigs, rng = _fresh_oak(19)
        leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng, density=0.7)
        assert np.isfinite(leaves.center).all()


# ---------------------------------------------------------------------------
# 8. Empty / early-exit cases
# ---------------------------------------------------------------------------


class TestEarlyExitConditions:
    def test_rounds_zero_returns_empty(self):
        sk, limbs, twigs, rng = _fresh_oak(23)
        result = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng, rounds=0, density=0.7)
        assert result.n_leaves == 0

    def test_density_zero_returns_empty(self):
        sk, limbs, twigs, rng = _fresh_oak(23)
        result = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng, density=0.0)
        assert result.n_leaves == 0

    def test_empty_ids_returns_empty(self):
        sk, _limbs, _twigs, rng = _fresh_oak(23)
        result = leaves_at_tips(sk, np.empty(0, dtype=np.int32), rng, density=0.7)
        assert result.n_leaves == 0
        assert result.center.shape == (0, 3)

    def test_legacy_ca_kwargs_accepted(self):
        """Old call sites still pass cell_m / per_cell — must not raise."""
        sk, limbs, twigs, rng = _fresh_oak(23)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(
            sk,
            ids,
            rng,
            cell_m=0.26,
            rounds=3,
            density=0.85,
            per_cell=(1, 2),
            leaf_size_m=(0.12, 0.18),
            max_leaves=420,
        )
        assert leaves.n_leaves > 0


# ---------------------------------------------------------------------------
# 9. max_leaves cap — deterministic thinning
# ---------------------------------------------------------------------------


class TestMaxLeavesCap:
    def test_at_or_below_cap(self):
        # Grid thinning runs BEFORE the cap, so an over-subscribed canopy lands
        # at (or just under) the cap, never over it.
        sk, limbs, twigs, rng = _fresh_oak(37)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, density=1.0, leaves_per_m=400, max_leaves=50)
        assert 40 <= leaves.n_leaves <= 50

    def test_cap_not_exceeded(self):
        for cap in (10, 25, 100):
            set_world_seed(37)
            rng2 = for_domain("test", "cap")
            sk2, _, limbs2, twigs2 = _build_oak(rng2)
            leaves = leaves_at_tips(
                sk2, np.concatenate([limbs2, twigs2]), rng2, density=1.0, max_leaves=cap
            )
            assert leaves.n_leaves <= cap

    def test_high_cap_allows_thousands(self):
        """Species may request a few thousand leaves — the ceiling must allow it."""
        sk, limbs, twigs, rng = _fresh_oak(41)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, density=1.0, leaves_per_m=400, max_leaves=3000)
        assert leaves.n_leaves > 600


# ---------------------------------------------------------------------------
# 10. Leaves dataclass attribute access
# ---------------------------------------------------------------------------


class TestDataclassContract:
    def test_attribute_names(self):
        sk, limbs, twigs, rng = _fresh_oak(59)
        ids = np.concatenate([limbs, twigs])
        leaves = leaves_at_tips(sk, ids, rng, density=0.7)
        _ = leaves.center
        _ = leaves.radius
        _ = leaves.sway

    def test_n_leaves_is_int(self):
        assert isinstance(Leaves.empty().n_leaves, int)
        sk, limbs, twigs, rng = _fresh_oak(59)
        leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng, density=0.7)
        assert isinstance(leaves.n_leaves, int)
