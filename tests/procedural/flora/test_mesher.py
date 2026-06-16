"""
tests/test_flora_mesher_determinism.py — characterization / golden-master
tests for procedural/flora/mesher.py.

Pins current behaviour of mesh_branches, mesh_leaves, merge_parts, and
mesh_leaf_area_m2 without fixing any bugs.  Suspicions are noted in
comments where current behaviour looks surprising.

Headless, fixed-seed, numpy assertions only — no per-element Python loops.
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora import (
    Leaves,
    SkeletonBuilder,
    leaves_at_tips,
    merge_parts,
    mesh_branches,
    mesh_leaves,
)
from fire_engine.procedural.flora.mesher import TreeMesh, mesh_leaf_area_m2

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _fixed_rng(seed: int = 77) -> np.random.Generator:
    set_world_seed(seed)
    return for_domain("test", "mesher_det")


def _build_sk_and_leaves(seed: int = 77):
    """Build a reproducible skeleton + leaves pair."""
    set_world_seed(seed)
    rng = for_domain("test", "mesher_det")
    sb = SkeletonBuilder(rng)
    trunk = sb.trunk(height_m=5.0, base_radius_m=0.25, segments=3, wobble_m=0.2)
    limbs = sb.branches(
        trunk, count=(3, 4), pitch_set=(math.radians(85),), length_ratio=(0.4, 0.6), segments=2
    )
    sk = sb.skeleton()
    leaves = leaves_at_tips(sk, limbs, rng, cell_m=0.25, rounds=3, density=0.8)
    return sk, leaves, rng


# ---------------------------------------------------------------------------
# 1. TreeMesh dataclass: dtypes, shapes, and empty constructor
# ---------------------------------------------------------------------------


class TestTreeMeshDataclass:
    def setup_method(self):
        sk, leaves, rng = _build_sk_and_leaves()
        self.wood = mesh_branches(sk)
        self.foliage = mesh_leaves(leaves, rng)
        self.merged = merge_parts(self.wood, self.foliage)

    def test_positions_float32_shape_v3(self):
        for m in (self.wood, self.foliage, self.merged):
            assert m.positions.dtype == np.float32
            assert m.positions.ndim == 2 and m.positions.shape[1] == 3

    def test_normals_float32_shape_v3(self):
        for m in (self.wood, self.foliage, self.merged):
            assert m.normals.dtype == np.float32
            assert m.normals.ndim == 2 and m.normals.shape[1] == 3

    def test_uvs_float32_shape_v2(self):
        for m in (self.wood, self.foliage, self.merged):
            assert m.uvs.dtype == np.float32
            assert m.uvs.ndim == 2 and m.uvs.shape[1] == 2

    def test_colors_float32_shape_v4(self):
        for m in (self.wood, self.foliage, self.merged):
            assert m.colors.dtype == np.float32
            assert m.colors.ndim == 2 and m.colors.shape[1] == 4

    def test_indices_uint32_1d(self):
        for m in (self.wood, self.foliage, self.merged):
            assert m.indices.dtype == np.uint32
            assert m.indices.ndim == 1

    def test_all_arrays_same_vertex_count_as_n_vertices(self):
        for m in (self.wood, self.foliage, self.merged):
            n = m.n_vertices
            assert m.normals.shape[0] == n
            assert m.uvs.shape[0] == n
            assert m.colors.shape[0] == n

    def test_empty_static_constructor(self):
        e = TreeMesh.empty()
        assert e.n_vertices == 0
        assert e.positions.shape == (0, 3)
        assert e.indices.shape == (0,)
        assert e.height_m == 0.0 and e.radius_m == 0.0


# ---------------------------------------------------------------------------
# 2. mesh_branches: vertex/face counts, position dtype, normals, UVs, sway
# ---------------------------------------------------------------------------


class TestMeshBranches:
    def setup_method(self):
        sk, _, _ = _build_sk_and_leaves()
        self.sk = sk
        self.wood = mesh_branches(sk)

    def test_indices_triangle_list(self):
        assert self.wood.indices.shape[0] % 3 == 0

    def test_index_bounds(self):
        # All indices must reference valid vertices.
        assert int(self.wood.indices.max()) < self.wood.n_vertices

    def test_positions_are_float32(self):
        assert self.wood.positions.dtype == np.float32

    def test_normals_approximately_unit(self):
        norms = np.linalg.norm(self.wood.normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_uv_within_default_bark_rect(self):
        # Default uv_rect=(0.0, 0.0, 0.5, 1.0): u in [0, 0.5], v in [0, 1].
        uvs = self.wood.uvs
        assert (uvs[:, 0] >= -1e-6).all()
        assert (uvs[:, 0] <= 0.5 + 1e-6).all()
        assert (uvs[:, 1] >= -1e-6).all()
        assert (uvs[:, 1] <= 1.0 + 1e-6).all()

    def test_custom_uv_rect_respected(self):
        sk, _, _ = _build_sk_and_leaves(seed=11)
        rect = (0.1, 0.2, 0.4, 0.9)
        w = mesh_branches(sk, uv_rect=rect)
        u0, v0, u1, v1 = rect
        assert (w.uvs[:, 0] >= u0 - 1e-6).all()
        assert (w.uvs[:, 0] <= u1 + 1e-6).all()
        assert (w.uvs[:, 1] >= v0 - 1e-6).all()
        assert (w.uvs[:, 1] <= v1 + 1e-6).all()

    def test_sway_weight_in_01(self):
        wa = self.wood.colors[:, 3]
        assert (wa >= 0.0).all() and (wa <= 1.0).all()

    def test_sway_near_zero_at_trunk_base(self):
        # Vertices very close to z=0 belong to the trunk base and must have
        # low sway weight (hard-pinned by the wind shader).
        base_mask = self.wood.positions[:, 2] < 0.05
        assert base_mask.any(), "Expected at least one near-ground vertex"
        # SUSPICION: if a trunk has wobble the base ring may sit slightly
        # above z=0 for non-root segments; pin the observed < 0.5 threshold.
        assert self.wood.colors[base_mask, 3].max() < 0.5

    def test_sway_increases_toward_tips(self):
        # The maximum sway in the upper half of the tree must exceed the
        # maximum sway in the lower half — directional pin, not a tight bound.
        z = self.wood.positions[:, 2]
        midpoint = float(z.max()) / 2.0
        lo_sway = self.wood.colors[z < midpoint, 3].max()
        hi_sway = self.wood.colors[z >= midpoint, 3].max()
        assert hi_sway > lo_sway

    def test_vertex_count_scales_with_segments(self):
        # A skeleton with more trunk segments should produce more vertices.
        set_world_seed(55)
        rng = for_domain("test", "seg_count")
        sb_a = SkeletonBuilder(rng)
        sb_a.trunk(height_m=4.0, base_radius_m=0.2, segments=2)
        sk_a = sb_a.skeleton()

        set_world_seed(55)
        rng = for_domain("test", "seg_count")
        sb_b = SkeletonBuilder(rng)
        sb_b.trunk(height_m=4.0, base_radius_m=0.2, segments=6)
        sk_b = sb_b.skeleton()

        assert mesh_branches(sk_b).n_vertices > mesh_branches(sk_a).n_vertices

    def test_tint_baked_into_rgb(self):
        sk, _, _ = _build_sk_and_leaves(seed=22)
        tint = (0.5, 0.7, 0.9)
        w = mesh_branches(sk, tint=tint)
        assert np.allclose(w.colors[:, 0], tint[0])
        assert np.allclose(w.colors[:, 1], tint[1])
        assert np.allclose(w.colors[:, 2], tint[2])

    def test_empty_skeleton_returns_empty_mesh(self):
        # SkeletonBuilder.skeleton() raises if nothing was grown — cover the
        # direct zero-segment path by constructing a bare TreeSkeleton manually.
        # SUSPICION: the docstring says S==0 returns empty, but SkeletonBuilder
        # raises before you can finalize; test the internal guard path directly.
        from fire_engine.procedural.flora.skeleton import TreeSkeleton

        sk_empty = TreeSkeleton(
            parent=np.empty(0, dtype=np.int32),
            start=np.empty((0, 3), dtype=np.float32),
            end=np.empty((0, 3), dtype=np.float32),
            radius_start=np.empty(0, dtype=np.float32),
            radius_end=np.empty(0, dtype=np.float32),
            depth=np.empty(0, dtype=np.int32),
            sway=np.empty(0, dtype=np.float32),
        )
        result = mesh_branches(sk_empty)
        assert result.n_vertices == 0


# ---------------------------------------------------------------------------
# 3. mesh_leaves: determinism (rng-dependent), counts, normals, UVs, sway
# ---------------------------------------------------------------------------


class TestMeshLeavesDeterminism:
    """mesh_leaves consumes rng — identical state must give byte-identical output."""

    def _make_leaves_and_rng(self, seed: int = 77):
        """Return a (Leaves, fresh_rng) pair built from a fixed state."""
        _sk, leaves, _ = _build_sk_and_leaves(seed)
        # The rng was advanced by build; return the leaves as-is.
        # For the mesh_leaves call we need a FRESH, separately-seeded rng
        # so we can replay the exact same state twice.
        set_world_seed(seed + 100)
        mesh_rng = for_domain("test", "mesh_leaves_rng")
        return leaves, mesh_rng

    def test_same_leaves_same_rng_byte_identical_positions(self):
        leaves, _ = self._make_leaves_and_rng(77)

        set_world_seed(200)
        rng1 = for_domain("test", "det_leaves")
        m1 = mesh_leaves(leaves, rng1)

        set_world_seed(200)
        rng2 = for_domain("test", "det_leaves")
        m2 = mesh_leaves(leaves, rng2)

        assert np.array_equal(m1.positions, m2.positions)

    def test_same_leaves_same_rng_byte_identical_normals(self):
        leaves, _ = self._make_leaves_and_rng(77)

        set_world_seed(201)
        rng1 = for_domain("test", "det_normals")
        m1 = mesh_leaves(leaves, rng1)

        set_world_seed(201)
        rng2 = for_domain("test", "det_normals")
        m2 = mesh_leaves(leaves, rng2)

        assert np.array_equal(m1.normals, m2.normals)

    def test_same_leaves_same_rng_byte_identical_uvs(self):
        leaves, _ = self._make_leaves_and_rng(77)

        set_world_seed(202)
        rng1 = for_domain("test", "det_uvs")
        m1 = mesh_leaves(leaves, rng1)

        set_world_seed(202)
        rng2 = for_domain("test", "det_uvs")
        m2 = mesh_leaves(leaves, rng2)

        assert np.array_equal(m1.uvs, m2.uvs)

    def test_same_leaves_same_rng_byte_identical_colors(self):
        leaves, _ = self._make_leaves_and_rng(77)

        set_world_seed(203)
        rng1 = for_domain("test", "det_colors")
        m1 = mesh_leaves(leaves, rng1)

        set_world_seed(203)
        rng2 = for_domain("test", "det_colors")
        m2 = mesh_leaves(leaves, rng2)

        assert np.array_equal(m1.colors, m2.colors)

    def test_same_leaves_same_rng_byte_identical_indices(self):
        leaves, _ = self._make_leaves_and_rng(77)

        set_world_seed(204)
        rng1 = for_domain("test", "det_idx1")
        m1 = mesh_leaves(leaves, rng1)

        set_world_seed(204)
        rng2 = for_domain("test", "det_idx2")
        m2 = mesh_leaves(leaves, rng2)

        assert np.array_equal(m1.indices, m2.indices)

    def test_one_quad_per_leaf(self):
        leaves, _ = self._make_leaves_and_rng(77)
        set_world_seed(205)
        rng = for_domain("test", "qpl")
        m = mesh_leaves(leaves, rng)
        assert m.n_vertices == leaves.n_leaves * 4
        assert m.indices.shape[0] == leaves.n_leaves * 6

    def test_leaf_normals_unit(self):
        leaves, _ = self._make_leaves_and_rng(77)
        set_world_seed(206)
        rng = for_domain("test", "ln")
        m = mesh_leaves(leaves, rng)
        norms = np.linalg.norm(m.normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_leaf_normals_positive_z(self):
        # Default tilt_range_rad=(0.26, 1.22) — max tilt is ~70° from +Z.
        # The Z component of a unit vector at 70° off +Z is cos(70°)≈0.34 > 0.
        leaves, _ = self._make_leaves_and_rng(77)
        set_world_seed(207)
        rng = for_domain("test", "lnz")
        m = mesh_leaves(leaves, rng)
        assert (m.normals[:, 2] > 0.0).all()

    def test_leaf_uv_in_right_half(self):
        # Default uv_rect=(0.5, 0.0, 1.0, 1.0): u in [0.5, 1.0].
        leaves, _ = self._make_leaves_and_rng(77)
        set_world_seed(208)
        rng = for_domain("test", "luv")
        m = mesh_leaves(leaves, rng)
        assert (m.uvs[:, 0] >= 0.5 - 1e-6).all()
        assert (m.uvs[:, 0] <= 1.0 + 1e-6).all()

    def test_leaf_sway_high(self):
        # The CA sets sway_min=0.85; all leaf sway weights must be >= 0.85.
        leaves, _ = self._make_leaves_and_rng(77)
        set_world_seed(209)
        rng = for_domain("test", "lsway")
        m = mesh_leaves(leaves, rng)
        assert (m.colors[:, 3] >= 0.85).all()
        assert (m.colors[:, 3] <= 1.0).all()

    def test_empty_leaves_returns_empty_mesh(self):
        set_world_seed(99)
        rng = for_domain("test", "empty_l")
        m = mesh_leaves(Leaves.empty(), rng)
        assert m.n_vertices == 0
        assert m.indices.shape[0] == 0
        assert m.height_m == 0.0 and m.radius_m == 0.0


# ---------------------------------------------------------------------------
# 4. merge_parts: concatenation, index offset, no out-of-range
# ---------------------------------------------------------------------------


class TestMergeParts:
    def setup_method(self):
        sk, leaves, rng = _build_sk_and_leaves()
        self.wood = mesh_branches(sk)
        self.foliage = mesh_leaves(leaves, rng)
        self.merged = merge_parts(self.wood, self.foliage)

    def test_vertex_count_is_sum(self):
        assert self.merged.n_vertices == self.wood.n_vertices + self.foliage.n_vertices

    def test_index_count_is_sum(self):
        assert (
            self.merged.indices.shape[0]
            == self.wood.indices.shape[0] + self.foliage.indices.shape[0]
        )

    def test_no_out_of_range_index(self):
        assert int(self.merged.indices.max()) < self.merged.n_vertices

    def test_foliage_indices_offset_by_wood_vertex_count(self):
        wood_idx_count = self.wood.indices.shape[0]
        foliage_block = self.merged.indices[wood_idx_count:]
        # Every foliage index must point into the foliage vertex block.
        assert int(foliage_block.min()) >= self.wood.n_vertices

    def test_merged_triangle_list(self):
        assert self.merged.indices.shape[0] % 3 == 0

    def test_merge_with_empty_foliage(self):
        sk, _, _ = _build_sk_and_leaves(seed=50)
        set_world_seed(50)
        rng = for_domain("test", "empty_merge")
        wood = mesh_branches(sk)
        empty_foliage = mesh_leaves(Leaves.empty(), rng)
        merged = merge_parts(wood, empty_foliage)
        assert merged.n_vertices == wood.n_vertices
        assert np.array_equal(merged.positions, wood.positions)

    def test_merge_all_empty_returns_empty(self):
        result = merge_parts(TreeMesh.empty(), TreeMesh.empty())
        assert result.n_vertices == 0

    def test_merged_positions_concatenated_in_order(self):
        # First wood.n_vertices rows must match wood.positions.
        nw = self.wood.n_vertices
        assert np.array_equal(self.merged.positions[:nw], self.wood.positions)
        assert np.array_equal(self.merged.positions[nw:], self.foliage.positions)


# ---------------------------------------------------------------------------
# 5. mesh_leaf_area_m2
# ---------------------------------------------------------------------------


class TestMeshLeafAreaM2:
    def test_empty_mesh_returns_zero(self):
        assert mesh_leaf_area_m2(TreeMesh.empty()) == 0.0

    def test_positive_for_non_empty_leaves(self):
        sk, leaves, rng = _build_sk_and_leaves(seed=33)
        wood = mesh_branches(sk)
        foliage = mesh_leaves(leaves, rng)
        merged = merge_parts(wood, foliage)
        area = mesh_leaf_area_m2(merged)
        assert area > 0.0

    def test_bark_only_mesh_returns_zero(self):
        # A mesh with only bark UVs (u < 0.5) should have zero leaf area.
        sk, _, _ = _build_sk_and_leaves(seed=44)
        wood = mesh_branches(sk)
        # Wood UVs are all in [0, 0.5]; no leaf triangles.
        assert mesh_leaf_area_m2(wood) == 0.0

    def test_scales_monotonically_with_leaf_count(self):
        # More leaves → larger total area (pin the monotonic direction).
        sk, _, _ = _build_sk_and_leaves(seed=55)
        set_world_seed(55)
        rng_a = for_domain("test", "area_a")
        small_leaves = leaves_at_tips(
            sk, sk.tip_ids(), rng_a, cell_m=0.25, rounds=1, density=0.5, max_leaves=20
        )
        foliage_a = mesh_leaves(small_leaves, rng_a)
        area_a = mesh_leaf_area_m2(merge_parts(mesh_branches(sk), foliage_a))

        set_world_seed(55)
        rng_b = for_domain("test", "area_b")
        big_leaves = leaves_at_tips(
            sk, sk.tip_ids(), rng_b, cell_m=0.25, rounds=3, density=0.9, max_leaves=400
        )
        foliage_b = mesh_leaves(big_leaves, rng_b)
        area_b = mesh_leaf_area_m2(merge_parts(mesh_branches(sk), foliage_b))

        assert area_b > area_a


# ---------------------------------------------------------------------------
# 6. Full-path determinism: skeleton + leaves + mesh, twice
# ---------------------------------------------------------------------------


class TestFullPathDeterminism:
    def _full(self, world_seed: int):
        set_world_seed(world_seed)
        rng = for_domain("test", "full_path")
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=5.0, base_radius_m=0.25, segments=3, wobble_m=0.2)
        limbs = sb.branches(
            trunk, count=(3, 4), pitch_set=(math.radians(85),), length_ratio=(0.4, 0.6), segments=2
        )
        sk = sb.skeleton()
        leaves = leaves_at_tips(sk, limbs, rng, cell_m=0.25, rounds=3, density=0.8)
        wood = mesh_branches(sk)
        foliage = mesh_leaves(leaves, rng)
        return merge_parts(wood, foliage)

    def test_identical_seed_gives_identical_positions(self):
        m1 = self._full(88)
        m2 = self._full(88)
        assert np.array_equal(m1.positions, m2.positions)

    def test_identical_seed_gives_identical_indices(self):
        m1 = self._full(88)
        m2 = self._full(88)
        assert np.array_equal(m1.indices, m2.indices)

    def test_identical_seed_gives_identical_colors(self):
        m1 = self._full(88)
        m2 = self._full(88)
        assert np.array_equal(m1.colors, m2.colors)

    def test_different_seeds_differ(self):
        m1 = self._full(88)
        m2 = self._full(89)
        # Different world seeds must produce different meshes.
        assert not np.array_equal(m1.positions, m2.positions)
