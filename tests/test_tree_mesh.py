"""
tests/test_tree_mesh.py — mesher invariants (procedural/flora/mesher.py).

The TreeMesh contract is the engine's interleaved V3N3T2C4 vertex layout
(``world/geometry_bridge``): float32 positions/normals/uvs/colors + uint32
indices.  These tests pin that contract headlessly plus the sway-weight
semantics in ``colors[:, 3]``.
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora import (
    SkeletonBuilder,
    leaves_at_tips,
    merge_parts,
    mesh_branches,
    mesh_leaves,
)


def _build(seed: int = 21):
    set_world_seed(seed)
    rng = for_domain("test", "mesh")
    sb = SkeletonBuilder(rng)
    trunk = sb.trunk(height_m=5.0, base_radius_m=0.25, segments=3, wobble_m=0.2)
    limbs = sb.branches(
        trunk, count=(3, 4), pitch_set=(math.radians(85),), length_ratio=(0.4, 0.6), segments=2
    )
    sk = sb.skeleton()
    leaves = leaves_at_tips(sk, limbs, rng, cell_m=0.25, rounds=3, density=0.8)
    wood = mesh_branches(sk)
    foliage = mesh_leaves(leaves, rng)
    return sk, leaves, wood, foliage, merge_parts(wood, foliage)


class TestContract:
    def setup_method(self):
        (self.sk, self.leaves, self.wood, self.foliage, self.mesh) = _build()

    def test_dtypes_and_shapes(self):
        m = self.mesh
        assert m.positions.dtype == np.float32 and m.positions.shape[1] == 3
        assert m.normals.dtype == np.float32 and m.normals.shape[1] == 3
        assert m.uvs.dtype == np.float32 and m.uvs.shape[1] == 2
        assert m.colors.dtype == np.float32 and m.colors.shape[1] == 4
        assert m.indices.dtype == np.uint32
        n = m.n_vertices
        assert m.normals.shape[0] == n and m.uvs.shape[0] == n and m.colors.shape[0] == n

    def test_index_bounds_and_triangles(self):
        m = self.mesh
        assert m.indices.shape[0] % 3 == 0
        assert int(m.indices.max()) < m.n_vertices

    def test_normals_unit(self):
        norms = np.linalg.norm(self.mesh.normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_metadata_bounds_positions(self):
        m = self.mesh
        assert np.isclose(m.positions[:, 2].max(), m.height_m)
        reach = np.linalg.norm(m.positions[:, 0:2], axis=1).max()
        assert np.isclose(reach, m.radius_m)
        assert m.height_m > 3.0 and m.radius_m > 0.3

    def test_sway_semantics(self):
        # Wood: trunk-base vertices barely sway; leaves: all sway hard.
        wa = self.wood.colors[:, 3]
        assert (wa >= 0.0).all() and (wa <= 1.0).all()
        base_verts = self.wood.positions[:, 2] < 0.1
        assert wa[base_verts].max() < 0.5
        fa = self.foliage.colors[:, 3]
        assert (fa >= 0.85).all() and (fa <= 1.0).all()

    def test_uv_regions(self):
        # Bark verts live in the left atlas half, leaf verts in the right.
        assert (self.wood.uvs[:, 0] <= 0.5 + 1e-6).all()
        assert (self.foliage.uvs[:, 0] >= 0.5 - 1e-6).all()
        for part in (self.wood, self.foliage):
            assert (part.uvs >= -1e-6).all() and (part.uvs <= 1.0 + 1e-6).all()

    def test_determinism(self):
        *_, m1 = _build(33)
        *_, m2 = _build(33)
        assert np.array_equal(m1.positions, m2.positions)
        assert np.array_equal(m1.indices, m2.indices)
        assert np.array_equal(m1.colors, m2.colors)


class TestMergeAndEmpty:
    def test_merge_offsets_indices(self):
        *_, wood, foliage, merged = _build()
        assert merged.n_vertices == wood.n_vertices + foliage.n_vertices
        assert merged.indices.shape[0] == (wood.indices.shape[0] + foliage.indices.shape[0])
        # Second part's indices reference its own vertex block.
        second = merged.indices[wood.indices.shape[0] :]
        assert int(second.min()) >= wood.n_vertices

    def test_empty_leaves_mesh_ok(self):
        set_world_seed(3)
        rng = for_domain("test", "deadmesh")
        sb = SkeletonBuilder(rng)
        sb.trunk(height_m=4.0, base_radius_m=0.2)
        sk = sb.skeleton()
        wood = mesh_branches(sk)
        from fire_engine.procedural.flora import Leaves

        foliage = mesh_leaves(Leaves.empty(), rng)
        merged = merge_parts(wood, foliage)
        assert foliage.n_vertices == 0
        assert merged.n_vertices == wood.n_vertices
        assert merged.height_m > 3.0

    def test_one_quad_per_leaf(self):
        _, leaves, _, foliage, _ = _build()
        assert foliage.n_vertices == leaves.n_leaves * 4
        assert foliage.indices.shape[0] == leaves.n_leaves * 6

    def test_leaf_normals_upward_biased(self):
        # Tilt range caps at 70° off vertical — every leaf normal keeps a
        # positive Z component (canopies light from above).
        _, _, _, foliage, _ = _build()
        assert (foliage.normals[:, 2] > 0.0).all()
