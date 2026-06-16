"""
tests/procedural/flora/test_tree_cohesion.py
— Geometric-invariant ("mesh cohesion") tests for the 3-D flora pipeline.

These are NOT "does the code run" tests — those live in the per-module test
mirrors.  These assert that the GEOMETRY a species produces is *correct*:
the trunk is one connected piece (no joint gaps), no triangle is degenerate,
no vertex is NaN, and no leaf floats free of the wood.  They are the headless,
deterministic, automated stand-in for a "does the tree look right" visual
check — invariant/property tests over the mesh, not a rendered-image diff.

Two layers:
* Synthetic — skeletons built directly with ``SkeletonBuilder`` so the
  expected chain/segment structure is known exactly (no registry/species
  dependency); these pin the cohesion guarantees of the shared mesher.
* Per-species — every registered tree/bush species grown and meshed; the
  same invariants must hold for real content.  Asserts PROPERTIES, not exact
  counts, so it stays valid as species recipes are tuned.

Headless — no panda3d, fully deterministic (fixed seeds).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.procedural.flora.leaves import Leaves
from fire_engine.procedural.flora.mesher import TreeMesh, merge_parts, mesh_branches, mesh_leaves
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton

# The four shipped species defs (imported directly → no registry cache).
from fire_engine.procedural.flora.species.berry_bush import BerryBushDef
from fire_engine.procedural.flora.species.dead_tree import DeadTreeDef
from fire_engine.procedural.flora.species.gnarled_oak import GnarledOakDef
from fire_engine.procedural.flora.species.scrub_bush import ScrubBushDef
from fire_engine.procedural.flora.types import validate_skeleton

_WELD_M = 1e-4  # position quantum for welding coincident bark vertices (0.1 mm)
_SEED = 1337

SPECIES = [GnarledOakDef, DeadTreeDef, BerryBushDef, ScrubBushDef]
SPECIES_IDS = [d.name for d in SPECIES]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bark_tris(mesh: TreeMesh) -> np.ndarray:
    """Triangles whose 3 vertices are all bark (atlas left half, uv.x < 0.5)."""
    tris = mesh.indices.reshape(-1, 3)
    leaf_vert = mesh.uvs[:, 0] >= 0.5
    return tris[~leaf_vert[tris].any(axis=1)]


def _count_components(n: int, edges: np.ndarray) -> int:
    """Number of connected components over ``n`` welded vertex ids and edges."""
    parent = np.arange(n)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    present = np.unique(edges)
    return len({find(int(v)) for v in present})


def _bark_components(mesh: TreeMesh) -> int:
    """
    Connected components of the BARK mesh after welding vertices that share a
    position (rounded to ``_WELD_M``).

    A continuous tube welds adjacent segment rings to identical positions, so a
    whole trunk chain collapses to ONE component; a pre-fix mesher (independent
    per-segment prisms) leaves every segment its own island.
    """
    btris = _bark_tris(mesh)
    if btris.size == 0:
        return 0
    keys = np.round(mesh.positions / _WELD_M).astype(np.int64)
    _, weld = np.unique(keys, axis=0, return_inverse=True)  # vertex -> welded id
    wt = weld[btris]  # (T, 3) welded ids per bark triangle
    edges = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]], axis=0)
    return _count_components(int(weld.max()) + 1, edges)


def _n_chains(sk: TreeSkeleton) -> int:
    """
    Number of branch CHAINS — a chain starts at a root or at a fork (a child
    whose start is NOT its parent's endpoint).  Continuation segments (a child
    welded onto its parent's tip) extend a chain without starting a new one.
    Equals the expected count of welded bark components.
    """
    parent = sk.parent
    root = parent < 0
    child = parent >= 0
    cont = np.zeros(sk.n_segments, dtype=bool)
    if child.any():
        idx = np.nonzero(child)[0]
        cont[idx] = np.all(np.isclose(sk.start[idx], sk.end[parent[idx]], atol=_WELD_M), axis=1)
    fork = child & ~cont
    return int(root.sum() + fork.sum())


def _leaf_attachment_slack(leaves: Leaves, sk: TreeSkeleton) -> np.ndarray:
    """
    Per-leaf signed slack ``dist_to_nearest_wood - allowed_offset`` (m).

    ``<= 0`` means the leaf sits within ``radius(at nearest point) + 1.5·leaf``
    of some branch — i.e. it hugs the wood and does not float.
    """
    a = sk.start.astype(np.float64)  # (S,3)
    b = sk.end.astype(np.float64)
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)  # (S,)
    c = leaves.center.astype(np.float64)[:, None, :]  # (L,1,3)
    t = np.clip(np.sum((c - a[None]) * ab[None], axis=2) / denom[None], 0.0, 1.0)  # (L,S)
    nearest = a[None] + ab[None] * t[..., None]  # (L,S,3)
    dist = np.linalg.norm(c - nearest, axis=2)  # (L,S)
    j = np.argmin(dist, axis=1)  # nearest segment per leaf
    li = np.arange(leaves.n_leaves)
    d_near = dist[li, j]
    r_near = (sk.radius_start[j] + (sk.radius_end[j] - sk.radius_start[j]) * t[li, j]).astype(
        np.float64
    )
    allowed = r_near + 1.5 * leaves.radius.astype(np.float64) + 1e-3
    return d_near - allowed


def _assert_mesh_integrity(mesh: TreeMesh) -> None:
    """No NaN/inf, valid indices, unit normals, no degenerate tris, ranges."""
    assert mesh.n_vertices > 0
    for arr in (mesh.positions, mesh.normals, mesh.uvs, mesh.colors):
        assert np.isfinite(arr).all(), "non-finite vertex data"
    assert mesh.indices.size % 3 == 0
    assert int(mesh.indices.max()) < mesh.n_vertices and int(mesh.indices.min()) >= 0
    nlen = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(nlen, 1.0, atol=1e-3), f"non-unit normals (min {nlen.min()})"
    # sway weight rides in colors[:, 3]; albedo tint stays sane.
    assert (mesh.colors[:, 3] >= -1e-6).all() and (mesh.colors[:, 3] <= 1.0 + 1e-6).all()
    assert (mesh.uvs >= -1e-6).all() and (mesh.uvs <= 1.0 + 1e-6).all()
    # No degenerate (zero-area) triangles — a collapsed tri is a mesh defect.
    tris = mesh.indices.reshape(-1, 3)
    p = mesh.positions
    area2 = np.linalg.norm(
        np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]), axis=1
    )
    assert (area2 > 1e-9).all(), f"{int((area2 <= 1e-9).sum())} degenerate triangle(s)"


def _grow_species(def_cls, variant: int = 0, seed: int = _SEED):
    """Grow + mesh one variant of a species def directly (no registry cache)."""
    d = def_cls()
    rng = np.random.default_rng(seed)
    sk, leaves = d.grow(rng, variant)
    wood = mesh_branches(sk)
    foliage = mesh_leaves(leaves, rng)
    return sk, leaves, wood, merge_parts(wood, foliage)


# ---------------------------------------------------------------------------
# Synthetic — known structure, pins the shared mesher's cohesion guarantees
# ---------------------------------------------------------------------------


class TestSyntheticTrunkCohesion:
    def test_straight_trunk_is_one_piece(self):
        sb = SkeletonBuilder(np.random.default_rng(1))
        sb.trunk(height_m=5.0, base_radius_m=0.25, segments=6, wobble_m=0.0)
        mesh = mesh_branches(sb.skeleton())
        assert _bark_components(mesh) == 1

    def test_wobbly_trunk_is_one_piece(self):
        # The bend at every node is exactly where the pre-fix mesher tore.
        sb = SkeletonBuilder(np.random.default_rng(2))
        sb.trunk(
            height_m=6.0, base_radius_m=0.3, segments=8, wobble_m=0.5, lean_rad=math.radians(8)
        )
        mesh = mesh_branches(sb.skeleton())
        assert _bark_components(mesh) == 1

    def test_components_bounded_by_chains_not_segments(self):
        sb = SkeletonBuilder(np.random.default_rng(3))
        trunk = sb.trunk(height_m=5.0, base_radius_m=0.28, segments=4, wobble_m=0.3)
        sb.branches(trunk, count=(2, 3), segments=2)
        sk = sb.skeleton()
        mesh = mesh_branches(sk)
        comps = _bark_components(mesh)
        # A chain never FRAGMENTS into per-segment islands (the pre-fix bug had
        # comps == n_segments); sibling branches off a shared point MAY weld
        # together, so the bound is `<=` chains, not `==`.
        assert comps <= _n_chains(sk)
        # Cohesion proof: far fewer islands than segments (welding happened).
        assert comps < sk.n_segments

    def test_trunk_joint_vertices_coincide(self):
        # Direct no-gap check: every continuation child's start ring equals its
        # parent's end ring (welded → zero gap).
        sb = SkeletonBuilder(np.random.default_rng(4))
        sb.trunk(height_m=5.0, base_radius_m=0.25, segments=5, wobble_m=0.4)
        sk = sb.skeleton()
        mesh = mesh_branches(sk)
        # Trunk has no forks → exactly one welded component spanning all rings.
        assert _bark_components(mesh) == 1
        _assert_mesh_integrity(mesh)


# ---------------------------------------------------------------------------
# Per-species — real content must satisfy the same invariants (properties,
# not pinned counts, so density tuning doesn't churn these)
# ---------------------------------------------------------------------------


class TestSpeciesMeshIntegrity:
    @pytest.mark.parametrize("def_cls", SPECIES, ids=SPECIES_IDS)
    def test_every_variant_mesh_is_sound(self, def_cls):
        d = def_cls()
        for v in range(d.variants):
            _, _, _, mesh = _grow_species(def_cls, variant=v)
            _assert_mesh_integrity(mesh)

    @pytest.mark.parametrize("def_cls", SPECIES, ids=SPECIES_IDS)
    def test_trunk_base_at_origin(self, def_cls):
        _, _, mesh, _ = _grow_species(def_cls)
        # Base ring sits on the ground plane; nothing buried far below it.
        assert mesh.positions[:, 2].min() >= -0.1
        assert abs(mesh.positions[:, 2].min()) < 0.2


class TestSpeciesStructuralCohesion:
    @pytest.mark.parametrize("def_cls", SPECIES, ids=SPECIES_IDS)
    def test_skeleton_valid(self, def_cls):
        for v in range(def_cls().variants):
            sk, _, _, _ = _grow_species(def_cls, variant=v)
            validate_skeleton(sk)  # raises on a floating / mis-tapered branch

    @pytest.mark.parametrize("def_cls", SPECIES, ids=SPECIES_IDS)
    def test_bark_chains_never_fragment(self, def_cls):
        # The trunk + every branch is a cohesive tube: welded bark resolves to
        # AT MOST one island per chain (siblings may weld), never one per
        # segment.  Pre-fix this was comps == n_segments (every joint a gap).
        for v in range(def_cls().variants):
            sk, _, wood, _ = _grow_species(def_cls, variant=v)
            if _bark_tris(wood).size == 0:
                continue
            comps = _bark_components(wood)
            assert comps <= _n_chains(sk)
            # Multi-segment trunks/branches mean welding MUST collapse joints.
            if sk.n_segments > _n_chains(sk):
                assert comps < sk.n_segments


class TestSpeciesLeafAttachment:
    @pytest.mark.parametrize("def_cls", SPECIES, ids=SPECIES_IDS)
    def test_no_leaf_floats(self, def_cls):
        for v in range(def_cls().variants):
            sk, leaves, _, _ = _grow_species(def_cls, variant=v)
            if leaves.n_leaves == 0:
                continue
            slack = _leaf_attachment_slack(leaves, sk)
            assert (slack <= 0.0).all(), (
                f"{def_cls.name} v{v}: {int((slack > 0).sum())} floating leaf/leaves "
                f"(worst {slack.max():.3f} m past the allowed offset)"
            )


class TestCohesionDeterminism:
    @pytest.mark.parametrize("def_cls", SPECIES, ids=SPECIES_IDS)
    def test_same_seed_identical_geometry(self, def_cls):
        _, _, _, m1 = _grow_species(def_cls, seed=99)
        _, _, _, m2 = _grow_species(def_cls, seed=99)
        assert np.array_equal(m1.positions, m2.positions)
        assert np.array_equal(m1.indices, m2.indices)
