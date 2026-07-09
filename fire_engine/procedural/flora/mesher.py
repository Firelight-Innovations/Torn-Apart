"""
procedural/flora/mesher.py — skeleton + leaves → renderable tree mesh arrays.

The shared geometry stage of the 3-D flora pipeline: species scripts grow a
:class:`~fire_engine.procedural.flora.skeleton.TreeSkeleton` and a
:class:`~fire_engine.procedural.flora.leaves.Leaves`; this module turns
them into a :class:`TreeMesh` — pure numpy arrays in EXACTLY the layout of
the engine's interleaved V3N3T2C4 vertex format
(``world/geometry_bridge.make_vertex_format``), so the renderer uploads a
variant with the existing one-memoryview-write bridge.

Geometry style (the pixel-art read):
- **Branches** are tapered square prisms (``sides=4``) with flat per-face
  normals — chunky Minecraft/Vintage-Story wood, one quad per side plus a
  tip cap.
- **Leaves** are INDIVIDUAL small quads — one card per leaf from the CA
  grower (``leaves.leaves_at_tips``), each with its own upward-biased
  random orientation so Lambert dapples the canopy leaf by leaf, all
  alpha-cutout against the species atlas's single-leaf texture and merged
  into the variant mesh (hundreds of leaves, still ONE draw per variant).

Per-vertex **sway weight** is baked into ``colors[:, 3]``: ≈0 on the trunk
base, rising along branches, ≈1 on leaves.  ``world/shaders/tree.vert``
multiplies wind lean by this weight — trunks pin, canopies ride gusts.

Everything here is headless, deterministic and vectorized over all segments/
clusters at once (Hard Rules 1, 2, 4).

Units: meters, Z-up, tree-local (trunk base at the origin).

Docs: docs/systems/procedural.flora.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fire_engine.procedural.flora.leaves import Leaves
from fire_engine.procedural.flora.skeleton import (
    TreeSkeleton,
    _frames,
    _normalize,
)

__all__ = ["TreeMesh", "merge_parts", "mesh_branches", "mesh_leaf_area_m2", "mesh_leaves"]


@dataclass
class TreeMesh:
    """
    Renderable mesh arrays for one tree variant — V3N3T2C4, field-for-field.

    Upload with ``world/geometry_bridge.to_geom`` (the terrain bridge — this
    dataclass intentionally exposes the same attribute names as terrain's
    ``MeshArrays``).

    Attributes
    ----------
    positions : numpy.ndarray
        ``float32 (N, 3)`` tree-local meters, Z-up, base at the origin.
    normals : numpy.ndarray
        ``float32 (N, 3)`` unit, flat per face.
    uvs : numpy.ndarray
        ``float32 (N, 2)`` into the species atlas (bark rect for wood,
        leaf rect for foliage — see ``procedural/flora/atlas.py``).
    colors : numpy.ndarray
        ``float32 (N, 4)`` — RGB albedo tint multiplier, **A = sway
        weight** in [0, 1] (NOT alpha; the shader reads it as wind weight).
    indices : numpy.ndarray
        ``uint32 (M,)`` triangle list.
    height_m : float
        Top of the mesh above the base (m).
    radius_m : float
        Max horizontal reach from the trunk axis (m) — for render bounds.

    Docs: docs/systems/procedural.flora.md
    """

    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    colors: np.ndarray
    indices: np.ndarray
    height_m: float
    radius_m: float

    @property
    def n_vertices(self) -> int:
        """Vertex count ``N``.

        Docs: docs/systems/procedural.flora.md
        """
        return int(self.positions.shape[0])

    @staticmethod
    def empty() -> TreeMesh:
        """A zero-vertex mesh part.

        Docs: docs/systems/procedural.flora.md
        """
        return TreeMesh(
            positions=np.empty((0, 3), dtype=np.float32),
            normals=np.empty((0, 3), dtype=np.float32),
            uvs=np.empty((0, 2), dtype=np.float32),
            colors=np.empty((0, 4), dtype=np.float32),
            indices=np.empty(0, dtype=np.uint32),
            height_m=0.0,
            radius_m=0.0,
        )


def mesh_leaf_area_m2(mesh: TreeMesh) -> float:
    """
    Total one-sided LEAF area of a variant mesh, in square meters.

    Leaf triangles are identified by the atlas layout contract
    (``procedural/flora/atlas.py``): leaves map into the atlas's RIGHT half
    (``uv.x >= 0.5``), bark into the left.  A triangle counts as leaf when
    all three of its vertices sit in the leaf half.

    This is the "how thick are the leaves" measure the lighting occluders
    use: leaf area ÷ canopy volume gives a per-meter extinction density, so
    a dense oak crown blocks more sun than a scraggly snag with two tufts
    (see ``lighting/occluders.py`` and ``world/tree_renderer.py``).

    Parameters
    ----------
    mesh : TreeMesh
        One variant mesh (tree-local meters).

    Returns
    -------
    float
        Sum of leaf-triangle areas (m²); 0.0 for a leafless mesh.

    Example
    -------
    >>> from fire_engine.procedural.flora.mesher import TreeMesh, mesh_leaf_area_m2
    >>> mesh_leaf_area_m2(TreeMesh.empty())
    0.0

    Docs: docs/systems/procedural.flora.md
    """
    if mesh.indices.shape[0] == 0:
        return 0.0
    tris = mesh.indices.reshape(-1, 3)
    leaf_vert = mesh.uvs[:, 0] >= 0.5
    leaf_tri = leaf_vert[tris].all(axis=1)
    if not bool(leaf_tri.any()):
        return 0.0
    p = mesh.positions
    t = tris[leaf_tri]
    e1 = p[t[:, 1]] - p[t[:, 0]]
    e2 = p[t[:, 2]] - p[t[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    return float(areas.sum())


def _quad_indices(n_quads: int) -> np.ndarray:
    """Triangle indices for ``n_quads`` independent 4-vertex quads (uint32)."""
    base = (np.arange(n_quads, dtype=np.uint32) * 4)[:, None]
    quad = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)[None, :]
    return (base + quad).reshape(-1)


def _metadata(positions: np.ndarray) -> tuple[float, float]:
    """(height_m, radius_m) of a vertex cloud (0, 0 when empty)."""
    if positions.shape[0] == 0:
        return 0.0, 0.0
    height = float(max(positions[:, 2].max(), 0.0))
    radius = float(np.linalg.norm(positions[:, 0:2], axis=1).max())
    return height, radius


def _segment_frames(sk: TreeSkeleton, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    One cross-section frame ``(u, v)`` per segment via parallel transport.

    Returns two ``float32 (S, 3)`` basis arrays, perpendicular to each
    segment's *axis*, propagated parent→child by the **minimal** rotation
    that carries a parent's axis onto its child's axis.  This rotation-
    minimizing (no arbitrary twist) frame is what lets adjacent rings line
    up corner-for-corner, so a continuation joint welds with zero twist
    instead of the per-segment ``_frames`` look (each ring rotated
    independently → visible notches at every joint).

    Root segments (``parent == -1``) seed from :func:`_frames`.  Segments
    are processed in ascending id order; children always have larger ids
    than their parents, so every parent frame is ready first (tens of
    segments — a plain Python loop is allowed by the module contract).
    """
    u, v = (a.copy() for a in _frames(axis))  # seed frames (used for roots)
    parent = sk.parent
    for i in range(sk.n_segments):
        p = int(parent[i])
        if p < 0:
            continue  # root: keep the seed frame
        # Minimal rotation carrying the parent's axis onto the child's.
        rot_axis = np.cross(axis[p], axis[i])
        rl = float(np.linalg.norm(rot_axis))
        cos_a = float(np.clip(np.dot(axis[p], axis[i]), -1.0, 1.0))
        if rl < 1e-8:  # axes (anti)parallel — carry the in-plane basis straight
            sgn = 1.0 if cos_a > 0.0 else -1.0
            u[i], v[i] = sgn * u[p], v[p]
            continue
        k = rot_axis / rl  # unit rotation axis; sin(angle) == rl
        for dst, src in ((u, u[p]), (v, v[p])):  # Rodrigues per basis vector
            dst[i] = src * cos_a + np.cross(k, src) * rl + k * float(np.dot(k, src)) * (1.0 - cos_a)
    return _normalize(u), _normalize(v)


def mesh_branches(
    sk: TreeSkeleton,
    *,
    sides: int = 4,
    uv_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 1.0),
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cap_tips: bool = True,
    weld_tol_m: float = 1e-4,
) -> TreeMesh:
    """
    Mesh every skeleton segment as a tapered ``sides``-gon prism — joints
    weld into one **continuous tube** (no gaps or twists at segment seams).

    Each segment becomes ``sides`` independent flat-shaded quads (plus an
    end-cap fan when *cap_tips*) — every quad keeps its own four vertices and
    face normal for the chunky pixel read.  Continuity comes from the ring
    *positions*, not index sharing: every segment gets ONE cross-section
    frame propagated parent→child by the minimal (twist-free) rotation
    (:func:`_segment_frames`); a continuation (``start ≈ parent.end`` within
    *weld_tol_m*) reuses its parent's end-ring positions (zero gap down a
    whole trunk); a fork (start partway along the parent) pulls its base ring
    back into the parent by ~the attachment radius so it sockets in.

    ``radius`` is the prism's half-width (apothem), so a trunk of
    ``radius_start=0.25`` renders 0.5 m thick — corner vertices sit at
    ``radius / cos(π/sides)``.

    Parameters
    ----------
    sk : TreeSkeleton
        Finalized skeleton.
    sides : int
        Cross-section sides.  4 (square) is the engine's pixel look.
    uv_rect : tuple
        ``(u0, v0, u1, v1)`` atlas sub-rect for bark (defaults to the left
        half per ``atlas.AtlasLayout``).  U spans each face, V spans each
        segment's length.
    tint : tuple[float, float, float]
        RGB multiplier baked into vertex colors (the species' per-variant
        hue drift).
    cap_tips : bool
        Add an end-cap polygon at every TIP segment's end ring (childless
        segments only — interior caps are hidden inside the next ring and
        are skipped to save verts) so cut branch ends never look hollow.
        Default True.
    weld_tol_m : float
        Distance (m) under which a child's start counts as a true
        continuation of its parent's end and the two rings are welded.
        Default 1e-4.

    Returns
    -------
    TreeMesh
        ``S × sides`` quads (+ tip caps); ``colors[:, 3]`` carries sway.

    Docs: docs/systems/procedural.flora.md
    """
    S = sk.n_segments
    if S == 0:
        return TreeMesh.empty()
    u0, v0, u1, v1 = (float(c) for c in uv_rect)

    axis = _normalize(sk.end - sk.start)  # (S, 3)
    fu, fv = _segment_frames(sk, axis)  # (S, 3) each — twist-free propagation
    theta = (np.arange(sides, dtype=np.float32) + 0.5) * (2.0 * math.pi / sides)  # (sides,)
    corner_mult = 1.0 / math.cos(math.pi / sides)
    # Corner directions per segment: (S, sides, 3).
    cd = (
        fu[:, None, :] * np.cos(theta)[None, :, None]
        + fv[:, None, :] * np.sin(theta)[None, :, None]
    )

    # Ring centers.  A fork's start is constructed ON its parent's CENTRE
    # axis (skeleton.branches), so the base ring already sits ~the parent
    # radius deep inside the wood — it sockets cleanly with zero extra work.
    # The previous code pushed the base back along -axis by the parent's
    # radius "to bury it", but that shoved the ring PAST the centre and out
    # the FAR wall: the branch's back end protruded from the opposite side of
    # the trunk (the "branches poke through the trunk" bug, worst at the thick
    # trunk→limb joints).  The centre-line position is provably optimal —
    # maximally buried (parent_radius from the near wall) yet never crossing
    # to the far side — so we leave the start ring exactly at the attachment
    # point.  A "fork" is a child whose start is NOT at its parent's end.
    start_ctr = sk.start.copy()
    parent = sk.parent
    has_parent = parent >= 0
    p_safe = np.maximum(parent, 0)
    is_continuation = has_parent & np.all(np.abs(sk.start - sk.end[p_safe]) <= weld_tol_m, axis=1)

    ring0 = start_ctr[:, None, :] + cd * (sk.radius_start * corner_mult)[:, None, None]
    ring1 = sk.end[:, None, :] + cd * (sk.radius_end * corner_mult)[:, None, None]

    # Weld continuations: copy the parent's end-ring positions into the
    # child's start ring (frames+radii already match) → exact zero-gap joint.
    if is_continuation.any():
        for i in np.nonzero(is_continuation)[0]:
            ring0[i] = ring1[int(parent[i])]

    nxt = (np.arange(sides) + 1) % sides
    # Side-face quads (S, sides, 4, 3): start_k, start_k+1, end_k+1, end_k — CCW
    # seen from outside (tree node is two-sided; tree.frag flips back faces).
    p0, p1 = ring0, ring0[:, nxt]
    p2, p3 = ring1[:, nxt], ring1
    quads = np.stack([p0, p1, p2, p3], axis=2).astype(np.float32)

    face_n = _normalize(np.cross(p1 - p0, p3 - p0))  # (S, sides, 3)
    normals = np.broadcast_to(face_n[:, :, None, :], quads.shape)

    uv_quad = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float32)
    uvs = np.broadcast_to(uv_quad[None, None, :, :], (S, sides, 4, 2))

    sway0 = sk.sway_start()
    sway_vert = np.stack([sway0, sway0, sk.sway, sk.sway], axis=1)  # (S, 4)
    colors: np.ndarray = np.empty((S, sides, 4, 4), dtype=np.float32)
    colors[..., 0:3] = np.asarray(tint, dtype=np.float32)
    colors[..., 3] = sway_vert[:, None, :]

    positions = quads.reshape(-1, 3)
    normals = np.ascontiguousarray(normals.reshape(-1, 3), dtype=np.float32)
    uvs = np.ascontiguousarray(uvs.reshape(-1, 2), dtype=np.float32)
    colors = colors.reshape(-1, 4)
    indices = _quad_indices(S * sides)

    if cap_tips:
        # Cap ONLY true tips (childless segments) — interior caps sit inside
        # the next welded ring and are never seen, so emitting them just
        # wastes verts.  One end-cap fan per tip over its `sides` end-ring
        # corners; normal along the segment axis; bark-rect center UV.
        tip_ids = sk.tip_ids().astype(np.int64)
        if tip_ids.size:
            cap_verts = ring1[tip_ids].reshape(-1, 3).astype(np.float32)  # (T*sides, 3)
            cap_norms = np.repeat(axis[tip_ids], sides, axis=0).astype(np.float32)
            cap_uv = np.full(
                (tip_ids.size * sides, 2), ((u0 + u1) * 0.5, (v0 + v1) * 0.5), dtype=np.float32
            )
            cap_col = np.empty((tip_ids.size * sides, 4), dtype=np.float32)
            cap_col[:, 0:3] = np.asarray(tint, dtype=np.float32)
            cap_col[:, 3] = np.repeat(sk.sway[tip_ids], sides)
            # Fan indices: (0, k, k+1) per cap, offset past the side-face verts.
            offset = positions.shape[0]
            base = (np.arange(tip_ids.size, dtype=np.uint32) * sides)[:, None] + offset
            k = np.arange(1, sides - 1, dtype=np.uint32)
            fan = np.stack([np.zeros_like(k), k, k + 1], axis=1).reshape(-1)
            cap_idx = (base + fan[None, :]).reshape(-1)

            positions = np.concatenate([positions, cap_verts])
            normals = np.concatenate([normals, cap_norms])
            uvs = np.concatenate([uvs, cap_uv])
            colors = np.concatenate([colors, cap_col])
            indices = np.concatenate([indices, cap_idx.astype(np.uint32)])

    height, radius = _metadata(positions)
    return TreeMesh(
        positions=np.ascontiguousarray(positions, np.float32),
        normals=np.ascontiguousarray(normals, np.float32),
        uvs=np.ascontiguousarray(uvs, np.float32),
        colors=np.ascontiguousarray(colors, np.float32),
        indices=np.ascontiguousarray(indices, np.uint32),
        height_m=height,
        radius_m=radius,
    )


def mesh_leaves(
    leaves: Leaves,
    rng: np.random.Generator,
    *,
    uv_rect: tuple[float, float, float, float] = (0.5, 0.0, 1.0, 1.0),
    yaw_jitter_rad: float = 0.4,
    size_jitter: tuple[float, float] = (0.85, 1.2),
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> TreeMesh:
    """
    Mesh individual leaves as one small quad each, **rooted at the base**.

    A leaf is built like a real one: the card's base edge sits on the leaf's
    anchor (``leaves.center - out_dir·radius``, a point ON the bark) and the
    blade extends OUTWARD along ``leaves.out_dir`` (outward off the branch
    with an upward reach).  So every leaf visibly grows from its branch in a
    sensible direction instead of being a centre-pinned card facing a random
    way.  A small per-leaf yaw twist about the growth axis (*yaw_jitter_rad*)
    plus the varied growth directions give the canopy its dappled Lambert
    read.  The card normal is pushed into the upper hemisphere so foliage
    catches the overhead light consistently (richer leaf shading — proper
    subsurface scattering — is iteration 5).  All cards UV-map the full
    single-leaf rect of the species atlas; 4 verts / 2 tris per leaf, merged
    into the variant mesh (one draw, fully GPU-batched).

    Parameters
    ----------
    leaves : Leaves
        From ``leaves.leaves_at_tips`` (may be empty).
    rng : numpy.random.Generator
        Deterministic generator (per-leaf yaw twist, size jitter).
    uv_rect : tuple
        ``(u0, v0, u1, v1)`` leaf sub-rect of the atlas (right half by
        default, per ``atlas.AtlasLayout``).  ``v0`` is the stem edge.
    yaw_jitter_rad : float
        Max per-leaf twist of the card about its growth axis (radians) — a
        little so neighbouring leaves don't all lie in one plane.  Default 0.4.
    size_jitter : tuple[float, float]
        Uniform per-leaf size multiplier on ``leaves.radius``.
    tint : tuple[float, float, float]
        RGB multiplier baked into vertex colors.

    Returns
    -------
    TreeMesh
        ``colors[:, 3]`` carries the per-leaf sway weight on every vertex.

    Docs: docs/systems/procedural.flora.md
    """
    L = leaves.n_leaves
    if L == 0:
        return TreeMesh.empty()
    u0, v0, u1, v1 = (float(c) for c in uv_rect)

    half = (leaves.radius * rng.uniform(size_jitter[0], size_jitter[1], L)).astype(np.float32)

    # Growth (stem→tip) axis and a width axis perpendicular to it.  The width
    # axis starts horizontal (cross with world up) so the leaf blade isn't
    # edge-on to the camera, then gets a small per-leaf twist about the growth
    # axis for variety.
    g = _normalize(leaves.out_dir.astype(np.float32))  # (L, 3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(g, up[None, :])
    rn = np.linalg.norm(right, axis=1, keepdims=True)
    fallback = np.array([[1.0, 0.0, 0.0]], np.float32)
    right = np.where(rn > 1e-6, right / np.maximum(rn, 1e-9), fallback)
    ang = rng.uniform(-yaw_jitter_rad, yaw_jitter_rad, L).astype(np.float32)[:, None]
    ca, sa = np.cos(ang), np.sin(ang)  # Rodrigues twist of `right` about unit `g`
    twist = np.sum(g * right, axis=1, keepdims=True) * (1.0 - ca)
    right = _normalize(right * ca + np.cross(g, right) * sa + g * twist)
    # Card normal ⟂ blade; flip into the upper hemisphere so leaves face the
    # sky-ish light (iteration-5 shading will supersede this).
    n = _normalize(np.cross(g, right))
    n = np.where(n[:, 2:3] < 0.0, -n, n).astype(np.float32)

    hw = half[:, None]
    base = leaves.center.astype(np.float32) - g * hw  # base edge ≈ the bark anchor
    length = (2.0 * half)[:, None]
    # Base edge (p0,p1) on the anchor; tip edge (p2,p3) one leaf-length out.
    p0 = base - right * hw
    p1 = base + right * hw
    p2 = base + right * hw + g * length
    p3 = base - right * hw + g * length
    positions = np.stack([p0, p1, p2, p3], axis=1).reshape(-1, 3).astype(np.float32)
    normals = np.repeat(n, 4, axis=0)

    uv_quad = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float32)
    uvs = np.tile(uv_quad, (L, 1))
    colors = np.empty((L * 4, 4), dtype=np.float32)
    colors[:, 0:3] = np.asarray(tint, dtype=np.float32)
    colors[:, 3] = np.repeat(leaves.sway, 4)

    indices = _quad_indices(L)
    height, radius = _metadata(positions)
    return TreeMesh(
        positions=np.ascontiguousarray(positions, np.float32),
        normals=np.ascontiguousarray(normals, np.float32),
        uvs=np.ascontiguousarray(uvs, np.float32),
        colors=np.ascontiguousarray(colors, np.float32),
        indices=np.ascontiguousarray(indices, np.uint32),
        height_m=height,
        radius_m=radius,
    )


def merge_parts(*parts: TreeMesh) -> TreeMesh:
    """
    Concatenate mesh parts (offsetting indices) into one draw-ready mesh.

    Typically ``merge_parts(mesh_branches(...), mesh_leaves(...))`` —
    one tree variant, one Geom, one draw.  Empty parts are skipped;
    ``height_m`` / ``radius_m`` are recomputed over the union.

    Returns
    -------
    TreeMesh

    Docs: docs/systems/procedural.flora.md
    """
    live = [p for p in parts if p.n_vertices > 0]
    if not live:
        return TreeMesh.empty()
    offsets = np.cumsum([0] + [p.n_vertices for p in live[:-1]])
    positions = np.concatenate([p.positions for p in live])
    indices = np.concatenate(
        [p.indices.astype(np.uint64) + off for p, off in zip(live, offsets, strict=True)]
    ).astype(np.uint32)
    height, radius = _metadata(positions)
    return TreeMesh(
        positions=np.ascontiguousarray(positions, np.float32),
        normals=np.ascontiguousarray(np.concatenate([p.normals for p in live]), np.float32),
        uvs=np.ascontiguousarray(np.concatenate([p.uvs for p in live]), np.float32),
        colors=np.ascontiguousarray(np.concatenate([p.colors for p in live]), np.float32),
        indices=np.ascontiguousarray(indices, np.uint32),
        height_m=height,
        radius_m=radius,
    )
