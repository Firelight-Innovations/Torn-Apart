"""
procedural/flora/mesher.py — skeleton + leaves → renderable tree mesh arrays.

The shared geometry stage of the 3-D flora pipeline: species scripts grow a
:class:`~fire_engine.procedural.flora.skeleton.TreeSkeleton` and a
:class:`~fire_engine.procedural.flora.leaves.LeafClusters`; this module turns
them into a :class:`TreeMesh` — pure numpy arrays in EXACTLY the layout of
the engine's interleaved V3N3T2C4 vertex format
(``world/geometry_bridge.make_vertex_format``), so the renderer uploads a
variant with the existing one-memoryview-write bridge.

Geometry style (the pixel-art read):
- **Branches** are tapered square prisms (``sides=4``) with flat per-face
  normals — chunky Minecraft/Vintage-Story wood, one quad per side plus a
  tip cap.
- **Leaf clusters** are crossed vertical quads plus one horizontal top quad
  (the top quad catches sun Lambert from real normals so canopies read
  solid from above), alpha-cutout against the species atlas's leaf region.

Per-vertex **sway weight** is baked into ``colors[:, 3]``: ≈0 on the trunk
base, rising along branches, ≈1 on leaves.  ``world/shaders/tree.vert``
multiplies wind lean by this weight — trunks pin, canopies ride gusts.

Everything here is headless, deterministic and vectorized over all segments/
clusters at once (Hard Rules 1, 2, 4).

Units: meters, Z-up, tree-local (trunk base at the origin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fire_engine.procedural.flora.leaves import LeafClusters
from fire_engine.procedural.flora.skeleton import (
    TreeSkeleton,
    _frames,
    _normalize,
)

__all__ = ["TreeMesh", "mesh_branches", "mesh_leaf_clusters", "merge_parts"]


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
        """Vertex count ``N``."""
        return int(self.positions.shape[0])

    @staticmethod
    def empty() -> "TreeMesh":
        """A zero-vertex mesh part."""
        return TreeMesh(positions=np.empty((0, 3), dtype=np.float32),
                        normals=np.empty((0, 3), dtype=np.float32),
                        uvs=np.empty((0, 2), dtype=np.float32),
                        colors=np.empty((0, 4), dtype=np.float32),
                        indices=np.empty(0, dtype=np.uint32),
                        height_m=0.0, radius_m=0.0)


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


def mesh_branches(
    sk: TreeSkeleton,
    *,
    sides: int = 4,
    uv_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 1.0),
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cap_tips: bool = True,
) -> TreeMesh:
    """
    Mesh every skeleton segment as a tapered ``sides``-gon prism.

    Each segment becomes ``sides`` independent flat-shaded quads (plus an
    end-cap fan when *cap_tips*), fully vectorized over all segments at
    once.  ``radius`` is the prism's half-width (apothem), so a trunk of
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
        Add an end-cap polygon at every segment's end ring so cut branch
        ends never look hollow.  Default True.

    Returns
    -------
    TreeMesh
        ``S × sides`` quads (+ caps); ``colors[:, 3]`` carries sway.
    """
    S = sk.n_segments
    if S == 0:
        return TreeMesh.empty()
    u0, v0, u1, v1 = (float(c) for c in uv_rect)

    axis = _normalize(sk.end - sk.start)                       # (S, 3)
    fu, fv = _frames(axis)                                     # (S, 3) each
    theta = ((np.arange(sides, dtype=np.float32) + 0.5)
             * (2.0 * math.pi / sides))                        # (sides,)
    corner_mult = 1.0 / math.cos(math.pi / sides)
    # Corner directions per segment: (S, sides, 3).
    cd = (fu[:, None, :] * np.cos(theta)[None, :, None]
          + fv[:, None, :] * np.sin(theta)[None, :, None])
    ring0 = (sk.start[:, None, :]
             + cd * (sk.radius_start * corner_mult)[:, None, None])
    ring1 = (sk.end[:, None, :]
             + cd * (sk.radius_end * corner_mult)[:, None, None])

    nxt = (np.arange(sides) + 1) % sides
    # Side-face quads (S, sides, 4, 3): start_k, start_k+1, end_k+1, end_k —
    # CCW seen from outside (right-handed frames; tree node is two-sided
    # anyway and tree.frag flips back faces).
    p0, p1 = ring0, ring0[:, nxt]
    p2, p3 = ring1[:, nxt], ring1
    quads = np.stack([p0, p1, p2, p3], axis=2).astype(np.float32)

    face_n = _normalize(np.cross(p1 - p0, p3 - p0))            # (S, sides, 3)
    normals = np.broadcast_to(face_n[:, :, None, :], quads.shape)

    uv_quad = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
                       dtype=np.float32)
    uvs = np.broadcast_to(uv_quad[None, None, :, :],
                          (S, sides, 4, 2))

    sway0 = sk.sway_start()
    sway_vert = np.stack([sway0, sway0, sk.sway, sk.sway], axis=1)  # (S, 4)
    colors = np.empty((S, sides, 4, 4), dtype=np.float32)
    colors[..., 0:3] = np.asarray(tint, dtype=np.float32)
    colors[..., 3] = sway_vert[:, None, :]

    positions = quads.reshape(-1, 3)
    normals = np.ascontiguousarray(normals.reshape(-1, 3), dtype=np.float32)
    uvs = np.ascontiguousarray(uvs.reshape(-1, 2), dtype=np.float32)
    colors = colors.reshape(-1, 4)
    indices = _quad_indices(S * sides)

    if cap_tips:
        # End-cap fan per segment over its `sides` end-ring corners; normal
        # along the segment axis; bark-rect center UV (caps are tiny).
        cap_verts = ring1.reshape(-1, 3).astype(np.float32)        # (S*sides, 3)
        cap_norms = np.repeat(axis, sides, axis=0).astype(np.float32)
        cap_uv = np.full((S * sides, 2),
                         ((u0 + u1) * 0.5, (v0 + v1) * 0.5), dtype=np.float32)
        cap_col = np.empty((S * sides, 4), dtype=np.float32)
        cap_col[:, 0:3] = np.asarray(tint, dtype=np.float32)
        cap_col[:, 3] = np.repeat(sk.sway, sides)
        # Fan indices: (0, k, k+1) per cap, offset past the side-face verts.
        offset = positions.shape[0]
        base = (np.arange(S, dtype=np.uint32) * sides)[:, None] + offset
        k = np.arange(1, sides - 1, dtype=np.uint32)
        fan = np.stack([np.zeros_like(k), k, k + 1], axis=1).reshape(-1)
        cap_idx = (base + fan[None, :]).reshape(-1)

        positions = np.concatenate([positions, cap_verts])
        normals = np.concatenate([normals, cap_norms])
        uvs = np.concatenate([uvs, cap_uv])
        colors = np.concatenate([colors, cap_col])
        indices = np.concatenate([indices, cap_idx.astype(np.uint32)])

    height, radius = _metadata(positions)
    return TreeMesh(positions=np.ascontiguousarray(positions, np.float32),
                    normals=np.ascontiguousarray(normals, np.float32),
                    uvs=np.ascontiguousarray(uvs, np.float32),
                    colors=np.ascontiguousarray(colors, np.float32),
                    indices=np.ascontiguousarray(indices, np.uint32),
                    height_m=height, radius_m=radius)


def mesh_leaf_clusters(
    clusters: LeafClusters,
    rng: np.random.Generator,
    *,
    quads: int = 3,
    top_quad: bool = True,
    uv_rect: tuple[float, float, float, float] = (0.5, 0.0, 1.0, 1.0),
    size_jitter: tuple[float, float] = (0.8, 1.25),
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> TreeMesh:
    """
    Mesh leaf clusters as crossed vertical quads (+ one horizontal top quad).

    Each cluster becomes *quads* vertical square quads fanned evenly around
    Z at a random base yaw, ``2 × radius × jitter`` wide/tall, centered on
    the cluster — plus, when *top_quad*, one horizontal quad slightly above
    center whose +Z normal catches overhead sun (canopies read solid from
    above instead of as flat fins).  All quads UV-map the full leaf rect of
    the species atlas (alpha-cutout texels carve the blob shape).

    Parameters
    ----------
    clusters : LeafClusters
        From ``leaves.leaf_clusters_at_tips`` (may be empty).
    rng : numpy.random.Generator
        Deterministic generator (per-cluster yaw + size jitter).
    quads : int
        Vertical crossed quads per cluster.  Default 3.
    top_quad : bool
        Add the horizontal top quad.  Default True.
    uv_rect : tuple
        ``(u0, v0, u1, v1)`` leaf sub-rect of the atlas (right half by
        default, per ``atlas.AtlasLayout``).
    size_jitter : tuple[float, float]
        Uniform per-cluster size multiplier.  Default (0.8, 1.25).
    tint : tuple[float, float, float]
        RGB multiplier baked into vertex colors.

    Returns
    -------
    TreeMesh
        ``colors[:, 3]`` carries the cluster sway weight on every vertex.
    """
    L = clusters.n_clusters
    if L == 0:
        return TreeMesh.empty()
    u0, v0, u1, v1 = (float(c) for c in uv_rect)

    half = (clusters.radius
            * rng.uniform(size_jitter[0], size_jitter[1], L)
            ).astype(np.float32)                               # (L,)
    yaw0 = rng.uniform(0.0, math.pi, L).astype(np.float32)     # (L,)
    theta = (yaw0[:, None]
             + (np.arange(quads, dtype=np.float32)
                * (math.pi / quads))[None, :])                 # (L, quads)
    c, s = np.cos(theta), np.sin(theta)
    d = np.stack([c, s, np.zeros_like(c)], axis=-1)            # (L, quads, 3)
    zhat = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    ctr = clusters.center[:, None, :]                          # (L, 1, 3)
    hw = half[:, None, None]
    # Quad corners: center ± d·half, z ± half (square quads).
    p0 = ctr - d * hw - zhat * hw
    p1 = ctr + d * hw - zhat * hw
    p2 = ctr + d * hw + zhat * hw
    p3 = ctr - d * hw + zhat * hw
    verts = np.stack([p0, p1, p2, p3], axis=2).astype(np.float32)
    n = np.stack([-s, c, np.zeros_like(c)], axis=-1)           # horiz normal
    normals = np.broadcast_to(n[:, :, None, :], verts.shape)

    n_quads_total = L * quads
    positions = verts.reshape(-1, 3)
    normals = np.ascontiguousarray(normals.reshape(-1, 3), dtype=np.float32)
    uv_quad = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
                       dtype=np.float32)
    uvs = np.tile(uv_quad, (n_quads_total, 1))
    colors = np.empty((n_quads_total * 4, 4), dtype=np.float32)
    colors[:, 0:3] = np.asarray(tint, dtype=np.float32)
    colors[:, 3] = np.repeat(clusters.sway, quads * 4)

    if top_quad:
        # Horizontal quad above center, aligned to the cluster's base yaw.
        dx = np.stack([np.cos(yaw0), np.sin(yaw0),
                       np.zeros_like(yaw0)], axis=-1)          # (L, 3)
        dy = np.stack([-np.sin(yaw0), np.cos(yaw0),
                       np.zeros_like(yaw0)], axis=-1)
        top_c = clusters.center + zhat[None, :] * (half * 0.35)[:, None]
        hwl = half[:, None]
        t0 = top_c - dx * hwl - dy * hwl
        t1 = top_c + dx * hwl - dy * hwl
        t2 = top_c + dx * hwl + dy * hwl
        t3 = top_c - dx * hwl + dy * hwl
        tverts = np.stack([t0, t1, t2, t3], axis=1) \
            .reshape(-1, 3).astype(np.float32)
        tnorms = np.tile(zhat, (L * 4, 1)).astype(np.float32)
        tuvs = np.tile(uv_quad, (L, 1))
        tcols = np.empty((L * 4, 4), dtype=np.float32)
        tcols[:, 0:3] = np.asarray(tint, dtype=np.float32)
        tcols[:, 3] = np.repeat(clusters.sway, 4)

        positions = np.concatenate([positions, tverts])
        normals = np.concatenate([normals, tnorms])
        uvs = np.concatenate([uvs, tuvs])
        colors = np.concatenate([colors, tcols])
        n_quads_total += L

    indices = _quad_indices(n_quads_total)
    height, radius = _metadata(positions)
    return TreeMesh(positions=np.ascontiguousarray(positions, np.float32),
                    normals=np.ascontiguousarray(normals, np.float32),
                    uvs=np.ascontiguousarray(uvs, np.float32),
                    colors=np.ascontiguousarray(colors, np.float32),
                    indices=np.ascontiguousarray(indices, np.uint32),
                    height_m=height, radius_m=radius)


def merge_parts(*parts: TreeMesh) -> TreeMesh:
    """
    Concatenate mesh parts (offsetting indices) into one draw-ready mesh.

    Typically ``merge_parts(mesh_branches(...), mesh_leaf_clusters(...))`` —
    one tree variant, one Geom, one draw.  Empty parts are skipped;
    ``height_m`` / ``radius_m`` are recomputed over the union.

    Returns
    -------
    TreeMesh
    """
    live = [p for p in parts if p.n_vertices > 0]
    if not live:
        return TreeMesh.empty()
    offsets = np.cumsum([0] + [p.n_vertices for p in live[:-1]])
    positions = np.concatenate([p.positions for p in live])
    indices = np.concatenate(
        [p.indices.astype(np.uint64) + off
         for p, off in zip(live, offsets)]).astype(np.uint32)
    height, radius = _metadata(positions)
    return TreeMesh(
        positions=np.ascontiguousarray(positions, np.float32),
        normals=np.ascontiguousarray(
            np.concatenate([p.normals for p in live]), np.float32),
        uvs=np.ascontiguousarray(
            np.concatenate([p.uvs for p in live]), np.float32),
        colors=np.ascontiguousarray(
            np.concatenate([p.colors for p in live]), np.float32),
        indices=np.ascontiguousarray(indices, np.uint32),
        height_m=height, radius_m=radius)
