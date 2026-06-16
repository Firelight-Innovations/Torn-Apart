"""
terrain/surface_nets.py — Flat-shaded naive-surface-nets mesher (faceted look).

This is the "Daggerfall" mesher: terrain that is *between* blocky Minecraft
cubes and fully smooth marching-cubes blobs.  Vertices are pulled onto the
actual solid/air surface (so crater walls become chamfered 45-degree-ish
slopes instead of stair-steps), but every triangle keeps a single flat normal
and its own baked shade, so individual polygons stay clearly visible.

Algorithm (naive surface nets over a binary voxel grid)
--------------------------------------------------------
1. Build a ``(34, 34, 34)`` padded *materials* array (this chunk plus a
   one-voxel shell from all 26 neighbours — face, edge AND corner neighbours,
   because dual cells on chunk borders straddle up to 8 voxels from 4 chunks).
2. Dual cells: every 2x2x2 block of voxel centres is a "cell", indexed by the
   lattice point between them — ``(33, 33, 33)`` cells, covering the chunk
   border on all sides.  A cell that contains both solid and air voxels gets
   one vertex, placed at the **centroid of the midpoints of its sign-changing
   edges** (the classic surface-nets vertex rule; with binary voxels every
   crossing is at an edge midpoint).
3. Faces: for every *exposed voxel face* (solid voxel, air across the face —
   the exact same exposure mask as the blocky mesher in ``meshing.py``), emit
   the quad connecting the 4 dual-cell vertices around that voxel-pair edge.
4. Each quad is emitted as **two independent flat triangles** (6 vertices per
   face, nothing shared/smoothed), each with its own normal and a subtle
   normal-based "facet accent" shade multiplied into the baked light so
   adjacent non-coplanar triangles read as distinct facets even though the
   scene renders with lighting off (texture x vertex-colour pipeline).

Why the exposure mask is unchanged
----------------------------------
Surface-nets quads correspond 1:1 to solid/air voxel-pair transitions, which
are exactly the faces the culled-face mesher emits.  This keeps the
``light_sampler`` contract identical (one sample per exposed face) and makes
"faceted face_count == blocky face_count" a testable invariant.

Seam guarantee (cross-chunk)
----------------------------
A dual cell on a chunk border is computed by *both* chunks from the same
world voxels (via the padded shell), so both produce byte-identical vertex
positions — no cracks, regardless of meshing order.  The caller must supply
all 26 neighbour material arrays for correct borders; ``ChunkManager``
fulfils this with loaded chunks or the deterministic ``generate_chunk``
baseline (identical either way unless the neighbour was brush-edited, and
brush edits dirty border neighbours — see ``apply_brush``).

Units: positions in world **meters**, Z-up, voxel = 0.5 m.  Fully vectorised
numpy — no per-voxel/per-face Python loops (Hard Rule 4); the only loops are
over the 12 cube edges, 26 neighbour offsets, and 6 face directions.

Docs: docs/systems/world.terrain.md
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from fire_engine.world.terrain.generation import MATERIAL_DIRT
from fire_engine.world.terrain.meshing import (
    _FACE_DIRS,
    _UV_TILE_M,
    WORLD_FLOOR_SOLID,
    MeshArrays,
)

__all__ = ["NEIGHBOR_OFFSETS_26", "build_mesh_faceted"]

# All 26 neighbour chunk offsets (face + edge + corner).  Dual cells on a
# chunk border straddle voxels from up to 8 chunks, so faceted meshing needs
# the full shell, not just the 6 face neighbours the blocky mesher uses.
NEIGHBOR_OFFSETS_26: tuple[tuple[int, int, int], ...] = tuple(
    (o[0], o[1], o[2]) for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)
)

# The 12 edges of a 2x2x2 cell, as pairs of corner offsets in {0,1}^3.
_CUBE_EDGES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = tuple(
    ((a[0], a[1], a[2]), (b[0], b[1], b[2]))
    for a in itertools.product((0, 1), repeat=3)
    for b in itertools.product((0, 1), repeat=3)
    if sum(abs(a[i] - b[i]) for i in range(3)) == 1 and a < b
)

# Per-direction dual-quad cell deltas for the two non-axis dims (in ascending
# axis order: X faces -> (y, z), Y faces -> (x, z), Z faces -> (x, y)).
# These mirror meshing.py's _FACE_TEMPLATES corner order exactly, so winding
# is CCW seen from outside (the default surface-nets vertex sits ON the voxel
# corner lattice, where the dual quad degenerates to the primal cube face).
_QUAD_DELTAS: dict[tuple[int, int, int], tuple[tuple[int, int], ...]] = {
    (1, 0, 0): ((0, 0), (1, 0), (1, 1), (0, 1)),
    (-1, 0, 0): ((1, 0), (0, 0), (0, 1), (1, 1)),
    (0, 1, 0): ((1, 0), (0, 0), (0, 1), (1, 1)),
    (0, -1, 0): ((0, 0), (1, 0), (1, 1), (0, 1)),
    (0, 0, 1): ((0, 0), (1, 0), (1, 1), (0, 1)),
    (0, 0, -1): ((0, 1), (1, 1), (1, 0), (0, 0)),
}

# Fixed facet-accent direction (normalized below).  This is NOT the sun — the
# real sun light arrives via light_sampler.  It is an art-direction constant
# that differentiates facet orientations in the lighting-off pipeline (see
# DECISIONS.md 2026-06-11).  Roughly "high noon, slightly south-east".
_FACET_ACCENT_DIR = np.array([0.35, 0.25, 0.90], dtype=np.float32)
_FACET_ACCENT_DIR /= np.linalg.norm(_FACET_ACCENT_DIR)


def _pad_slices(offset: int, n: int) -> tuple[slice, slice]:
    """(destination slice in the padded array, source slice in the neighbour)."""
    if offset == -1:
        return slice(0, 1), slice(n - 1, n)
    if offset == 0:
        return slice(1, n + 1), slice(0, n)
    return slice(n + 1, n + 2), slice(0, 1)


def _build_padded_materials(
    materials: np.ndarray,
    neighbor_materials: dict[tuple[int, int, int], np.ndarray | str] | None,
    n: int,
) -> np.ndarray:
    """
    Build the ``(n+2,)*3 uint8`` padded materials array from 26 neighbours.

    Each entry of ``neighbor_materials`` maps an offset in
    :data:`NEIGHBOR_OFFSETS_26` to a ``uint8 (n,n,n)`` materials array, the
    :data:`~fire_engine.world.terrain.meshing.WORLD_FLOOR_SOLID` sentinel (fill that
    shell region solid dirt — used at the world floor), or ``None``/absent
    (fill air).  Missing entries pad AIR, so the world edge is open.
    """
    pad = np.zeros((n + 2,) * 3, dtype=np.uint8)
    pad[1 : n + 1, 1 : n + 1, 1 : n + 1] = materials
    ns = neighbor_materials or {}
    for off in NEIGHBOR_OFFSETS_26:
        nb = ns.get(off)
        if nb is None:
            continue
        (dx, sx), (dy, sy), (dz, sz) = (
            _pad_slices(off[0], n),
            _pad_slices(off[1], n),
            _pad_slices(off[2], n),
        )
        if nb is WORLD_FLOOR_SOLID:
            pad[dx, dy, dz] = np.uint8(MATERIAL_DIRT)
        else:
            pad[dx, dy, dz] = np.asarray(nb, dtype=np.uint8)[sx, sy, sz]
    return pad


def _cell_vertices(solid_pad: np.ndarray, n: int) -> np.ndarray:
    """
    Surface-nets vertex positions for every dual cell, in cell-local coords.

    Parameters
    ----------
    solid_pad : numpy.ndarray
        ``bool (n+2,)*3`` padded solidity.
    n : int
        Chunk edge in voxels (32).

    Returns
    -------
    numpy.ndarray
        ``float32 (n+1, n+1, n+1, 3)`` — for each cell, the centroid of the
        midpoints of its sign-changing edges, in the cell's local unit cube
        (``[0,1]^3``, corner 0 = the cell's min voxel centre).  Cells with no
        sign change ("inactive") hold zeros; they are never referenced,
        because every emitted quad only touches cells adjacent to a crossing.
    """
    m = n + 1  # cells per axis
    corner = {
        c: solid_pad[c[0] : c[0] + m, c[1] : c[1] + m, c[2] : c[2] + m]
        for c in itertools.product((0, 1), repeat=3)
    }
    pos_sum = np.zeros((m, m, m, 3), dtype=np.float32)
    count = np.zeros((m, m, m), dtype=np.float32)
    for a, b in _CUBE_EDGES:
        cross = corner[a] != corner[b]  # (m,m,m) bool
        mid = (np.asarray(a, np.float32) + np.asarray(b, np.float32)) * 0.5
        count += cross
        pos_sum += cross[..., None] * mid
    return pos_sum / np.maximum(count, 1.0)[..., None]


def build_mesh_faceted(
    chunk: Any,
    neighbor_materials: dict[tuple[int, int, int], np.ndarray | str] | None = None,
    light_sampler: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    shade_strength: float = 0.25,
) -> MeshArrays:
    """
    Build a flat-shaded surface-nets mesh for ``chunk`` (the faceted look).

    Parameters
    ----------
    chunk : Chunk
        The chunk to mesh (reads ``materials``, ``world_origin``, voxel size).
    neighbor_materials : dict | None, optional
        Maps each offset in :data:`NEIGHBOR_OFFSETS_26` (all 26 face/edge/
        corner neighbours) to one of:

        - ``uint8 (32,32,32)`` materials array of that neighbour chunk, OR
        - :data:`~fire_engine.world.terrain.meshing.WORLD_FLOOR_SOLID` -> pad that
          shell region solid dirt (world floor), OR
        - ``None`` / key absent -> pad air (open world edge).

        ``None`` for the whole dict pads everything air (isolated fixtures).
        ``ChunkManager._neighbor_materials`` supplies the real thing.
    light_sampler : Callable | None, optional
        Identical contract to ``build_mesh``: takes face-centre world
        positions ``float32 (F, 3)`` meters (here F = exposed voxel faces ==
        the blocky mesher's face count; centres are the *deformed* quad
        centroids) and returns per-face light ``float32 (F,)`` in [0, 1].
        ``None`` -> full-bright.
    shade_strength : float, default 0.25
        Strength of the normal-based facet accent in ``[0, 1]``: vertex grey
        = ``light * ((1 - s) + s * clamp(normal . accent_dir, 0, 1))``.
        ``0`` disables the accent (pure baked light, facets only visible via
        silhouette/texture).  Comes from ``config.facet_shade_strength``.

    Returns
    -------
    MeshArrays
        With ``verts_per_face = 6`` (two independent flat triangles per
        exposed voxel face) and ``face_materials`` set (``uint8 (F,)``, the
        material id of each face's solid voxel — drives per-material
        textures in ``world/geometry_bridge.py``).

    Determinism
    -----------
    Pure function of (chunk materials, neighbour materials, light_sampler,
    shade_strength).  No randomness.

    Example
    -------
    >>> from fire_engine.world.terrain.chunk import Chunk
    >>> c = Chunk((0, 0, 0))
    >>> c.materials[5, 5, 5] = 1            # one solid voxel in air
    >>> mesh = build_mesh_faceted(c)
    >>> mesh.face_count, mesh.tri_count, mesh.vertex_count
    (6, 12, 36)

    Docs: docs/systems/world.terrain.md
    """
    n = chunk.materials.shape[0]
    vs = float(chunk._voxel_size)
    origin = chunk.world_origin.to_numpy().astype(np.float32)  # (3,)

    pad = _build_padded_materials(chunk.materials, neighbor_materials, n)
    solid_pad = pad > 0
    interior = solid_pad[1 : n + 1, 1 : n + 1, 1 : n + 1]

    vert_local = _cell_vertices(solid_pad, n)  # (n+1, n+1, n+1, 3)

    # World position of every cell vertex.  Cell (ci,cj,ck) spans voxel
    # centres (ci-1, ci) per axis; its local origin (corner 0) is the centre
    # of local voxel ci-1, i.e. world origin + (ci - 0.5) * vs.
    m = n + 1
    grid = np.stack(
        np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij"),
        axis=-1,
    ).astype(np.float32)
    verts_world = origin + (grid - 0.5 + vert_local) * vs  # (m,m,m,3)

    quad_blocks: list[np.ndarray] = []  # (f,4,3) world-space quad corners
    dir_blocks: list[np.ndarray] = []  # (f,3) face direction (fallback normal)
    mat_blocks: list[np.ndarray] = []  # (f,) uint8 face material

    for d in _FACE_DIRS:
        dx, dy, dz = d
        nb = solid_pad[1 + dx : 1 + dx + n, 1 + dy : 1 + dy + n, 1 + dz : 1 + dz + n]
        face_mask = interior & ~nb
        if not face_mask.any():
            continue
        vx, vy, vz = np.nonzero(face_mask)
        f = vx.shape[0]

        axis = 0 if dx != 0 else (1 if dy != 0 else 2)
        others = tuple(a for a in (0, 1, 2) if a != axis)

        base = np.stack([vx, vy, vz], axis=1)  # (f,3) int cell coords
        if d[axis] > 0:
            base[:, axis] += 1
        cells = np.repeat(base[:, None, :], 4, axis=1)  # (f,4,3)
        for k, (da, db) in enumerate(_QUAD_DELTAS[d]):
            cells[:, k, others[0]] += da
            cells[:, k, others[1]] += db

        quad_blocks.append(verts_world[cells[..., 0], cells[..., 1], cells[..., 2]])
        dir_blocks.append(np.broadcast_to(np.asarray(d, np.float32), (f, 3)).copy())
        mat_blocks.append(chunk.materials[vx, vy, vz])

    if not quad_blocks:
        return MeshArrays(
            positions=np.zeros((0, 3), np.float32),
            normals=np.zeros((0, 3), np.float32),
            uvs=np.zeros((0, 2), np.float32),
            colors=np.zeros((0, 4), np.float32),
            indices=np.zeros((0,), np.uint32),
            face_materials=np.zeros((0,), np.uint8),
            verts_per_face=6,
        )

    quads = np.concatenate(quad_blocks, axis=0)  # (F,4,3)
    face_dirs = np.concatenate(dir_blocks, axis=0)  # (F,3)
    face_mats = np.concatenate(mat_blocks, axis=0).astype(np.uint8)  # (F,)
    F = quads.shape[0]

    # --- Triangles: (0,1,2) and (0,2,3), 6 independent vertices per face ---
    tris = quads[:, [[0, 1, 2], [0, 2, 3]], :]  # (F,2,3,3)
    tris = tris.reshape(F * 2, 3, 3)
    positions = tris.reshape(F * 6, 3).astype(np.float32)

    # --- Flat per-triangle normals (degenerate tris fall back to face dir) --
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    tri_n = np.cross(e1, e2)  # (2F,3)
    nlen = np.linalg.norm(tri_n, axis=1, keepdims=True)
    fallback = np.repeat(face_dirs, 2, axis=0)  # (2F,3)
    tri_n = np.where(nlen > 1e-12, tri_n / np.maximum(nlen, 1e-12), fallback)
    normals = np.repeat(tri_n, 3, axis=0).astype(np.float32)  # (6F,3)

    # --- Planar UVs by dominant quad-normal axis (world meters / tile) ------
    quad_n = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    dom = np.argmax(np.abs(quad_n), axis=1)  # (F,)
    u_axis = np.where(dom == 0, 1, 0)  # X-dom -> (y,z)
    v_axis = np.where(dom == 2, 1, 2)  # Z-dom -> (x,y)
    u_axis_v = np.repeat(u_axis, 6)
    v_axis_v = np.repeat(v_axis, 6)
    rows = np.arange(positions.shape[0])
    uvs = np.stack(
        [positions[rows, u_axis_v] / _UV_TILE_M, positions[rows, v_axis_v] / _UV_TILE_M],
        axis=1,
    ).astype(np.float32)

    # --- Baked light (per face) x facet accent (per triangle) --------------
    face_centers = quads.mean(axis=1).astype(np.float32)  # (F,3)
    if light_sampler is not None:
        light = np.asarray(light_sampler(face_centers), dtype=np.float32)
        light = np.clip(light, 0.0, 1.0)
    else:
        light = np.ones((F,), dtype=np.float32)
    s = float(np.clip(shade_strength, 0.0, 1.0))
    ndl = np.clip(tri_n @ _FACET_ACCENT_DIR, 0.0, 1.0)  # (2F,)
    shade = (1.0 - s) + s * ndl
    grey = np.repeat(light, 6) * np.repeat(shade, 3)  # (6F,)
    colors = np.empty((positions.shape[0], 4), np.float32)
    colors[:, 0] = grey
    colors[:, 1] = grey
    colors[:, 2] = grey
    # Alpha carries the face material id (id / 255) so the GPU terrain shader
    # can pick a per-material, world-space procedural palette per fragment
    # (see world/shaders/terrain.frag).  Terrain is opaque with no transparency
    # attrib, so the fixed-function fallback never reads this alpha — safe to
    # repurpose.  6 verts per face, matching the 6-vertex triangle expansion.
    colors[:, 3] = np.repeat(face_mats, 6).astype(np.float32) / 255.0

    indices = np.arange(positions.shape[0], dtype=np.uint32)

    return MeshArrays(
        positions=positions,
        normals=normals,
        uvs=uvs,
        colors=colors,
        indices=indices,
        face_materials=face_mats,
        verts_per_face=6,
    )
