"""
terrain/meshing.py — Culled-face voxel mesher, fully vectorised.

Emits one quad (two triangles) per *exposed* voxel face — a face is exposed
when its voxel is solid and the neighbour across that face is air.  Flat
per-face normals (vertices are duplicated per face, never shared/smoothed):
this hard-edged look IS the retro art direction (ARCHITECTURE.md §5.5).

No per-face / per-vertex Python loops anywhere — every array is built with
numpy reshaping, tiling and broadcasting (Hard Rule 4).  The DDA raycaster in
``raycast.py`` is the only place a short loop is allowed.

Output: a :class:`MeshArrays` dataclass of numpy arrays handed to the World
layer's ``world/geometry_bridge.py`` for a single bulk upload per array.

Edge padding (the critical correctness detail)
----------------------------------------------
To decide whether a face on the chunk *boundary* is exposed we need the
neighbouring chunk's solidity.  We build a ``(34, 34, 34)`` padded solid array
(the 32³ interior plus one voxel of neighbour data on each side) and slice it
to compute face masks — never ``np.roll`` (roll wraps and leaks faces).

The 6 boundary slabs are filled from ``neighbor_solids`` when a neighbour is
present.  For an **absent** neighbour the pad is AIR (so the world edge is
visible/open) **except the −Z (bottom) world boundary, which pads SOLID** so the
map has no see-through floor.  "Bottom of the world" is determined by the
caller via the ``world_floor`` flag on the missing −Z neighbour entry (the
chunk manager passes ``world_floor=True`` when the chunk below is absent and
this chunk sits at/below the lowest streamed Z band — see chunk_manager.py).

Docs: docs/systems/world.terrain.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Sentinel used inside neighbor_solids to mean "absent neighbour, pad SOLID".
WORLD_FLOOR_SOLID = "world_floor_solid"

# Face directions: (dx, dy, dz). Order is stable and documented.
# +X, -X, +Y, -Y, +Z, -Z
_FACE_DIRS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

# UV tile size in meters: planar UVs repeat every _UV_TILE_M meters of world space.
_UV_TILE_M: float = 1.0


@dataclass
class MeshArrays:
    """
    CPU-side mesh data for one chunk — handed to ``world/geometry_bridge.py``.

    All arrays are contiguous numpy and describe a flat (non-vertex-shared)
    triangle list.  Two producers exist:

    - ``build_mesh`` (blocky): 4 vertices + 2 triangles per exposed face
      (``verts_per_face = 4``), ``face_materials = None``.
    - ``build_mesh_faceted`` (surface nets): 6 vertices = 2 *independent*
      flat triangles per exposed face (``verts_per_face = 6``), and
      ``face_materials`` set per face for material-split rendering.

    Attributes
    ----------
    positions : numpy.ndarray
        ``float32`` ``(N, 3)`` — vertex positions in **world meters** (Z-up).
    normals : numpy.ndarray
        ``float32`` ``(N, 3)`` — flat normal, identical for all vertices of a
        face (blocky) or of a triangle (faceted).  Never smoothed.
    uvs : numpy.ndarray
        ``float32`` ``(N, 2)`` — planar UVs from world coords mod 1 m tile.
    colors : numpy.ndarray
        ``float32`` ``(N, 4)`` — RGBA vertex colours in ``[0, 1]``.  Greyscale
        × baked light (see ``light_sampler``); alpha = 1.0.
    indices : numpy.ndarray
        ``uint32`` ``(M,)`` — triangle indices, two triangles per face,
        counter-clockwise when viewed from outside (front faces).
    face_materials : numpy.ndarray | None
        ``uint8`` ``(F,)`` — material id of each face's solid voxel, in face
        order.  ``None`` for the blocky mesher (single-texture rendering);
        set by the faceted mesher so ``world/geometry_bridge.py`` can split
        the chunk into one Geom per material (grass vs dirt textures).
    verts_per_face : int
        Vertices emitted per exposed face: 4 (blocky) or 6 (faceted).

    Counts (handy for tests / debug overlays)
    -----------------------------------------
    ``face_count``  = number of exposed faces
    ``vertex_count``= ``verts_per_face * face_count``
    ``tri_count``   = ``2 * face_count``

    Docs: docs/systems/world.terrain.md
    """

    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    colors: np.ndarray
    indices: np.ndarray
    face_materials: np.ndarray | None = None
    verts_per_face: int = 4

    @property
    def face_count(self) -> int:
        """Number of exposed faces emitted.

        Docs: docs/systems/world.terrain.md
        """
        return int(self.positions.shape[0]) // self.verts_per_face

    @property
    def vertex_count(self) -> int:
        """Number of vertices (``verts_per_face`` per face).

        Docs: docs/systems/world.terrain.md
        """
        return int(self.positions.shape[0])

    @property
    def tri_count(self) -> int:
        """Number of triangles (2 per face).

        Docs: docs/systems/world.terrain.md
        """
        return int(self.indices.shape[0]) // 3

    @property
    def is_empty(self) -> bool:
        """True when no faces were emitted (fully buried or empty chunk).

        Docs: docs/systems/world.terrain.md
        """
        return bool(self.positions.shape[0] == 0)


# ---------------------------------------------------------------------------
# Per-face geometry templates (computed once, module level).
# For each of the 6 directions we precompute the 4 corner offsets of the unit
# quad (in voxel units, relative to the voxel min-corner) and the normal.
# Winding is CCW seen from outside.
# ---------------------------------------------------------------------------


def _build_face_templates() -> dict[tuple[int, int, int], dict[str, np.ndarray]]:
    """
    Build the static per-direction quad corner offsets and normals.

    Returns a dict mapping direction → {'corners': (4,3) float32 voxel-space
    offsets, 'normal': (3,) float32}.  Computed once at import.
    """
    # Voxel occupies the unit cube [0,1]^3 in voxel units (later scaled by voxel_size).
    # Each face's 4 corners ordered CCW when viewed from outside (normal toward viewer).
    templates: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
    # +X face (x=1 plane), normal +X. CCW from +X looking toward -X.
    templates[(1, 0, 0)] = dict(
        corners=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], np.float32),
        normal=np.array([1, 0, 0], np.float32),
    )
    # -X face (x=0 plane), normal -X.
    templates[(-1, 0, 0)] = dict(
        corners=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], np.float32),
        normal=np.array([-1, 0, 0], np.float32),
    )
    # +Y face (y=1 plane), normal +Y.
    templates[(0, 1, 0)] = dict(
        corners=np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]], np.float32),
        normal=np.array([0, 1, 0], np.float32),
    )
    # -Y face (y=0 plane), normal -Y.
    templates[(0, -1, 0)] = dict(
        corners=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float32),
        normal=np.array([0, -1, 0], np.float32),
    )
    # +Z face (z=1 plane), normal +Z (top).
    templates[(0, 0, 1)] = dict(
        corners=np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32),
        normal=np.array([0, 0, 1], np.float32),
    )
    # -Z face (z=0 plane), normal -Z (bottom).
    templates[(0, 0, -1)] = dict(
        corners=np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]], np.float32),
        normal=np.array([0, 0, -1], np.float32),
    )
    return templates


_FACE_TEMPLATES = _build_face_templates()


def _fill_pad_slab(
    pad: np.ndarray,
    nb: np.ndarray | str | None,
    dst: tuple[Any, ...],
    src: tuple[Any, ...],
) -> None:
    """Fill one slab of *pad* from a neighbour entry (or sentinel / absent)."""
    if nb is WORLD_FLOOR_SOLID:
        pad[dst] = True
    elif nb is not None:
        assert isinstance(nb, np.ndarray)
        pad[dst] = nb[src]


def _build_padded_solid(
    solid: np.ndarray,
    neighbor_solids: dict[tuple[int, int, int], np.ndarray | str] | None,
    n: int,
) -> np.ndarray:
    """
    Build the ``(n+2, n+2, n+2)`` padded solidity array.

    The interior ``[1:n+1, 1:n+1, 1:n+1]`` is this chunk's solidity.  Each of
    the 6 one-voxel boundary slabs is filled with the *facing* slab of the
    neighbour chunk (the neighbour's voxels immediately across the shared face).

    Padding rule for absent neighbours
    ----------------------------------
    - Most edges: pad AIR (False) → the world edge is open/visible.
    - ``-Z`` world floor: pad SOLID (True) when the entry is the
      :data:`WORLD_FLOOR_SOLID` sentinel → no see-through floor.

    Parameters
    ----------
    solid : numpy.ndarray
        ``bool`` ``(n, n, n)`` solidity of this chunk, indexed ``[x, y, z]``.
    neighbor_solids : dict | None
        Maps direction ``(dx, dy, dz)`` (one of the 6 unit face dirs) to either
        a ``bool`` ``(n, n, n)`` solidity array of that neighbour chunk, the
        sentinel :data:`WORLD_FLOOR_SOLID`, or ``None``/absent (treated as air).
    n : int
        Chunk edge in voxels (32).
    """
    pad = np.zeros((n + 2, n + 2, n + 2), dtype=bool)
    pad[1 : n + 1, 1 : n + 1, 1 : n + 1] = solid

    ns: dict[tuple[int, int, int], np.ndarray | str] = neighbor_solids or {}
    s = slice(1, n + 1)

    # +X neighbour fills the x = n+1 slab using its x=0 slab.
    _fill_pad_slab(pad, ns.get((1, 0, 0)), (n + 1, s, s), (0, slice(None), slice(None)))
    # -X neighbour fills x = 0 slab using its x=n-1 slab.
    _fill_pad_slab(pad, ns.get((-1, 0, 0)), (0, s, s), (n - 1, slice(None), slice(None)))
    # +Y
    _fill_pad_slab(pad, ns.get((0, 1, 0)), (s, n + 1, s), (slice(None), 0, slice(None)))
    # -Y
    _fill_pad_slab(pad, ns.get((0, -1, 0)), (s, 0, s), (slice(None), n - 1, slice(None)))
    # +Z
    _fill_pad_slab(pad, ns.get((0, 0, 1)), (s, s, n + 1), (slice(None), slice(None), 0))
    # -Z (the bottom-of-world case): sentinel → SOLID pad.
    _fill_pad_slab(pad, ns.get((0, 0, -1)), (s, s, 0), (slice(None), slice(None), n - 1))

    return pad


def build_mesh(
    chunk: Any,
    neighbor_solids: dict[tuple[int, int, int], np.ndarray | str] | None = None,
    light_sampler: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MeshArrays:
    """
    Build a culled-face mesh for ``chunk``.

    Parameters
    ----------
    chunk : Chunk
        The chunk to mesh.  Reads ``chunk.materials`` / ``chunk.is_solid_mask``,
        ``chunk.world_origin`` and the voxel size.
    neighbor_solids : dict | None, optional
        Solidity of the 6 face-adjacent chunks for correct edge culling.  Maps
        direction ``(dx, dy, dz)`` ∈ the 6 unit face vectors to one of:

        - ``bool`` ``(32, 32, 32)`` solidity array of that neighbour chunk, OR
        - :data:`WORLD_FLOOR_SOLID` sentinel → pad that face SOLID
          (used for the −Z world floor so there is no see-through bottom), OR
        - ``None`` / key absent → pad that face AIR (open world edge).

        When ``None`` is passed for the whole dict, every face is padded air —
        useful for isolated single-chunk fixture tests.
    light_sampler : Callable[[np.ndarray], np.ndarray] | None, optional
        **Phase 4 lighting hook — exact contract:**
        a callable taking face-centre world positions ``float32 (F, 3)`` in
        **meters** (one row per exposed face, F = ``face_count``) and returning
        per-face light as ``float32 (F,)`` in the range **[0.0, 1.0]**
        (0 = black, 1 = full sun).  The value is multiplied into the greyscale
        base colour and written to every vertex of that face.  When ``None``
        (the default), all faces are full-bright (1.0) so the mesher is fully
        testable without lighting.

    Returns
    -------
    MeshArrays
        Vectorised mesh arrays.  Empty (zero-length arrays) when the chunk has
        no exposed faces.

    Determinism
    -----------
    Pure function of (chunk voxel data, neighbour solidity, light_sampler).
    No randomness.

    Example
    -------
    >>> from fire_engine.world.terrain.chunk import Chunk
    >>> import numpy as np
    >>> c = Chunk((0, 0, 0))
    >>> c.materials[5, 5, 5] = 1          # one solid voxel, surrounded by air
    >>> mesh = build_mesh(c)              # no neighbours → all edges open
    >>> mesh.face_count, mesh.tri_count, mesh.vertex_count
    (6, 12, 24)

    Docs: docs/systems/world.terrain.md
    """
    n = chunk.materials.shape[0]
    vs = float(chunk._voxel_size)
    origin = chunk.world_origin.to_numpy()  # (3,) float32, world meters

    solid = chunk.is_solid_mask()  # (n,n,n) bool [x,y,z]
    pad = _build_padded_solid(solid, neighbor_solids, n)  # (n+2,)*3
    interior = pad[1 : n + 1, 1 : n + 1, 1 : n + 1]  # == solid (view)

    # Collect per-direction face voxel coordinates.
    pos_blocks = []
    nrm_blocks = []
    uv_blocks = []
    facecenter_blocks = []  # world-space face centres for the light sampler

    for d in _FACE_DIRS:
        dx, dy, dz = d
        # neighbour solidity across this face, via padded slicing (NOT np.roll).
        # neighbour at (x+dx, y+dy, z+dz) lives in pad shifted by +1 + d.
        nb = pad[
            1 + dx : 1 + dx + n,
            1 + dy : 1 + dy + n,
            1 + dz : 1 + dz + n,
        ]
        face_mask = interior & ~nb  # (n,n,n) bool: exposed faces in this direction
        if not face_mask.any():
            continue

        # voxel indices (x,y,z) of every exposed face in this direction
        vx, vy, vz = np.nonzero(face_mask)  # each (F_d,)
        f = vx.shape[0]
        voxel_idx = np.stack([vx, vy, vz], axis=1).astype(np.float32)  # (F_d, 3)

        tpl = _FACE_TEMPLATES[d]
        corners = tpl["corners"]  # (4, 3) voxel-space corner offsets
        normal = tpl["normal"]  # (3,)

        # Vertex positions: (voxel_idx + corner) * voxel_size + origin
        # broadcast: (F_d,1,3) + (1,4,3) -> (F_d,4,3)
        vpos = (voxel_idx[:, None, :] + corners[None, :, :]) * vs + origin[None, None, :]
        vpos = vpos.reshape(f * 4, 3)  # (F_d*4, 3)

        # Normals: same flat normal for all 4 verts of every face.
        vnrm = np.broadcast_to(normal, (f, 4, 3)).reshape(f * 4, 3).copy()

        # Face centre world position (for the light sampler): voxel centre + half-step in d.
        center_voxel = voxel_idx + 0.5  # (F_d, 3) voxel centre in voxel units
        offset = np.array(d, dtype=np.float32) * 0.5  # half a voxel toward the face
        fcenter = (center_voxel + offset) * vs + origin  # (F_d, 3) world meters
        facecenter_blocks.append((d, fcenter))

        # Planar UVs from world coords mod tile size. Pick the 2 axes ⟂ to the normal.
        # For +-X: use (Y,Z); +-Y: use (X,Z); +-Z: use (X,Y).
        if dx != 0:
            u = vpos[:, 1]
            v = vpos[:, 2]
        elif dy != 0:
            u = vpos[:, 0]
            v = vpos[:, 2]
        else:
            u = vpos[:, 0]
            v = vpos[:, 1]
        vuv = np.stack([u / _UV_TILE_M, v / _UV_TILE_M], axis=1).astype(np.float32)

        pos_blocks.append(vpos.astype(np.float32))
        nrm_blocks.append(vnrm.astype(np.float32))
        uv_blocks.append(vuv)

    if not pos_blocks:
        # Fully buried / empty: return empty arrays.
        return MeshArrays(
            positions=np.zeros((0, 3), np.float32),
            normals=np.zeros((0, 3), np.float32),
            uvs=np.zeros((0, 2), np.float32),
            colors=np.zeros((0, 4), np.float32),
            indices=np.zeros((0,), np.uint32),
        )

    positions = np.concatenate(pos_blocks, axis=0)  # (N, 3)
    normals = np.concatenate(nrm_blocks, axis=0)  # (N, 3)
    uvs = np.concatenate(uv_blocks, axis=0)  # (N, 2)
    total_faces = positions.shape[0] // 4

    # --- Light baking ---------------------------------------------------
    # Assemble per-face centre array in the SAME face order as positions.
    face_centers = np.concatenate([fc for _, fc in facecenter_blocks], axis=0)  # (F, 3)
    if light_sampler is not None:
        light = np.asarray(light_sampler(face_centers), dtype=np.float32)  # (F,)
        light = np.clip(light, 0.0, 1.0)
    else:
        light = np.ones((total_faces,), dtype=np.float32)
    # Expand per-face light to per-vertex (4 verts/face) and build greyscale RGBA.
    light_v = np.repeat(light, 4)  # (N,)
    colors = np.empty((positions.shape[0], 4), np.float32)
    colors[:, 0] = light_v
    colors[:, 1] = light_v
    colors[:, 2] = light_v
    colors[:, 3] = 1.0

    # --- Indices: two CCW triangles per quad (0,1,2) (0,2,3) ------------
    # Vectorised: base = 4*face_index, add the template offsets.
    face_base = (np.arange(total_faces, dtype=np.uint32) * 4)[:, None]  # (F,1)
    quad = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)[None, :]  # (1,6)
    indices = (face_base + quad).reshape(-1)  # (F*6,)

    return MeshArrays(
        positions=positions,
        normals=normals,
        uvs=uvs,
        colors=colors,
        indices=indices,
    )
