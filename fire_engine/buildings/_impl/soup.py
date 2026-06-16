"""
buildings/_impl/soup.py — outward-facing triangle-soup accumulator.

Shared low-level mesh builder for the buildings package: :class:`Soup` collects
outward-facing quads/triangles (auto-flipping winding to a supplied outward
normal) and bakes a :class:`~fire_engine.world.terrain.meshing.MeshArrays`
(building-LOCAL positions, flat-white colours, per-meter planar UVs).  Walls
(``buildings/meshing.py``) and pitched roofs (``buildings/_impl/roofs.py``)
both build into one ``Soup`` so the whole building uploads as a single geom.

Two solid primitives live here because more than one producer needs them:

- :meth:`Soup.add_slab` — a flat horizontal slab between ``z0`` and ``z1``
  (floors, foundation, flat roofs, corner-filler posts).
- :meth:`Soup.add_prism` — a **sloped** solid: a planar top polygon dropped
  vertically by a constant thickness (one pitched-roof panel, given real depth
  at the eave).

Vectorized throughout (Hard Rule 4): the only Python loops are over walls /
roof panels (a handful); every per-vertex array is numpy.

Docs: docs/systems/buildings._impl.md
"""

from __future__ import annotations

import numpy as np

from fire_engine.buildings.triangulate import triangulate_polygon
from fire_engine.world.terrain.meshing import MeshArrays

__all__ = ["Soup"]

_EPS = 1e-9


def _normalize(v: np.ndarray) -> np.ndarray:
    ln = np.linalg.norm(v, axis=-1, keepdims=True)
    ln[ln < _EPS] = 1.0
    result: np.ndarray = v / ln
    return result


class Soup:
    """
    Accumulates outward-facing triangles, then bakes a ``MeshArrays``.

    Add geometry with :meth:`add_quads` / :meth:`add_tris` (low level) or the
    :meth:`add_slab` / :meth:`add_prism` solid helpers, then call
    :meth:`build`.  Every face is handed an explicit outward normal and its
    winding is flipped to match, so back-face culling is correct regardless of
    corner order.

    Docs: docs/systems/buildings._impl.md
    """

    def __init__(self) -> None:
        self._pos: list[np.ndarray] = []
        self._nrm: list[np.ndarray] = []
        self._uv: list[np.ndarray] = []
        self._mat: list[np.ndarray] = []  # per-face (per-triangle) material id

    # -- quads: (Q,4,3) corners + (Q,3) outward normals --------------------
    def add_quads(self, corners: np.ndarray, normals: np.ndarray, material: int = 0) -> None:
        """Add ``(Q, 4, 3)`` quad corners with ``(Q, 3)`` outward normals.

        ``material`` tags both emitted triangles with a surface-material id
        (see :class:`~fire_engine.buildings.enums.SurfaceMaterial`).

        Docs: docs/systems/buildings._impl.md
        """
        q = corners.shape[0]
        if q == 0:
            return
        corners = corners.astype(np.float64, copy=True)
        n = _normalize(normals.astype(np.float64))
        # Flip winding where the geometric normal opposes the desired one.
        geo = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
        flip = np.sum(geo * n, axis=1) < 0.0
        corners[flip] = corners[flip][:, [0, 3, 2, 1], :]
        # Per-corner UVs in meters along the two quad edges.
        le1 = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
        le2 = np.linalg.norm(corners[:, 3] - corners[:, 0], axis=1)
        uv = np.zeros((q, 4, 2), dtype=np.float64)
        uv[:, 1, 0] = le1
        uv[:, 2, 0] = le1
        uv[:, 2, 1] = le2
        uv[:, 3, 1] = le2
        # Two triangles (0,1,2) (0,2,3).
        tri = corners[:, [0, 1, 2, 0, 2, 3], :].reshape(-1, 3)
        tuv = uv[:, [0, 1, 2, 0, 2, 3], :].reshape(-1, 2)
        self._pos.append(tri)
        self._uv.append(tuv)
        self._nrm.append(np.repeat(n, 6, axis=0))
        self._mat.append(np.full(q * 2, int(material), dtype=np.uint8))

    # -- triangles: (T,3,3) verts + (T,3) outward normals ------------------
    def add_tris(self, verts: np.ndarray, normals: np.ndarray, material: int = 0) -> None:
        """Add ``(T, 3, 3)`` triangle verts with ``(T, 3)`` outward normals.

        ``material`` tags each triangle with a surface-material id.

        Docs: docs/systems/buildings._impl.md
        """
        t = verts.shape[0]
        if t == 0:
            return
        verts = verts.astype(np.float64, copy=True)
        n = _normalize(normals.astype(np.float64))
        geo = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
        flip = np.sum(geo * n, axis=1) < 0.0
        verts[flip] = verts[flip][:, [0, 2, 1], :]
        self._pos.append(verts.reshape(-1, 3))
        self._uv.append(verts[:, :, :2].reshape(-1, 2))  # planar (x,y) UVs
        self._nrm.append(np.repeat(n, 3, axis=0))
        self._mat.append(np.full(t, int(material), dtype=np.uint8))

    # -- solids ------------------------------------------------------------
    def add_slab(self, polygon: np.ndarray, z0: float, z1: float, material: int = 0) -> None:
        """Flat slab: top + bottom faces (ear-clipped) and perimeter side quads.

        Every face is tagged ``material``.

        Docs: docs/systems/buildings._impl.md
        """
        poly = np.asarray(polygon, dtype=np.float64)
        if poly.shape[0] < 3:
            return
        tris = triangulate_polygon(poly)  # CCW (T,3) indices
        if tris.shape[0]:
            flat = poly[tris]  # (T,3,2)
            top = np.dstack([flat, np.full(flat.shape[:2], z1)])
            self.add_tris(top, np.tile([0.0, 0.0, 1.0], (tris.shape[0], 1)), material)
            bot = np.dstack([flat, np.full(flat.shape[:2], z0)])
            self.add_tris(bot, np.tile([0.0, 0.0, -1.0], (tris.shape[0], 1)), material)
        # Side quads (one per polygon edge); outward normal = right of CCW edge.
        p0 = poly
        p1 = np.roll(poly, -1, axis=0)
        e = p1 - p0
        out = _normalize(np.stack([e[:, 1], -e[:, 0]], axis=1))
        n = poly.shape[0]
        corners = np.empty((n, 4, 3), dtype=np.float64)
        corners[:, 0, :2] = p0
        corners[:, 0, 2] = z0
        corners[:, 1, :2] = p1
        corners[:, 1, 2] = z0
        corners[:, 2, :2] = p1
        corners[:, 2, 2] = z1
        corners[:, 3, :2] = p0
        corners[:, 3, 2] = z1
        normals = np.concatenate([out, np.zeros((n, 1))], axis=1)
        self.add_quads(corners, normals, material)

    def add_prism(self, top: np.ndarray, drop: float, material: int = 0) -> None:
        """
        Sloped solid: a planar ``top`` polygon ``(N, 3)`` and a copy dropped
        ``drop`` meters in z, closed with perimeter side quads.

        ``top`` must be a graph over the XY plane (true for any roof pitch
        below vertical); it is triangulated by its XY projection and the real
        3-D plane normal (oriented +z up) is used for the top/bottom faces so
        the slope lights correctly.  ``drop <= 0`` emits just the single top
        sheet (a zero-thickness panel).  Every face is tagged ``material``.

        Docs: docs/systems/buildings._impl.md
        """
        top = np.asarray(top, dtype=np.float64)
        n = top.shape[0]
        if n < 3:
            return
        tris = triangulate_polygon(top[:, :2])  # (T,3) indices into top
        if tris.shape[0]:
            t0 = top[tris[0]]
            nrm = np.cross(t0[1] - t0[0], t0[2] - t0[0])
            if nrm[2] < 0.0:
                nrm = -nrm
            self.add_tris(top[tris], np.tile(nrm, (tris.shape[0], 1)), material)
            if drop > _EPS:
                bottom = top.copy()
                bottom[:, 2] -= drop
                self.add_tris(bottom[tris], np.tile(-nrm, (tris.shape[0], 1)), material)
        if drop <= _EPS:
            return
        # Perimeter side quads connecting the top edge to the dropped edge.
        b_top = top
        b_bot = top.copy()
        b_bot[:, 2] -= drop
        p1_top = np.roll(b_top, -1, axis=0)
        p1_bot = np.roll(b_bot, -1, axis=0)
        e = (p1_top - b_top)[:, :2]
        out = _normalize(np.stack([e[:, 1], -e[:, 0]], axis=1))
        corners = np.stack([b_bot, p1_bot, p1_top, b_top], axis=1)  # (n,4,3)
        normals = np.concatenate([out, np.zeros((n, 1))], axis=1)
        self.add_quads(corners, normals, material)

    def build(self) -> MeshArrays:
        """Bake the accumulated faces into a ``MeshArrays`` (flat-white colours).

        ``face_materials`` is a ``uint8`` per-face array when more than one
        material id was used (so the renderer can split the geom per material),
        else ``None`` (single-texture fast path).

        Docs: docs/systems/buildings._impl.md
        """
        if not self._pos:
            z = np.zeros
            return MeshArrays(
                z((0, 3), np.float32),
                z((0, 3), np.float32),
                z((0, 2), np.float32),
                z((0, 4), np.float32),
                z((0,), np.uint32),
                None,
                3,
            )
        pos = np.concatenate(self._pos, axis=0).astype(np.float32)
        nrm = np.concatenate(self._nrm, axis=0).astype(np.float32)
        uv = np.concatenate(self._uv, axis=0).astype(np.float32)
        col = np.ones((pos.shape[0], 4), dtype=np.float32)
        idx = np.arange(pos.shape[0], dtype=np.uint32)
        mats = np.concatenate(self._mat, axis=0)
        face_materials = mats if np.any(mats != 0) else None
        return MeshArrays(pos, nrm, uv, col, idx, face_materials, 3)
