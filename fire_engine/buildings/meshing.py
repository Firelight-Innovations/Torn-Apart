"""
buildings/meshing.py — turn a :class:`~fire_engine.buildings.model.Building`
into renderable triangle geometry.

The output is a :class:`fire_engine.world.terrain.meshing.MeshArrays` — the same flat
(non-vertex-shared) triangle-soup contract terrain uses — so the World layer's
``world/geometry_bridge.py`` uploads buildings with the existing bulk path.
Two differences from terrain meshes, both deliberate and documented:

- **Positions are building-LOCAL meters, not world space.**  The renderer puts
  the building's `position`/`rotation` on the scene node; moving or rotating a
  building is a transform write, never a remesh (so `building.vert` must derive
  the world position from `p3d_ModelMatrix`).
- **`verts_per_face = 3`** — the soup mixes wall panels, reveals, caps and slab
  triangles, so "faces" are individual triangles; `colors` is a flat white
  ``(1,1,1,1)`` (the building shader ignores vertex colour, unlike terrain
  which packs a material id into alpha) and `face_materials` is ``None``.

Meshing strategy is **partition, never CSG** (design decision D5):

- *Walls* — the centerline is tessellated (arcs → chords at
  `building_arc_segments_per_quarter`), opening edges are inserted as extra
  centerline vertices, and a per-vertex miter offset of ±thickness/2 builds the
  outer/inner faces.  The face is partitioned into a grid of panels by the
  union of segment break-points (s) and opening sill/head levels (z); panels
  whose centre falls inside an opening rect are dropped, and each opening emits
  reveal faces (jambs + head + sill).  Top and end caps close the prism.
- *Slabs* (floor / foundation / roof) — ear-clipped top + bottom faces plus
  perimeter side quads (`triangulate.py`).

Winding is made outward-facing by handing every quad/triangle an explicit
outward normal and flipping the vertex order to match — so back-face culling
is correct regardless of how the corners were listed.

Vectorized throughout (Hard Rule 4): the only Python loops iterate walls,
storeys and openings (a building has dozens, not thousands); every per-vertex
array is numpy.  Determinism: no RNG — identical buildings mesh byte-identically.
"""

from __future__ import annotations

import numpy as np

from fire_engine.buildings.model import Building, Storey, Wall, _convex_hull
from fire_engine.buildings.triangulate import triangulate_polygon
from fire_engine.core.config import Config
from fire_engine.world.terrain.meshing import MeshArrays

__all__ = ["mesh_building", "mesh_slab", "mesh_wall"]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Triangle-soup accumulator
# ---------------------------------------------------------------------------


class _Soup:
    """Accumulates outward-facing triangles, then bakes a MeshArrays."""

    def __init__(self) -> None:
        self._pos: list[np.ndarray] = []
        self._nrm: list[np.ndarray] = []
        self._uv: list[np.ndarray] = []

    # -- quads: (Q,4,3) corners + (Q,3) outward normals --------------------
    def add_quads(self, corners: np.ndarray, normals: np.ndarray) -> None:
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

    # -- triangles: (T,3,3) verts + (T,3) outward normals ------------------
    def add_tris(self, verts: np.ndarray, normals: np.ndarray) -> None:
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

    def build(self) -> MeshArrays:
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
        return MeshArrays(pos, nrm, uv, col, idx, None, 3)


def _normalize(v: np.ndarray) -> np.ndarray:
    ln = np.linalg.norm(v, axis=-1, keepdims=True)
    ln[ln < _EPS] = 1.0
    result: np.ndarray = v / ln
    return result


# ---------------------------------------------------------------------------
# Slabs
# ---------------------------------------------------------------------------


def _add_slab(soup: _Soup, polygon: np.ndarray, z0: float, z1: float) -> None:
    """Top + bottom faces (ear-clipped) and perimeter side quads."""
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.shape[0] < 3:
        return
    tris = triangulate_polygon(poly)  # CCW (T,3) indices
    if tris.shape[0]:
        flat = poly[tris]  # (T,3,2)
        top = np.dstack([flat, np.full(flat.shape[:2], z1)])
        soup.add_tris(top, np.tile([0.0, 0.0, 1.0], (tris.shape[0], 1)))
        bot = np.dstack([flat, np.full(flat.shape[:2], z0)])
        soup.add_tris(bot, np.tile([0.0, 0.0, -1.0], (tris.shape[0], 1)))
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
    soup.add_quads(corners, normals)


def mesh_slab(polygon: np.ndarray, z0: float, z1: float) -> MeshArrays:
    """Mesh a single horizontal slab between ``z0`` and ``z1`` (meters)."""
    soup = _Soup()
    _add_slab(soup, polygon, float(z0), float(z1))
    return soup.build()


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------


def _insert_arclengths(
    pts: np.ndarray, cum: np.ndarray, svals: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Insert points at the given arclengths into a centerline polyline."""
    out_pts = list(pts)
    out_cum = list(cum)
    for s in svals:
        # Skip values already (near) coincident with an existing vertex.
        if np.any(np.abs(np.asarray(out_cum) - s) <= 1e-7):
            continue
        seg = int(np.searchsorted(cum, s) - 1)
        seg = max(0, min(seg, len(cum) - 2))
        t = (s - cum[seg]) / max(cum[seg + 1] - cum[seg], _EPS)
        p = pts[seg] + t * (pts[seg + 1] - pts[seg])
        # Insert keeping arrays sorted by arclength.
        ins = int(np.searchsorted(out_cum, s))
        out_pts.insert(ins, p)
        out_cum.insert(ins, s)
    return np.array(out_pts, dtype=np.float64), np.array(out_cum, dtype=np.float64)


def mesh_wall(
    wall: Wall, z_bottom: float, z_top: float, arc_segments_per_quarter: int
) -> MeshArrays:
    """
    Mesh one wall as a thick extruded prism with openings cut out.

    Parameters
    ----------
    wall : Wall
        The wall span (straight or arc) with its openings.
    z_bottom, z_top : float
        Building-local z of the wall base (floor-slab top) and the wall top
        (meters).  Openings' ``sill_m``/``head_m`` are measured from
        ``z_bottom``.
    arc_segments_per_quarter : int
        Arc tessellation density (``Config.building_arc_segments_per_quarter``).
    """
    soup = _Soup()
    _add_wall(soup, wall, float(z_bottom), float(z_top), int(arc_segments_per_quarter))
    return soup.build()


def _add_wall(soup: _Soup, wall: Wall, zb: float, zt: float, qpq: int) -> None:
    pts = wall.tessellate(qpq)  # (P,2)
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    length = float(cum[-1])

    # Opening break-points along arclength, clamped inside the wall.
    s_extra: list[float] = []
    for op in wall.openings:
        s_extra.append(min(max(op.offset_m, 0.0), length))
        s_extra.append(min(max(op.offset_m + op.width_m, 0.0), length))
    cl, cum = _insert_arclengths(pts, cum, sorted(set(s_extra)))
    m = cl.shape[0]

    # Per-vertex miter offset (±thickness/2) so the offset faces stay closed.
    t_half = wall.thickness_m / 2.0
    tseg = cl[1:] - cl[:-1]
    tlen = np.hypot(tseg[:, 0], tseg[:, 1])[:, None]
    tlen[tlen < _EPS] = 1.0
    that = tseg / tlen
    segn = np.stack([-that[:, 1], that[:, 0]], axis=1)  # left normals (S-1,2)
    offv = np.empty((m, 2), dtype=np.float64)
    offv[0] = segn[0] * t_half
    offv[-1] = segn[-1] * t_half
    if m > 2:
        s_sum = segn[:-1] + segn[1:]
        denom = np.sum(s_sum * s_sum, axis=1, keepdims=True)
        denom[denom < _EPS] = 1.0
        offv[1:-1] = s_sum * wall.thickness_m / denom
    front = cl + offv  # outward/left face
    back = cl - offv  # inward/right face

    # Vertical break levels: wall band + each opening's sill/head.
    z_extra = [zb, zt]
    for op in wall.openings:
        z_extra.append(zb + op.sill_m)
        z_extra.append(zb + op.head_m)
    zlev = np.unique(np.clip(np.array(z_extra), zb, zt))
    nz = zlev.shape[0]

    # ---- front / back panel grid, minus opening cells --------------------
    # Hole mask over (s-cell, z-cell) by cell-centre membership in an opening.
    s_mid = 0.5 * (cum[:-1] + cum[1:])  # (m-1,)
    z_mid = 0.5 * (zlev[:-1] + zlev[1:])  # (nz-1,)
    hole = np.zeros((m - 1, nz - 1), dtype=bool)
    for op in wall.openings:
        in_s = (s_mid > op.offset_m + _EPS) & (s_mid < op.offset_m + op.width_m - _EPS)
        in_z = (z_mid > zb + op.sill_m + _EPS) & (z_mid < zb + op.head_m - _EPS)
        hole |= in_s[:, None] & in_z[None, :]

    soup.add_quads(*_panel_grid(front, offv, s_mid, zlev, hole, outward=True))
    soup.add_quads(*_panel_grid(back, -offv, s_mid, zlev, hole, outward=False))

    # ---- reveals around each opening -------------------------------------
    for op in wall.openings:
        ka = int(np.argmin(np.abs(cum - op.offset_m)))
        kb = int(np.argmin(np.abs(cum - (op.offset_m + op.width_m))))
        zs, zh = zb + op.sill_m, zb + op.head_m
        # Near + far jambs (outward normal = wall tangent away from opening).
        soup.add_quads(
            *_vert_quad(front[ka], back[ka], zs, zh, -that[max(ka - 1, 0)] if ka > 0 else -that[0])
        )
        soup.add_quads(*_vert_quad(front[kb], back[kb], zs, zh, that[min(kb, len(that) - 1)]))
        # Head (underside, faces down) and sill (top, faces up if present).
        soup.add_quads(*_horiz_quad(front[ka], back[ka], front[kb], back[kb], zh, up=False))
        if op.sill_m > _EPS:
            soup.add_quads(*_horiz_quad(front[ka], back[ka], front[kb], back[kb], zs, up=True))

    # ---- top cap + end caps ----------------------------------------------
    soup.add_quads(*_cap_strip(front, back, zt, up=True))
    soup.add_quads(*_vert_quad(front[0], back[0], zb, zt, -that[0]))
    soup.add_quads(*_vert_quad(front[-1], back[-1], zb, zt, that[-1]))


def _panel_grid(
    face: np.ndarray,
    offv: np.ndarray,
    s_mid: np.ndarray,
    zlev: np.ndarray,
    hole: np.ndarray,
    outward: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the solid panel quads of one wall face (front or back)."""
    keep = ~hole  # (m-1, nz-1)
    ki, ji = np.nonzero(keep)
    q = ki.shape[0]
    corners = np.empty((q, 4, 3), dtype=np.float64)
    # Corners in order: bottom-left, bottom-right, top-right, top-left.
    corners[:, 0, :2] = face[ki]
    corners[:, 0, 2] = zlev[ji]
    corners[:, 1, :2] = face[ki + 1]
    corners[:, 1, 2] = zlev[ji]
    corners[:, 2, :2] = face[ki + 1]
    corners[:, 2, 2] = zlev[ji + 1]
    corners[:, 3, :2] = face[ki]
    corners[:, 3, 2] = zlev[ji + 1]
    # Outward normal per panel = averaged offset direction of its two edges.
    ndir = _normalize(offv[ki] + offv[ki + 1])
    normals = np.concatenate([ndir, np.zeros((q, 1))], axis=1)
    return corners, normals


def _vert_quad(
    p_front: np.ndarray, p_back: np.ndarray, z0: float, z1: float, out_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One vertical quad spanning thickness (front↔back) and [z0,z1]."""
    c = np.array(
        [
            [
                [p_front[0], p_front[1], z0],
                [p_back[0], p_back[1], z0],
                [p_back[0], p_back[1], z1],
                [p_front[0], p_front[1], z1],
            ]
        ],
        dtype=np.float64,
    )
    n = np.array([[out_xy[0], out_xy[1], 0.0]], dtype=np.float64)
    return c, n


def _horiz_quad(
    fa: np.ndarray, ba: np.ndarray, fb: np.ndarray, bb: np.ndarray, z: float, up: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal quad across the opening span at height ``z`` (reveal ledge)."""
    c = np.array(
        [[[fa[0], fa[1], z], [ba[0], ba[1], z], [bb[0], bb[1], z], [fb[0], fb[1], z]]],
        dtype=np.float64,
    )
    n = np.array([[0.0, 0.0, 1.0 if up else -1.0]], dtype=np.float64)
    return c, n


def _cap_strip(
    front: np.ndarray, back: np.ndarray, z: float, up: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Top (or bottom) cap: one quad per centerline segment at height ``z``."""
    m = front.shape[0]
    corners = np.empty((m - 1, 4, 3), dtype=np.float64)
    corners[:, 0, :2] = front[:-1]
    corners[:, 1, :2] = back[:-1]
    corners[:, 2, :2] = back[1:]
    corners[:, 3, :2] = front[1:]
    corners[:, :, 2] = z
    n = np.tile([0.0, 0.0, 1.0 if up else -1.0], (m - 1, 1))
    return corners, n


# ---------------------------------------------------------------------------
# Whole building
# ---------------------------------------------------------------------------


def _storey_footprint(building: Building, storey: Storey) -> np.ndarray:
    """Convex-hull footprint of a storey's walls for its floor slab."""
    if storey.walls:
        xy = np.concatenate([w.tessellate(8) for w in storey.walls], axis=0)
        return _convex_hull(xy)
    if building.foundation is not None:
        return building.foundation.polygon
    return np.empty((0, 2), dtype=np.float64)


def mesh_building(building: Building, cfg: Config) -> MeshArrays:
    """
    Mesh an entire building into one building-local triangle soup.

    Parameters
    ----------
    building : Building
        The authored building.
    cfg : Config
        Engine config — supplies ``building_arc_segments_per_quarter``.

    Returns
    -------
    MeshArrays
        Building-LOCAL positions (the renderer applies the node transform);
        ``colors`` flat white, ``face_materials=None``, ``verts_per_face=3``.
    """
    qpq = int(cfg.building_arc_segments_per_quarter)
    soup = _Soup()
    for storey in building.storeys:
        base = building.storey_base_z(storey.index)
        z_floor0 = base
        z_floor1 = base + storey.slab_m
        _add_slab(soup, _storey_footprint(building, storey), z_floor0, z_floor1)
        for wall in storey.walls:
            band = wall.height_m if wall.height_m is not None else storey.height_m - storey.slab_m
            _add_wall(soup, wall, z_floor1, z_floor1 + band, qpq)
    if building.foundation is not None:
        _add_slab(soup, building.foundation.polygon, -building.foundation.depth_m, 0.0)
    if building.roof is not None:
        top = building.total_height_m
        _add_slab(soup, building.roof.polygon, top, top + building.roof.thickness_m)
    return soup.build()
