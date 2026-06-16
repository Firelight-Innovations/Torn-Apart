"""
buildings/_impl/seams.py — corner gap/seam closing for wall junctions.

Iteration-2 seam fixing (see ``docs/systems/buildings.md`` Roadmap): walls are
meshed butt-to-butt along their own centerlines, which leaves a wedge gap at
shared corners until the general cross-wall miter lands (Iteration 3).  This
module closes that gap cheaply and watertightly: for every plan node where two
or more wall ends meet, :func:`corner_filler_polys` returns a **filler-post
polygon** — the convex hull of those walls' ±thickness/2 offset corner points —
which the mesher extrudes over the shared wall band.  The hull never extends
past any wall's own outer face, so the corner reads as solid without a true
miter.

Pure geometry: returns polygons (no panda3d, no mesh accumulator), so the
module has no import cycle with ``buildings/meshing.py``.  Bounded Python loops
over a storey's walls/junctions (dozens — Hard Rule 4); the per-vertex math is
numpy.

Docs: docs/systems/buildings._impl.md
"""

from __future__ import annotations

import numpy as np

from fire_engine.buildings._impl.storey import Storey
from fire_engine.buildings.model import _convex_hull
from fire_engine.buildings.types import Wall

__all__ = ["corner_filler_polys"]

_EPS = 1e-9


def _wall_end_offsets(
    wall: Wall, qpq: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The wall's two centerline endpoints and the ±thickness/2 offset corner
    points at each, in plan space.

    Returns ``(a, a_off, b, b_off)`` where ``a``/``b`` are the endpoint plan
    points and ``a_off``/``b_off`` are ``float64 (2, 2)`` arrays holding the
    left/right offset corners at that endpoint (offset along the end segment's
    left normal) — the corner points a junction filler must span.

    Docs: docs/systems/buildings._impl.md
    """
    pts = wall.tessellate(qpq)
    t_half = wall.thickness_m / 2.0
    d0 = pts[1] - pts[0]
    dn = pts[-1] - pts[-2]
    d0 = d0 / max(float(np.hypot(d0[0], d0[1])), _EPS)
    dn = dn / max(float(np.hypot(dn[0], dn[1])), _EPS)
    n0 = np.array([-d0[1], d0[0]])  # left normal at endpoint a
    nn = np.array([-dn[1], dn[0]])  # left normal at endpoint b
    a, b = pts[0], pts[-1]
    a_off = np.array([a + n0 * t_half, a - n0 * t_half], dtype=np.float64)
    b_off = np.array([b + nn * t_half, b - nn * t_half], dtype=np.float64)
    return a, a_off, b, b_off


def corner_filler_polys(
    storey: Storey, qpq: int, snap_eps: float
) -> list[tuple[np.ndarray, float]]:
    """
    Filler-post polygons for every shared wall corner on ``storey``.

    For each plan node where two or more wall ends coincide (within
    ``snap_eps`` meters) the filler is the convex hull of those walls' offset
    corner points (plus the node), which exactly spans the butt-joint wedge.
    Lone wall ends (a single end at a node) yield no filler — their butt cap
    already closes the wall.

    Parameters
    ----------
    storey : Storey
        The storey whose walls are joined.
    qpq : int
        Arc tessellation density (``Config.building_arc_segments_per_quarter``).
    snap_eps : float
        Endpoint coincidence tolerance in meters (``Config.building_snap_eps_m``).

    Returns
    -------
    list[tuple[numpy.ndarray, float]]
        ``(hull_polygon (N, 2) float64 CCW, band_top_m)`` per junction; the
        mesher extrudes each over ``[z_bottom, z_bottom + band_top_m]`` where
        ``z_bottom`` is the storey's floor-slab top and ``band_top_m`` is the
        shortest incident wall band (so the post never overtops a low wall).

    Docs: docs/systems/buildings._impl.md
    """
    ends: list[tuple[np.ndarray, np.ndarray, float]] = []
    for wall in storey.walls:
        band = wall.height_m if wall.height_m is not None else storey.height_m - storey.slab_m
        a, a_off, b, b_off = _wall_end_offsets(wall, qpq)
        ends.append((a, a_off, float(band)))
        ends.append((b, b_off, float(band)))
    grid = max(snap_eps, _EPS)
    groups: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for node, off, band in ends:
        key = (round(float(node[0]) / grid), round(float(node[1]) / grid))
        groups.setdefault(key, []).append((node, off, band))
    fillers: list[tuple[np.ndarray, float]] = []
    for items in groups.values():
        if len(items) < 2:
            continue  # free end — its butt cap already closes the wall
        node = items[0][0][None, :]
        pts = np.concatenate([it[1] for it in items] + [node], axis=0)
        hull = _convex_hull(pts)
        if hull.shape[0] < 3:
            continue  # collinear pass-through — nothing to fill
        fillers.append((hull, min(it[2] for it in items)))
    return fillers
