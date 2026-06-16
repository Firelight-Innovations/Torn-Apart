"""
Ear-clipping triangulation for slab polygons (no external dependency).

Slabs (floors, foundation, roof) are simple plan-space polygons of at most a
few dozen vertices, so the classic O(n²) ear-clipping algorithm is ample and
adds no dependency.  The outer sweep is a bounded Python loop over polygon
vertices (dozens, not thousands — flagged per Hard Rule 4); the ear test
(point-in-triangle for every remaining vertex) is vectorized numpy.

The single public function returns triangle index triples into the *input*
polygon's vertices so callers can place those vertices at any z without
re-running the triangulation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["triangulate_polygon"]


def _signed_area(poly: np.ndarray) -> float:
    x: np.ndarray = poly[:, 0]
    y: np.ndarray = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _points_in_triangle(pts: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Boolean mask of which ``pts`` lie strictly inside CCW triangle a,b,c."""
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    d1 = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
    d2 = (c[0] - b[0]) * (pts[:, 1] - b[1]) - (c[1] - b[1]) * (pts[:, 0] - b[0])
    d3 = (a[0] - c[0]) * (pts[:, 1] - c[1]) - (a[1] - c[1]) * (pts[:, 0] - c[0])
    eps = 1e-12
    mask: np.ndarray = (d1 > eps) & (d2 > eps) & (d3 > eps)
    return mask


def triangulate_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    Triangulate a simple polygon by ear clipping.

    Parameters
    ----------
    polygon : np.ndarray
        ``(N, 2)`` simple (non-self-intersecting) plan-space polygon, not
        closed.  Winding may be CW or CCW; the result triangles are emitted
        CCW so a +z-facing slab top is front-facing.

    Returns
    -------
    np.ndarray
        ``uint32 (T, 3)`` triangle indices into ``polygon`` (``T = N - 2`` for
        a clean simple polygon; fewer if degenerate vertices are skipped).
    """
    poly = np.asarray(polygon, dtype=np.float64)
    n = poly.shape[0]
    if n < 3:
        return np.empty((0, 3), dtype=np.uint32)
    # Work CCW so a positive cross product means a convex corner.
    order = list(range(n - 1, -1, -1)) if _signed_area(poly) < 0.0 else list(range(n))

    tris: list[tuple[int, int, int]] = []
    idx = order[:]  # remaining vertex indices (CCW)
    # Bounded outer loop: each pass clips at least one ear (or breaks).
    guard = 0
    while len(idx) > 3 and guard < n * n:
        guard += 1
        m = len(idx)
        clipped = False
        for i in range(m):
            i0, i1, i2 = idx[(i - 1) % m], idx[i], idx[(i + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:
                continue  # reflex or collinear — not an ear tip
            others = [j for j in idx if j not in (i0, i1, i2)]
            if np.any(_points_in_triangle(poly[others], a, b, c)):
                continue  # another vertex inside — not an ear
            tris.append((i0, i1, i2))
            del idx[i]
            clipped = True
            break
        if not clipped:
            break  # numerically stuck — fan the rest
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    elif len(idx) > 3:
        # Degenerate fallback: triangle fan from the first remaining vertex.
        tris.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    return np.array(tris, dtype=np.uint32)
