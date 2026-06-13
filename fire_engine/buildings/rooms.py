"""
Room auto-detection from wall topology (planar half-edge minimal cycles).

A storey's walls form a plane graph; the bounded faces of that graph are its
rooms.  This module extracts those faces with the classic *next-edge-clockwise*
half-edge traversal:

1. **Tessellate** every wall to its centerline polyline (arcs become chords at
   the same density meshing uses, so a wall's room polygon and its mesh agree).
2. **Snap** polyline vertices onto a ``snap_eps_m`` grid so walls that were
   authored to meet at a shared endpoint collapse onto one graph node.
3. **Build** the undirected plane graph (one node per snapped point, one edge
   per polyline segment, duplicates merged) and split each edge into two
   *darts* (directed half-edges).
4. **Sort** the darts leaving each node by heading (``atan2``).
5. **Trace** faces: from a dart ``u→v`` the next dart in the face is the one
   immediately *clockwise* from the reverse dart ``v→u`` around ``v``.  This
   walks every bounded face counter-clockwise (positive signed area) and the
   single unbounded outer face clockwise.  Keep the positive-area faces.

Dangling edges (an open "L", a stub wall) get walked out-and-back in a
zero-area face and are dropped by the area threshold, so an unenclosed layout
yields no rooms.

**v1 limitation (documented in docs/systems/buildings.md):** walls must meet at
*endpoints*.  A wall whose endpoint lands partway along another wall's span (a
T-junction) does not split that wall, so the two do not share a graph node and
the face will not close — author the long wall as two spans sharing the
junction vertex.

The face-trace and per-node angle sort are bounded Python loops over the graph
darts (a storey has dozens of wall segments, not thousands), which is why they
are not vectorized (Hard Rule 4 — flagged here intentionally); the geometry
inside each step is numpy.

This module imports only numpy + the model layer — no panda3d, no RNG.
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.buildings.model import Wall

__all__ = ["detect_room_polygons"]

# Faces smaller than this in plan area are treated as degenerate (dangling
# edges traversed out-and-back, float slivers) and discarded.  Real rooms are
# square meters; 1 mm² is comfortably below anything authored.
_MIN_ROOM_AREA_M2 = 1e-6


def _signed_area(poly: np.ndarray) -> float:
    """Shoelace signed area of an open polygon (CCW positive), m²."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def detect_room_polygons(
    walls: list[Wall],
    *,
    snap_eps_m: float,
    arc_segments_per_quarter: int,
) -> list[np.ndarray]:
    """
    Extract enclosed-room polygons from a storey's walls.

    Parameters
    ----------
    walls : list[Wall]
        The storey's walls (straight or arc).  Direction is irrelevant — each
        wall contributes an undirected edge per tessellated segment.
    snap_eps_m : float
        Endpoint-snap tolerance in meters (``Config.building_snap_eps_m``).
        Vertices rounded to the same ``snap_eps_m`` grid cell merge into one
        graph node.
    arc_segments_per_quarter : int
        Arc tessellation density (``Config.building_arc_segments_per_quarter``);
        must match meshing so a room polygon and its wall geometry coincide.

    Returns
    -------
    list[np.ndarray]
        One ``float64 (N, 2)`` CCW polygon per detected room, ordered as the
        face trace found them.  Empty when nothing encloses.
    """
    if snap_eps_m <= 0.0:
        raise ValueError("snap_eps_m must be positive")
    inv_eps = 1.0 / snap_eps_m

    # ---- nodes: snap-quantized point -> node index -------------------------
    node_xy: list[tuple[float, float]] = []
    node_of_key: dict[tuple[int, int], int] = {}

    def _node(px: float, py: float) -> int:
        key = (int(round(px * inv_eps)), int(round(py * inv_eps)))
        idx = node_of_key.get(key)
        if idx is None:
            idx = len(node_xy)
            node_of_key[key] = idx
            node_xy.append((px, py))
        return idx

    # ---- undirected edges from every tessellated wall segment --------------
    edge_set: set[tuple[int, int]] = set()
    for wall in walls:
        poly = wall.tessellate(arc_segments_per_quarter)
        ids = [_node(float(px), float(py)) for px, py in poly]
        for u, v in zip(ids[:-1], ids[1:]):
            if u == v:
                continue  # zero-length segment after snapping
            edge_set.add((u, v) if u < v else (v, u))

    if not edge_set:
        return []

    # ---- darts (directed half-edges) + per-node outgoing adjacency ---------
    # dart id is an index into `dart`; twin(2k) == 2k+1 and vice versa.
    dart: list[tuple[int, int]] = []
    out_darts: list[list[int]] = [[] for _ in range(len(node_xy))]
    for u, v in edge_set:
        d0 = len(dart)
        dart.append((u, v))
        dart.append((v, u))
        out_darts[u].append(d0)
        out_darts[v].append(d0 + 1)

    def _twin(d: int) -> int:
        return d ^ 1

    def _heading(d: int) -> float:
        u, v = dart[d]
        return math.atan2(node_xy[v][1] - node_xy[u][1],
                          node_xy[v][0] - node_xy[u][0])

    # Sort each node's outgoing darts CCW by heading, and remember each dart's
    # position in its source node's order so we can step clockwise in O(1).
    pos_in_order: dict[int, int] = {}
    for node, ds in enumerate(out_darts):
        ds.sort(key=_heading)
        for i, d in enumerate(ds):
            pos_in_order[d] = i

    def _next_in_face(d: int) -> int:
        # Arrive at v along u->v; leave along the dart immediately clockwise
        # from the reverse dart v->u around v (keeps the face on the left).
        t = _twin(d)
        v = dart[d][1]
        ring = out_darts[v]
        return ring[(pos_in_order[t] - 1) % len(ring)]

    # ---- trace every face once; keep the CCW (positive-area) bounded ones --
    visited = [False] * len(dart)
    rooms: list[np.ndarray] = []
    for start in range(len(dart)):
        if visited[start]:
            continue
        cycle: list[int] = []
        d = start
        # Bounded loop: at most one pass over all darts (each visited once).
        for _ in range(len(dart) + 1):
            if visited[d]:
                break
            visited[d] = True
            cycle.append(dart[d][0])  # source node of this dart
            d = _next_in_face(d)
            if d == start:
                break
        if len(cycle) < 3:
            continue
        poly = np.array([node_xy[n] for n in cycle], dtype=np.float64)
        if _signed_area(poly) > _MIN_ROOM_AREA_M2:
            rooms.append(poly)
    return rooms
