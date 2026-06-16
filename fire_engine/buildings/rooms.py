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

import itertools
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


def _build_snap_graph(
    walls: list[Wall],
    inv_eps: float,
    arc_segments_per_quarter: int,
) -> tuple[list[tuple[float, float]], set[tuple[int, int]]]:
    """
    Tessellate walls, snap vertices to grid, and build an undirected edge set.

    Returns ``(node_xy, edge_set)`` where ``node_xy[i]`` is the float position
    of snap-node ``i`` and ``edge_set`` holds canonical ``(u, v)`` pairs with
    ``u < v``.
    """
    node_xy: list[tuple[float, float]] = []
    node_of_key: dict[tuple[int, int], int] = {}

    def _snap_node(px: float, py: float) -> int:
        key = (round(px * inv_eps), round(py * inv_eps))
        idx = node_of_key.get(key)
        if idx is None:
            idx = len(node_xy)
            node_of_key[key] = idx
            node_xy.append((px, py))
        return idx

    edge_set: set[tuple[int, int]] = set()
    for wall in walls:
        poly = wall.tessellate(arc_segments_per_quarter)
        ids = [_snap_node(float(px), float(py)) for px, py in poly]
        for u, v in itertools.pairwise(ids):
            if u == v:
                continue  # zero-length segment after snapping
            edge_set.add((u, v) if u < v else (v, u))

    return node_xy, edge_set


def _build_half_edges(
    node_xy: list[tuple[float, float]],
    edge_set: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[list[int]], dict[int, int]]:
    """
    Build directed half-edges (darts) from an undirected edge set.

    Each undirected edge ``(u, v)`` produces dart ``2k = u→v`` and twin
    ``2k+1 = v→u``.  Returns ``(dart, out_darts, pos_in_order)`` where
    ``out_darts[node]`` lists the darts leaving that node sorted CCW by
    heading, and ``pos_in_order[dart_id]`` is the dart's position in that
    sorted list.
    """
    dart: list[tuple[int, int]] = []
    out_darts: list[list[int]] = [[] for _ in range(len(node_xy))]
    for u, v in edge_set:
        d0 = len(dart)
        dart.append((u, v))
        dart.append((v, u))
        out_darts[u].append(d0)
        out_darts[v].append(d0 + 1)

    def _heading(d: int) -> float:
        u, v = dart[d]
        return math.atan2(node_xy[v][1] - node_xy[u][1], node_xy[v][0] - node_xy[u][0])

    pos_in_order: dict[int, int] = {}
    for _, ds in enumerate(out_darts):
        ds.sort(key=_heading)
        pos_in_order.update({d: i for i, d in enumerate(ds)})

    return dart, out_darts, pos_in_order


def _trace_face(
    start: int,
    dart: list[tuple[int, int]],
    out_darts: list[list[int]],
    pos_in_order: dict[int, int],
    visited: list[bool],
) -> list[int]:
    """
    Walk one face starting at dart ``start``; mark darts visited.

    Returns the ordered list of source nodes visited (the face boundary).
    Uses the *clockwise-from-twin* rule: from dart ``d = u→v`` the next dart
    in the face is the dart leaving ``v`` that is immediately clockwise from
    the reverse dart ``v→u`` (twin of ``d``) around ``v``.
    """
    cycle: list[int] = []
    d = start
    for _ in range(len(dart) + 1):
        if visited[d]:
            break
        visited[d] = True
        cycle.append(dart[d][0])
        t = d ^ 1  # twin
        v = dart[d][1]
        ring = out_darts[v]
        d = ring[(pos_in_order[t] - 1) % len(ring)]
        if d == start:
            break
    return cycle


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

    node_xy, edge_set = _build_snap_graph(walls, inv_eps, arc_segments_per_quarter)
    if not edge_set:
        return []

    dart, out_darts, pos_in_order = _build_half_edges(node_xy, edge_set)

    visited = [False] * len(dart)
    rooms: list[np.ndarray] = []
    for start in range(len(dart)):
        if visited[start]:
            continue
        cycle = _trace_face(start, dart, out_darts, pos_in_order, visited)
        if len(cycle) < 3:
            continue
        poly = np.array([node_xy[n] for n in cycle], dtype=np.float64)
        if _signed_area(poly) > _MIN_ROOM_AREA_M2:
            rooms.append(poly)
    return rooms
