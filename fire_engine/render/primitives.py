"""
world/primitives.py — small in-code primitive geometry (no assets required).

Shared helpers for the handful of places that need a basic cube or sphere
NodePath: the dev overlay's spawned props, the wind debug ball, and the
authored-scene visuals (``scene_visuals.py``). Panda3D imports are allowed
here (world/ package, ARCHITECTURE §3).

Sizes are normalised so a GameObject with ``local_scale == 1`` renders as a
1 m primitive (engine unit = meters, Z-up).
"""

from __future__ import annotations

import math

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)

__all__ = ["build_sphere_geom", "make_sphere_node", "load_cube_model", "CUBE_MODEL_SCALE"]

# Panda3D's stock "models/misc/rgbCube" spans -1..1 per axis; scaling it by 0.5
# yields a 1 m cube at unit GameObject scale. Multiply this into any per-axis
# scale you apply to the returned model.
CUBE_MODEL_SCALE: float = 0.5


def build_sphere_geom(radius_m: float, segments: int = 16, rings: int = 8) -> Geom:
    """
    Build a small UV-sphere Geom centred at the origin — no asset, in-code.

    A standard latitude/longitude sphere: ``rings`` horizontal bands ×
    ``segments`` meridians.  The vertex/triangle counts are tiny and fixed
    (``(rings+1)*(segments+1)`` verts), so the nested build loops are bounded
    setup, not a per-element hot path.

    Parameters
    ----------
    radius_m : float
        Sphere radius in meters.
    segments : int, default 16
        Meridian count (longitude divisions).
    rings : int, default 8
        Latitude band count.

    Returns
    -------
    panda3d.core.Geom

    Example
    -------
        node = GeomNode("ball")
        node.add_geom(build_sphere_geom(0.5))   # a 1 m diameter sphere
    """
    fmt = GeomVertexFormat.get_v3n3()  # position + normal
    vdata = GeomVertexData("sphere", fmt, Geom.UH_static)
    vdata.set_num_rows((rings + 1) * (segments + 1))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    tris = GeomTriangles(Geom.UH_static)

    for r in range(rings + 1):
        theta = math.pi * r / rings  # 0..pi (north → south pole)
        st, ct = math.sin(theta), math.cos(theta)
        for s in range(segments + 1):
            phi = 2.0 * math.pi * s / segments
            sp, cp = math.sin(phi), math.cos(phi)
            nx, ny, nz = st * cp, st * sp, ct
            vw.add_data3(nx * radius_m, ny * radius_m, nz * radius_m)
            nw.add_data3(nx, ny, nz)

    row = segments + 1
    for r in range(rings):
        for s in range(segments):
            a = r * row + s
            b = a + row
            tris.add_vertices(a, b, a + 1)
            tris.add_vertices(a + 1, b, b + 1)

    geom = Geom(vdata)
    geom.add_primitive(tris)
    return geom


def make_sphere_node(name: str, radius_m: float = 0.5) -> NodePath:
    """A free-standing NodePath holding a UV-sphere (1 m diameter by default).

    Parameters
    ----------
    name : str
        NodePath name (shows in scene-graph dumps).
    radius_m : float, default 0.5
        Sphere radius in meters at unit scale.
    """
    node = GeomNode(name)
    node.add_geom(build_sphere_geom(radius_m))
    return NodePath(node)


def load_cube_model(loader) -> NodePath | None:
    """
    Load Panda3D's stock cube model with the ``box`` fallback.

    The caller owns reparenting and scaling; remember the model spans -1..1, so
    apply :data:`CUBE_MODEL_SCALE` (×0.5) for a 1 m cube at unit object scale.

    Parameters
    ----------
    loader : the ShowBase ``loader``.

    Returns
    -------
    NodePath | None — ``None`` when neither stock model is available.
    """
    model = loader.load_model("models/misc/rgbCube")
    if model is None or model.is_empty():
        # Fallback: a plain box model name some Panda3D builds ship instead.
        model = loader.load_model("box")
    if model is None or model.is_empty():
        return None
    return model
