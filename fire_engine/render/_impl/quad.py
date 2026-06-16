"""
render/_impl/quad — shared unit billboard-quad geometry builder and additive
particle node setup helper.

Provides two helpers extracted to eliminate duplication across instanced-particle
renderers (rain streaks, dust motes, leaf litter):

* ``build_unit_quad`` — the shared 4-vertex / 2-triangle Geom
* ``setup_additive_instanced_node`` — apply additive blending, disable depth
  write, set an infinite bounding box, and call ``set_final`` on the pair
  ``(node, geom_node)``.

Docs: docs/systems/render.md
"""

from __future__ import annotations

from panda3d.core import (
    BoundingBox,
    ColorBlendAttrib,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LPoint3,
    NodePath,
    TransparencyAttrib,
)

__all__ = ["build_unit_quad", "setup_additive_instanced_node"]


def build_unit_quad(name: str = "unit_quad") -> Geom:
    """
    Build a unit billboard-quad Geom: corners at xy ∈ {-1, +1}, z=0, UV 0–1.

    The returned ``Geom`` is ``UH_static``; it is intended to be added once
    to a ``GeomNode`` and drawn N times via ``set_instance_count``.  The
    vertex shader is responsible for displacing the corners into screen/world
    space (view-aligned billboarding, tumble rotation, rain-streak tilt, etc.).

    Parameters
    ----------
    name : str
        ``GeomVertexData`` label used in Panda3D's profiler output.
        Defaults to ``"unit_quad"``; pass a renderer-specific label
        (``"rain_quad"``, ``"mote_quad"``) to preserve the original profiler
        names when migrating callers.

    Returns
    -------
    Geom
        A 4-vertex / 2-triangle static Geom, position + UV only
        (``GeomVertexFormat.get_v3t2()``).

    Example
    -------
    ::

        geom_node = GeomNode("my_particles")
        geom_node.add_geom(build_unit_quad("my_quad"))
        node = parent.attach_new_node(geom_node)
        node.set_instance_count(1000)
    """
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData(name, fmt, Geom.UH_static)
    vdata.set_num_rows(4)
    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")
    corners = (
        (-1.0, -1.0, 0.0, 0.0),
        (1.0, -1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-1.0, 1.0, 0.0, 1.0),
    )
    for x, y, u, v in corners:
        vw.add_data3(x, y, 0.0)
        tw.add_data2(u, v)
    tris = GeomTriangles(Geom.UH_static)
    tris.add_vertices(0, 1, 2)
    tris.add_vertices(0, 2, 3)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    return geom


def setup_additive_instanced_node(node: NodePath, geom_node: GeomNode) -> None:
    """
    Configure *node* / *geom_node* for additive-blended GPU-instanced particles.

    Applies:

    * ``TransparencyAttrib.M_none`` + ``ColorBlendAttrib`` additive blend
      (incoming-alpha × src + 1 × dst) — the standard "glow" composite.
    * ``set_depth_write(False)`` — additive is order-independent; no Z writes.
    * ``set_bin("fixed", 0)`` — renders in the fixed bin, after opaque geometry.
    * ``set_two_sided(True)`` — both faces visible (quads face the camera from
      any angle).
    * Infinite ``BoundingBox`` on *geom_node* + ``set_final(True)`` — instances
      are positioned in the shader; the base Geom's origin bounds would cull
      every off-origin particle.

    Call site: ``DustMoteComponent.start`` and ``build_particles`` in
    ``render/sky/_impl/rain_build``.

    Parameters
    ----------
    node : NodePath
        The instanced NodePath (has the shader + instance count).
    geom_node : GeomNode
        The underlying GeomNode added to *node*.
    """
    node.set_transparency(TransparencyAttrib.M_none)
    node.set_attrib(
        ColorBlendAttrib.make(
            ColorBlendAttrib.M_add, ColorBlendAttrib.O_incoming_alpha, ColorBlendAttrib.O_one
        )
    )
    node.set_depth_write(False)
    node.set_bin("fixed", 0)
    node.set_two_sided(True)

    big = 1.0e9
    geom_node.set_bounds(BoundingBox(LPoint3(-big, -big, -big), LPoint3(big, big, big)))
    geom_node.set_final(True)
