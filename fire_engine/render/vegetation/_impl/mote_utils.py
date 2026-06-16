"""
Shared Panda3D helpers for mote render components (DustMoteComponent and
LeafLitterComponent): billboard quad geometry and procedural texture loading.

Factored here to avoid a circular import between mote_renderer and
_impl.leaf_litter (both need the helpers; neither can import the other).

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import Any

from panda3d.core import (
    Geom,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
)

__all__ = ["build_quad_geom", "mote_texture"]


def build_quad_geom() -> Geom:
    """
    Build the shared unit billboard quad: corners at xy in {-1, +1}, z=0, UV 0-1.

    The vertex shaders offset these corners (in view space for dust, after a
    tumble rotation for leaves), so one tiny 4-vertex / 2-triangle Geom is the
    base for every instance — a fixed handful of vertices, never a per-particle
    array.
    """
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("mote_quad", fmt, Geom.UH_static)
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


def mote_texture(name: str) -> Any:
    """The procedural ``name`` texture as a Panda3D texture (linear-filtered
    so the soft dust falloff / leaf edges don't look chunky billboarded)."""
    from panda3d.core import SamplerState

    from fire_engine.procedural import get as get_procedural
    from fire_engine.render.bridges.texture_bridge import to_panda_texture

    tex = to_panda_texture(get_procedural(name))
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex
