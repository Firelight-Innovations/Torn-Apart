"""
Shared Panda3D helpers for mote render components (DustMoteComponent and
LeafLitterComponent): billboard quad geometry and procedural texture loading.

Factored here to avoid a circular import between mote_renderer and
_impl.leaf_litter (both need the helpers; neither can import the other).

The quad geometry is delegated to ``render._impl.quad.build_unit_quad``
(shared with the rain renderer) and re-exported under the legacy name so
call sites need no changes.

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import Any

from panda3d.core import Geom

from fire_engine.render._impl.quad import build_unit_quad

__all__ = ["build_quad_geom", "mote_texture"]


def build_quad_geom() -> Geom:
    """
    Build the shared unit billboard quad: corners at xy in {-1, +1}, z=0, UV 0-1.

    The vertex shaders offset these corners (in view space for dust, after a
    tumble rotation for leaves), so one tiny 4-vertex / 2-triangle Geom is the
    base for every instance — a fixed handful of vertices, never a per-particle
    array.

    Delegates to ``render._impl.quad.build_unit_quad`` with the legacy
    ``"mote_quad"`` profiler label preserved.

    Docs: docs/systems/render.vegetation._impl.md
    """
    return build_unit_quad("mote_quad")


def mote_texture(name: str) -> Any:
    """The procedural ``name`` texture as a Panda3D texture (linear-filtered
    so the soft dust falloff / leaf edges don't look chunky billboarded).

    Docs: docs/systems/render.vegetation._impl.md
    """
    from panda3d.core import SamplerState

    from fire_engine.procedural import get as get_procedural
    from fire_engine.render.bridges.texture_bridge import to_panda_texture

    tex = to_panda_texture(get_procedural(name))
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex
