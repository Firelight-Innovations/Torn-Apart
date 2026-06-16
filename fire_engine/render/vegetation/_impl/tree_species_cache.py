"""
Species resource cache helpers for TreeRendererComponent, extracted from
tree_renderer.py to satisfy the ≤500-line module limit.

Each function takes the component instance as ``self_obj`` and handles
uploading / caching per-species Panda3D resources (Geoms and Textures).
Results are memoised on the instance, so the upload cost is paid once even
when many zone volumes share a species.

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import Geom, Texture

from fire_engine.render.vegetation.flora_renderer import _build_cross_geom

if TYPE_CHECKING:
    from fire_engine.render.vegetation.tree_renderer import TreeRendererComponent

__all__ = ["impostor_geom", "mesh_geom", "species_atlas", "species_impostor"]


def mesh_geom(self_obj: TreeRendererComponent, name: str, variant: int, vs: Any) -> Geom:
    """The species' variant mesh as a Geom (uploaded once, shared).

    Docs: docs/systems/render.vegetation._impl.md
    """
    key = (name, variant)
    geom = self_obj._mesh_geoms.get(key)
    if geom is None:
        from fire_engine.render.bridges.geometry_bridge import to_geom

        geom = to_geom(vs.meshes[variant])
        self_obj._mesh_geoms[key] = geom
    return geom


def impostor_geom(self_obj: TreeRendererComponent, name: str, vs: Any) -> Geom:
    """Crossed-quad billboard sized to the species' impostor raster.

    Docs: docs/systems/render.vegetation._impl.md
    """
    geom = self_obj._impostor_geoms.get(name)
    if geom is None:
        geom = _build_cross_geom(vs.impostor_height_m, vs.impostor_width_m, 2)
        self_obj._impostor_geoms[name] = geom
    return geom


def species_atlas(self_obj: TreeRendererComponent, name: str, vs: Any) -> Texture:
    """The species' bark/leaf atlas as a nearest-filtered texture.

    Docs: docs/systems/render.vegetation._impl.md
    """
    tex = self_obj._atlas_tex.get(name)
    if tex is None:
        from fire_engine.render.bridges.texture_bridge import to_panda_texture

        tex = to_panda_texture(vs.atlas)
        self_obj._atlas_tex[name] = tex
    return tex


def species_impostor(self_obj: TreeRendererComponent, name: str, vs: Any) -> Texture:
    """The species' impostor sprite strip as a texture.

    Docs: docs/systems/render.vegetation._impl.md
    """
    tex = self_obj._impostor_tex.get(name)
    if tex is None:
        from fire_engine.render.bridges.texture_bridge import to_panda_texture

        tex = to_panda_texture(vs.impostors)
        self_obj._impostor_tex[name] = tex
    return tex
