"""
world/building_shaders.py — GLSL for the free-form building renderer.

Source files (loaded verbatim via ``load_glsl``, which expands the
``//#include "lit_surface.glsl"`` directive):

- ``shaders/building.vert`` — derives the world position from
  ``p3d_ModelMatrix`` (the building's node transform) so a building meshed in
  local space lights at its true world cell.
- ``shaders/building.frag`` — the shared lit-surface contract over the
  plaster-wall albedo (``p3d_Texture0``); identical lighting recipe to terrain
  and trees (one ``sampleCascades`` + soft-penumbra refine + voxel AO + fog +
  ``litFinish``).  Sampler budget: 9 cascade + 1 fog + 1 albedo = 11 of 16.

See ``world/building_renderer.py`` for the render component and
``docs/systems/world.md`` for the full reference.

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = ["BUILDING_FRAGMENT", "BUILDING_VERTEX"]


BUILDING_VERTEX = load_glsl(__file__, "building.vert")
BUILDING_FRAGMENT = load_glsl(__file__, "building.frag")
