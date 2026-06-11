"""
world/terrain_shader.py — GLSL surface shader for GPU-volumetric-lit terrain.

Replaces the fixed-function texture × baked-vertex-colour pipeline when
``config.lighting_backend == "gpu"``.  Per fragment it:

1. samples **direct sun/moon** light through the cascade visibility volume
   (voxel-marched shadows, computed in `lighting/glsl.py` INJECT) with
   Lambert shading on the (optionally normal-mapped) surface normal,
2. samples **indirect GI** from the flood-fill radiance cascades, with the
   sample position quantised to ``light_quant_m`` (0.25 m) — the pixelated
   "light pixels" look (2×2×2 per terrain voxel),
3. applies **voxel AO** from the occupancy volume,
4. adds **emission** (own-cell volume emission + per-texel emission map),
5. composites **volumetric fog** by one tap into the integrated froxel
   texture at this fragment's screen position + depth (god rays included),
6. tonemaps (ACES approximation) and gamma-encodes.

Texture stages per material Geom (built by `world/geometry_bridge.py`):
``p3d_Texture0`` albedo, ``p3d_Texture1`` tangent-space normal map,
``p3d_Texture2`` emission map.  The TBN basis is analytic from the dominant
normal axis — exactly the axis pair the mesher uses for planar UVs.

All lighting inputs (samplers + uniforms) are bound by
``GpuLightingPipeline.bind_surface_inputs`` / ``update_surface_inputs``.

Example
-------
    from fire_engine.world.terrain_shader import apply_terrain_shader
    apply_terrain_shader(app.terrain_root, pipeline)   # once at boot

The GLSL source now lives in ``world/shaders/terrain.vert`` and
``world/shaders/terrain.frag`` (loaded via ``core.shader_source.load_glsl``)
for editor syntax highlighting + LSP support.
"""

from __future__ import annotations

from panda3d.core import NodePath, Shader  # type: ignore[import]

from fire_engine.core.shader_source import load_glsl

__all__ = ["apply_terrain_shader", "TERRAIN_VERTEX", "TERRAIN_FRAGMENT"]


TERRAIN_VERTEX = load_glsl(__file__, "terrain.vert")
TERRAIN_FRAGMENT = load_glsl(__file__, "terrain.frag")


def apply_terrain_shader(terrain_root: NodePath, pipeline) -> None:
    """
    Compile and apply the volumetric terrain shader to ``terrain_root``.

    Binds the pipeline's static lighting inputs immediately; the App's frame
    loop must call ``pipeline.update_surface_inputs(terrain_root, sky_state)``
    each frame (window origins, radiance ping-pong, sun/moon uniforms).

    Parameters
    ----------
    terrain_root : NodePath
        Parent of every chunk Geom (``App.terrain_root``).
    pipeline : GpuLightingPipeline
        The active lighting pipeline (`lighting/gpu.py`).
    """
    shader = Shader.make(Shader.SL_GLSL,
                         vertex=TERRAIN_VERTEX,
                         fragment=TERRAIN_FRAGMENT)
    terrain_root.set_shader(shader)
    pipeline.bind_surface_inputs(terrain_root)
