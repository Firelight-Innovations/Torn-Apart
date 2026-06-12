"""
world/terrain_shader.py — GLSL surface shader for GPU-volumetric-lit terrain.

Replaces the fixed-function texture × baked-vertex-colour pipeline when
``config.lighting_backend == "gpu"``.  Per fragment it:

1. samples **direct sun/moon** light through the cascade visibility volume
   (voxel-marched shadows, computed in `lighting/glsl.py` INJECT) with
   Lambert shading on the (optionally normal-mapped) surface normal,
2. samples **indirect GI** from the ray-gathered radiance cascades, with the
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
from fire_engine.procedural.textures.ground_lut import build_ground_lut
from fire_engine.procedural.textures.grass_ground import (
    GRASS_PALETTE, GRASS_THRESHOLDS)
from fire_engine.procedural.textures.dirt_ground import (
    DIRT_PALETTE, DIRT_THRESHOLDS)
from fire_engine.terrain.generation import MATERIAL_DIRT, MATERIAL_GRASS
from fire_engine.world.texture_bridge import to_field_texture

__all__ = ["apply_terrain_shader", "TERRAIN_VERTEX", "TERRAIN_FRAGMENT"]


TERRAIN_VERTEX = load_glsl(__file__, "terrain.vert")
TERRAIN_FRAGMENT = load_glsl(__file__, "terrain.frag")


def apply_terrain_shader(
    terrain_root: NodePath,
    pipeline,
    *,
    seed: float = 0.0,
    texels_per_m: float = 16.0,
    extra_materials=None,
) -> None:
    """
    Compile and apply the volumetric terrain shader to ``terrain_root``.

    The lighting uniform contract is NOT bound here: main.py binds it once on
    ``app.render`` (``pipeline.bind_surface_inputs(app.render)``) and the
    App's frame loop refreshes it there each frame
    (``pipeline.update_surface_inputs(app.render, sky_state)``) — every
    lit-surface shader in the graph (terrain, foliage, future buildings/NPCs)
    inherits it from ``render``.

    Also bakes the **world-space procedural ground** palette LUT (one row per
    terrain material, from the ``grass_ground``/``dirt_ground`` colour ramps)
    and binds it plus the noise parameters.  The fragment shader colours the
    ground from a non-repeating world-space noise value indexed into this LUT,
    so the tiled 64×64 albedo never appears and the ground never repeats.

    Parameters
    ----------
    terrain_root : NodePath
        Parent of every chunk Geom (``App.terrain_root``).
    pipeline : GpuLightingPipeline
        The active lighting pipeline (`lighting/gpu.py`).
    seed : float, optional
        Per-world hash offset for the procedural ground noise (pass a value
        derived from the world seed via ``core.rng.for_domain`` for
        determinism).  Default 0.
    texels_per_m : float, optional
        Virtual texels per world meter for the ground pattern
        (``config.ground_texels_per_m``).  Default 16.
    extra_materials : Mapping[int, tuple[ndarray, ndarray]] | None, optional
        Additional ``material id → (palette, thresholds)`` LUT rows merged on
        top of the built-in grass/dirt entries — used by debug/test materials
        (e.g. the GI test-room white/red/green/glow surfaces) so they colour
        from a flat palette instead of clamping to the last LUT row.  Palettes
        are sRGB-encoded uint8 (the shader gamma-decodes via ``pow(alb, 2.2)``).
    """
    shader = Shader.make(Shader.SL_GLSL,
                         vertex=TERRAIN_VERTEX,
                         fragment=TERRAIN_FRAGMENT)
    terrain_root.set_shader(shader)
    # The lighting uniform contract itself is bound on ``render`` (main.py
    # calls ``pipeline.bind_surface_inputs(app.render)``) so EVERY lit-surface
    # shader in the graph inherits it — terrain only binds what is its own.
    # Terrain always runs the celestial-shadow refinement march (the
    # lit_surface.glsl LIT_REFINE block); foliage roots bind their own
    # value from ``config.gfx_foliage_shadow_refine``.
    terrain_root.set_shader_input("u_refine", 1.0)

    # World-space procedural ground palette LUT (rows indexed by material id).
    entries = {
        MATERIAL_DIRT:  (DIRT_PALETTE,  DIRT_THRESHOLDS),
        MATERIAL_GRASS: (GRASS_PALETTE, GRASS_THRESHOLDS),
    }
    if extra_materials:
        entries.update(extra_materials)
    lut = build_ground_lut(entries)
    terrain_root.set_shader_input("u_ground_lut", to_field_texture(lut))
    terrain_root.set_shader_input("u_ground_seed", float(seed))
    terrain_root.set_shader_input("u_ground_texels_per_m", float(texels_per_m))
    terrain_root.set_shader_input("u_ground_lut_rows", float(lut.shape[0]))
