"""
world/grass_shaders.py — GLSL for the GPU-only instanced grass.

The GLSL source now lives in ``world/shaders/grass.vert`` and
``world/shaders/grass.frag`` (loaded verbatim via ``load_glsl``) so editors get
syntax highlighting + LSP; this module just re-exports the loaded strings.

The CPU never stores a blade: every instance derives its placement in the
vertex shader from ``gl_InstanceID`` via the lowbias32 hash chain that
``zones/grass_placement.py`` mirrors line-for-line (edit BOTH or the headless
placement tests lie about what the GPU draws).

Vertex shader
-------------
1. Hash ``gl_InstanceID`` (+ per-volume ``u_hash_seed``) → base XY inside
   ``u_bounds_min/max``, yaw, scale jitter, sway phase, tint.
2. Sample ``u_height_field`` (R channel: terrain surface height inside the
   volume's Z window; 255 = no ground) — sentinel or fully-faded instances
   collapse to a clip-space point and rasterise nothing (craters cull grass).
3. Sway: blade-local Z² weighted lean along ``u_wind_dir`` — a static lean
   (``u_sway_base``) plus a gust oscillation (``u_sway_gust`` ×
   sin(``u_time_s·u_gust_freq`` + phase)); both amplitudes are computed
   CPU-side from the weather (storms move grass more).
4. Distance fade: blades shrink to nothing between ``u_fade_start_m`` and
   ``u_fade_end_m`` from the camera — no popping, no far-field shimmer.

Fragment shader
---------------
Binary alpha cutout of the pixel-art ``grass_tuft`` texture (discard < 0.5 —
no sorting, depth-write stays on), lit by the SAME radiance-cascade volumes
as the terrain: direct sun/moon × voxel-marched visibility + ray-gathered GI,
sampled at the blade base quantised to the ``u_quant_m`` light-pixel grid, so
grass shows the identical pixelated light patches, torch glow and crater
shadows as the ground it stands on.  Froxel fog composites with one tap, then
ACES + gamma — matching ``world/terrain_shader.py``.

All cascade/fog/celestial uniforms use the ``GpuLightingPipeline`` surface
contract names and are **inherited** from ``terrain_root`` (the pipeline
binds and refreshes them there each frame); only the grass-specific uniforms
are set by ``GrassRendererComponent``.
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = ["GRASS_FRAGMENT", "GRASS_VERTEX"]


GRASS_VERTEX = load_glsl(__file__, "grass.vert")
GRASS_FRAGMENT = load_glsl(__file__, "grass.frag")
