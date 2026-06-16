"""
world/sky_shaders.py — GLSL sources for the procedural sky + cloud renderer.

Pure string constants (NO panda3d imports — this module is importable headless;
only ``world/sky_renderer.py`` compiles them via ``panda3d.core.Shader.make``).
All shaders are ``#version 330 core`` and use Panda3D's GLSL conventions:
``p3d_Vertex``, ``p3d_ModelViewProjectionMatrix``, ``p3d_ModelMatrix``,
``p3d_Texture0``; custom uniforms are fed via ``NodePath.set_shader_input``.

Coordinate conventions: world space, **Z-up**, meters.  All direction vectors
are unit length and point *toward* the body (sun_dir points at the sun).

SKY DOME (``SKY_DOME_VERTEX`` / ``SKY_DOME_FRAGMENT``)
-------------------------------------------------------
Painted on an inverted UV-sphere centred on the camera (translation-only
follow, so model-space vertex position IS the world view direction).

The daytime sky is a **per-pixel physical single-scattering raymarch**
(Rayleigh + Mie, constants mirrored verbatim from ``sky/atmosphere.py``):
blue zenith, bright horizon, the sunset/sunrise glow concentrated around
the sun azimuth, and the earth-shadow twilight arch all emerge from the
physics — there is no hand-painted gradient anymore.  The sun is a large
(~1.25°) limb-darkened disc tinted by its own atmospheric transmittance;
the moon is a large (~1.0°) disc textured with the procedural
``"moon_surface"`` texture and shaded by the dynamic phase terminator.
HDR output is ACES-tonemapped with the same exposure as the terrain shader;
the LDR night-sky art (stars/galaxy/twinkle/shooting stars) composites
after tonemapping so it stays crisp.

Uniforms (set per frame unless noted):
    u_star_cube       samplerCube — "night_sky_cube" 6×512² galaxy + star
                      faces (set once); alpha = per-pixel luminance (twinkle
                      mask).  Cube sampling kills the old equirect pole
                      pinch/distortion.
    u_celestial_axis  vec3  — unit, toward the celestial north pole (set
                      once; tilted by the world's seed-derived latitude so
                      stars rise/set instead of pinwheeling around zenith).
    u_moon_tex        sampler2D — "moon_surface" 256x256 lunar disc (set once).
    u_sun_dir         vec3  — unit, toward the sun.
    u_sun_color       vec3  — legacy tint (disc color now derives from the
                      atmosphere transmittance; kept for stub skies).
    u_sun_intensity   float — 0-1; scales disc + halo (cloud-dimmed).
    u_moon_dir        vec3  — unit, toward the moon.
    u_moon_phase      float — 0-1; 0.5 = full moon (terminator shading).
    u_moon_glow       float — 0-1 illuminated fraction; gates the moonlit-sky
                      scatter and the moon halo (new moon = dark sky).
    u_zenith_color    vec3  — weather-graded gradient top (legacy consumers).
    u_horizon_color   vec3  — weather-graded gradient bottom.
    u_star_visibility float — 0-1; scales the night-sky texture + twinkle.
    u_star_rotation   float — radians; whole-celestial-sphere rotation about
                      ``u_celestial_axis`` (one revolution per game day).
    u_time            float — real seconds since component start (twinkle hash).
    u_daylight        float — SkyState.daylight; blends in the night floor.
    u_weather_gray    float — 0-1 overcast desaturation weight.
    u_exposure        float — ACES tonemap exposure (matches terrain shader).
    u_fog_color       vec3  — RGB 0-1; horizon band blends toward this.
    u_fog_blend       float — 0-1 legacy horizon fog (CPU lighting backend;
                      forced to 0 under external/GPU lighting).
    u_fog_integrated  sampler3D — froxel fog accumulation (GPU backend only).
    u_fog_enabled     float — 0/1; gates the froxel fog composite.
    u_viewport        vec2  — window pixel size (froxel screen UV).
    u_ss_active       float — 0 or 1; shooting-star streak enable.
    u_ss_start        vec3  — unit view dir of the streak's spawn point (set
                      once per spawn).
    u_ss_travel       vec3  — unit travel direction (tangent; orthogonalised
                      against u_ss_start on the CPU).
    u_ss_progress     float — 0-1 animation progress along the great circle.

VOLUMETRIC CLOUDS (``CLOUD_VOLUMETRIC_VERTEX`` / ``CLOUD_VOLUMETRIC_FRAGMENT``)
------------------------------------------------------------------------------
A true 3-D raymarched cloud layer on a second camera-centred inverted sphere
(the model-space vertex IS the world view direction).  The fragment shader
analytically intersects the horizontal slab ``[u_altitude, +u_thickness]`` and
ray-marches it sampling the baked tileable 3-D noise (``sky.cloud_noise``):
a Perlin-Worley base eroded by Worley octaves + a detail volume, gated by a
height profile and ``u_coverage``.  Each lit sample marches a few steps toward
the sun for self-shadow (Beer + powder) with a Henyey-Greenstein phase (silver
lining).  Output is premultiplied ``(scattered_radiance, transmittance)`` for a
``src + dst·srcAlpha`` over-blend, so a bright sun bleeds through thin cloud and
thick cloud occludes it.  See ``docs/systems/world.md`` for the full uniform
list; the GLSL lives in ``world/shaders/cloud_volumetric.{vert,frag}``.

The GLSL source lives in ``world/shaders/*.vert`` / ``*.frag`` (loaded verbatim
via ``load_glsl``) so editors get syntax highlighting + LSP support.

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = [
    "CLOUD_VOLUMETRIC_FRAGMENT",
    "CLOUD_VOLUMETRIC_VERTEX",
    "SKY_DOME_FRAGMENT",
    "SKY_DOME_VERTEX",
]


# ---------------------------------------------------------------------------
# Sky dome
# ---------------------------------------------------------------------------

SKY_DOME_VERTEX: str = load_glsl(__file__, "sky_dome.vert")


SKY_DOME_FRAGMENT: str = load_glsl(__file__, "sky_dome.frag")


# ---------------------------------------------------------------------------
# Volumetric raymarched clouds
# ---------------------------------------------------------------------------

CLOUD_VOLUMETRIC_VERTEX: str = load_glsl(__file__, "cloud_volumetric.vert")


CLOUD_VOLUMETRIC_FRAGMENT: str = load_glsl(__file__, "cloud_volumetric.frag")
