"""
world/rain_shaders.py — GLSL for the M6 volumetric rain (particles + cylinders).

The GLSL source lives in ``world/shaders/rain_particles.{vert,frag}`` and
``world/shaders/rain_cylinder.{vert,frag}`` (loaded verbatim via ``load_glsl``
so editors get syntax highlighting + LSP); this module just re-exports the
loaded strings — the ``world/mote_shaders.py`` / ``grass_shaders.py`` pattern.

Particle mode
-------------
A camera-anchored wrapping lattice of GPU-instanced falling streaks (zero CPU
per-particle state — every instance derives its placement / fall phase / sway
in the vertex shader from ``gl_InstanceID``).  Each instance is gated by two
samples at its world XY: the rain-cover heightmap (discard under a roof — the
M6 fix) and the weather-map precip channel (rain only inside storm footprints).

Cylinder mode
-------------
The cheap low-preset path: the old nested camera-following cylinders, but the
fragment shader now applies the SAME heightmap cull + precip gate per fragment,
so even the cheap mode stops raining under cover.
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = [
    "RAIN_CYLINDER_FRAGMENT",
    "RAIN_CYLINDER_VERTEX",
    "RAIN_PARTICLE_FRAGMENT",
    "RAIN_PARTICLE_VERTEX",
]


RAIN_PARTICLE_VERTEX = load_glsl(__file__, "rain_particles.vert")
RAIN_PARTICLE_FRAGMENT = load_glsl(__file__, "rain_particles.frag")
RAIN_CYLINDER_VERTEX = load_glsl(__file__, "rain_cylinder.vert")
RAIN_CYLINDER_FRAGMENT = load_glsl(__file__, "rain_cylinder.frag")
