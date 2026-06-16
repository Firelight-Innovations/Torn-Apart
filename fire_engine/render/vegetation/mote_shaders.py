"""
world/mote_shaders.py — GLSL for the GPU-instanced wind particles (dust + leaves).

The GLSL source lives in ``world/shaders/mote_dust.{vert,frag}`` and
``world/shaders/mote_leaf.{vert,frag}`` (loaded verbatim via ``load_glsl`` so
editors get syntax highlighting + LSP); this module just re-exports the loaded
strings — the ``world/grass_shaders.py`` pattern.

Both layers are **GPU-instanced with zero CPU per-particle state**: every
instance derives its placement / life / motion in the vertex shader from
``gl_InstanceID`` via the same lowbias32 hash chain grass uses, and advects by
sampling the inherited ``u_wind_tex`` (decoded with the two-line idiom from
``grass.vert``).  Nodes parent under ``terrain_root`` so the wind, fog, lighting
and camera uniforms arrive by scene-graph inheritance — no new uniform contract.

Dust shader
-----------
Vertex: a camera-anchored wrapping lattice (``floor(cam/box)*box`` + hashed
offset) of ``u_mote_count`` motes, each on a looping ``sin(life*PI)`` life,
carried downwind by the local field + a re-hashed Brownian jitter + a gentle
turbulence-driven rise, billboarded in view space.  Fragment: soft radial
``dust_mote`` texture, **additive**, distance-dimmed by the froxel fog so far
motes fade into it.

Leaf shader
-----------
Vertex: one instanced node per ``"trees"`` :class:`~fire_engine.zones.ZoneVolume`
(``u_bounds_min/max``), leaves spawned inside the volume biased low, carried by
the local wind × ``(0.3 + 0.7·gust)`` so litter settles in calm air and streams
in gusts, tumbling under two hashed angular rates about two hashed axes, and
picking one of the 3 ``leaf_sprite`` atlas variants from a hash.  Fragment: the
SAME radiance-cascade + froxel-fog taps as ``grass.frag`` (so leaves are lit by
the scene), alpha-blended with the grass discard threshold.
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = [
    "DUST_FRAGMENT",
    "DUST_VERTEX",
    "LEAF_FRAGMENT",
    "LEAF_VERTEX",
]


DUST_VERTEX = load_glsl(__file__, "mote_dust.vert")
DUST_FRAGMENT = load_glsl(__file__, "mote_dust.frag")
LEAF_VERTEX = load_glsl(__file__, "mote_leaf.vert")
LEAF_FRAGMENT = load_glsl(__file__, "mote_leaf.frag")
