"""
world/post_shaders.py — GLSL sources for the HDR post-processing chain.

Pure string constants (NO panda3d imports — importable headless; only
``world/post_process.py`` compiles them via ``panda3d.core.Shader.make``).
All passes share ``fullscreen.vert`` (a screen-spanning card emitting a [0,1]
UV) and read from RGBA16F float textures produced by ``FilterManager``.

Passes (added across phases; the loader lists only what exists today):
    POST_FULLSCREEN_VERTEX  — shared fullscreen-quad vertex shader.
    BLOOM_DOWN_FRAGMENT     — 13-tap downsample (+ soft-knee bright-pass/Karis
                              on the first level).
    BLOOM_UP_FRAGMENT       — 3x3 tent upsample + add (progressive combine).
    LENS_FLARE_FRAGMENT     — image-based ghosts + halo from the bright scene.
    GOD_RAYS_FRAGMENT       — radial crepuscular rays from the sun's screen pos.
    FXAA_FRAGMENT           — post anti-aliasing on the final LDR image.
    COMPOSITE_FRAGMENT      — scene HDR (+ bloom + flare + god rays) → ACES → gamma.

The GLSL lives in ``world/shaders/*.vert`` / ``*.frag`` (loaded via
``load_glsl``) so editors get syntax highlighting + LSP.
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = [
    "BLOOM_DOWN_FRAGMENT",
    "BLOOM_UP_FRAGMENT",
    "COMPOSITE_FRAGMENT",
    "FXAA_FRAGMENT",
    "GOD_RAYS_FRAGMENT",
    "LENS_FLARE_FRAGMENT",
    "POST_FULLSCREEN_VERTEX",
]


POST_FULLSCREEN_VERTEX: str = load_glsl(__file__, "fullscreen.vert")


BLOOM_DOWN_FRAGMENT: str = load_glsl(__file__, "bloom_down.frag")


BLOOM_UP_FRAGMENT: str = load_glsl(__file__, "bloom_up.frag")


LENS_FLARE_FRAGMENT: str = load_glsl(__file__, "lens_flare.frag")


GOD_RAYS_FRAGMENT: str = load_glsl(__file__, "god_rays.frag")


FXAA_FRAGMENT: str = load_glsl(__file__, "fxaa.frag")


COMPOSITE_FRAGMENT: str = load_glsl(__file__, "composite.frag")
