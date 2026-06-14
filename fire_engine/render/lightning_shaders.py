"""
world/lightning_shaders.py — GLSL for the M7 procedural lightning bolts.

The GLSL source lives in ``world/shaders/lightning.{vert,frag}`` (loaded verbatim
via ``load_glsl`` so editors get syntax highlighting + LSP); this module just
re-exports the loaded strings — the ``world/rain_shaders.py`` /
``world/mote_shaders.py`` pattern.

The vertex shader expands each bolt SEGMENT (uploaded as a quad with the segment
endpoints in custom vertex columns) into a camera-facing ribbon, hidden below a
top-down ``u_reveal`` front; the fragment shader emits a hot HDR core + soft glow
scaled by the per-phase ``u_flash`` brightness (additive, so it blooms in post).
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = ["LIGHTNING_VERTEX", "LIGHTNING_FRAGMENT"]


LIGHTNING_VERTEX = load_glsl(__file__, "lightning.vert")
LIGHTNING_FRAGMENT = load_glsl(__file__, "lightning.frag")
