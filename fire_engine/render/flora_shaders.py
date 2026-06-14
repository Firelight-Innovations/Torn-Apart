"""
world/flora_shaders.py — GLSL for the GPU-only instanced flora.

The GLSL source lives in ``world/shaders/flora.vert`` and
``world/shaders/flora.frag`` (loaded verbatim via ``load_glsl``); this module
just re-exports the loaded strings, mirroring ``world/grass_shaders.py``.

The CPU never stores a plant: every flower, bush and tree derives its
placement in the vertex shader from ``gl_InstanceID`` via the lowbias32 hash
chain that ``zones/flora_placement.py::flora_instance_attribs`` mirrors
line-for-line (edit BOTH or the headless placement tests lie about what the
GPU draws).  The chain is the grass chain plus one link: ``h5`` selects the
sprite-atlas variant.

See ``world/flora_renderer.py`` for the render component and
``docs/systems/world.md`` for the full reference.
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = ["FLORA_VERTEX", "FLORA_FRAGMENT"]


FLORA_VERTEX = load_glsl(__file__, "flora.vert")
FLORA_FRAGMENT = load_glsl(__file__, "flora.frag")
