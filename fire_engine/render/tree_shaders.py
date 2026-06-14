"""
world/tree_shaders.py — GLSL for the instanced 3-D tree/bush renderer.

Source files (loaded verbatim via ``load_glsl``):

- ``shaders/tree.vert`` + ``shaders/tree.frag`` — the variant-mesh draws:
  per-instance transform via ``texelFetch`` from the RGBA32F data texture
  baked by ``zones/tree_placement.py::instances_data_block`` (the layout is
  a CPU↔GLSL contract — tests/test_tree_placement.py pins it), per-vertex
  sway weights in ``p3d_Color.a``, real-normal Lambert in the fragment.
- ``shaders/tree_impostor.vert`` — the far-LOD billboard stage reading the
  SAME data texture with the opposite fade window.  Its fragment stage is
  ``flora_shaders.FLORA_FRAGMENT`` verbatim (sprite cutout + base-anchored
  cascade lighting).

See ``world/tree_renderer.py`` for the render component and
``docs/systems/world.md`` for the full reference.
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = ["TREE_FRAGMENT", "TREE_IMPOSTOR_VERTEX", "TREE_VERTEX"]


TREE_VERTEX = load_glsl(__file__, "tree.vert")
TREE_FRAGMENT = load_glsl(__file__, "tree.frag")
TREE_IMPOSTOR_VERTEX = load_glsl(__file__, "tree_impostor.vert")
