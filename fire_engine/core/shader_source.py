"""
core/shader_source.py — load GLSL shader source from sidecar files.

Shader source lives in real ``.vert`` / ``.frag`` / ``.comp`` files (under a
``shaders/`` directory beside the module that uses them) instead of Python
string literals, so editors give GLSL syntax highlighting and an LSP can lint
them.  The thin Python shader modules (``world/grass_shaders.py``,
``world/sky_shaders.py``, ``world/terrain_shader.py``, ``lighting/glsl.py``)
read those files at import time through :func:`load_glsl` and re-export the
text under their original constant names, so callers are unchanged.

Reading a text file is panda3d-free, so this lives in ``core`` and is callable
from any layer.

Example
-------
>>> from fire_engine.core.shader_source import load_glsl
>>> GRASS_VERTEX = load_glsl(__file__, "grass.vert")   # reads ./shaders/grass.vert
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["load_glsl"]


def load_glsl(anchor: str, name: str) -> str:
    """Return the text of GLSL source ``name`` from the ``shaders/`` directory
    beside ``anchor``.

    Parameters
    ----------
    anchor:
        The calling module's ``__file__``.  The shader is looked up at
        ``<dir of anchor>/shaders/<name>``.
    name:
        Shader filename including its stage extension, e.g. ``"grass.vert"``,
        ``"sky_dome.frag"`` or ``"inject.comp"``.

    Returns
    -------
    str
        The full shader source (including its ``#version`` line), ready to
        hand to ``panda3d.core.Shader.make`` / ``Shader.make_compute``.

    Raises
    ------
    FileNotFoundError
        If the shader file does not exist.
    """
    return (Path(anchor).parent / "shaders" / name).read_text(encoding="utf-8")
