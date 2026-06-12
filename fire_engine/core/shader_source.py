"""
core/shader_source.py — load GLSL shader source from sidecar files.

Shader source lives in real ``.vert`` / ``.frag`` / ``.comp`` files (under a
``shaders/`` directory beside the module that uses them) instead of Python
string literals, so editors give GLSL syntax highlighting and an LSP can lint
them.  The thin Python shader modules (``world/grass_shaders.py``,
``world/sky_shaders.py``, ``world/terrain_shader.py``, ``lighting/glsl.py``)
read those files at import time through :func:`load_glsl` and re-export the
text under their original constant names, so callers are unchanged.

Include directive
-----------------
A line of the exact shape ``//#include "lit_surface.glsl"`` is replaced by the
text of that file (looked up in the same ``shaders/`` directory), wrapped in
``// --- begin/end include`` marker comments.  The directive is a plain GLSL
comment, so editors and LSPs treat the source file as valid GLSL on its own.
Expansion is one level deep — an included file may not itself include — and
deliberately emits no ``#line`` directives (flaky on Intel drivers).  Shared
GLSL libraries (e.g. ``world/shaders/lit_surface.glsl``, the lit-surface
lighting contract) exist exactly for this directive.

Reading a text file is panda3d-free, so this lives in ``core`` and is callable
from any layer.

Example
-------
>>> from fire_engine.core.shader_source import load_glsl
>>> GRASS_VERTEX = load_glsl(__file__, "grass.vert")   # reads ./shaders/grass.vert
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["load_glsl"]


# One include per line, alone on its line: //#include "file.glsl"
_INCLUDE_RE = re.compile(r'^\s*//#include\s+"([^"]+)"\s*$')


def _expand_includes(text: str, shader_dir: Path, includer: str) -> str:
    """Expand ``//#include "<name>"`` lines in ``text`` (one level deep).

    Parameters
    ----------
    text:
        Raw GLSL source of ``includer``.
    shader_dir:
        Directory included files are looked up in (the includer's own
        ``shaders/`` directory).
    includer:
        Filename of the including shader, for error messages.

    Raises
    ------
    FileNotFoundError
        If an included file does not exist (message names both the includer
        and the missing path).
    ValueError
        If an included file itself contains an include directive (nesting is
        not supported — keep libraries flat).
    """
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        match = _INCLUDE_RE.match(line)
        if match is None:
            out.append(line)
            continue
        name = match.group(1)
        path = shader_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f'{includer}: //#include "{name}" not found at {path}'
            )
        included = path.read_text(encoding="utf-8")
        if any(_INCLUDE_RE.match(inner) for inner in included.splitlines()):
            raise ValueError(
                f'{includer}: included file "{name}" contains a nested '
                f"//#include — only one level of inclusion is supported"
            )
        out.append(f"// --- begin include: {name} ---\n")
        out.append(included if included.endswith("\n") else included + "\n")
        out.append(f"// --- end include: {name} ---\n")
    return "".join(out)


def load_glsl(anchor: str, name: str) -> str:
    """Return the text of GLSL source ``name`` from the ``shaders/`` directory
    beside ``anchor``, with ``//#include`` directives expanded.

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
        The full shader source (including its ``#version`` line) with any
        ``//#include "<file>"`` lines replaced by that file's text, ready to
        hand to ``panda3d.core.Shader.make`` / ``Shader.make_compute``.
        Sources without the directive pass through byte-identical.

    Raises
    ------
    FileNotFoundError
        If the shader file (or a file it includes) does not exist.
    ValueError
        If an included file itself contains an include directive.
    """
    shader_dir = Path(anchor).parent / "shaders"
    text = (shader_dir / name).read_text(encoding="utf-8")
    return _expand_includes(text, shader_dir, name)
