"""
tests/test_shader_source.py — the GLSL sidecar loader + include directive.

Headless (no panda3d): ``core/shader_source.py`` is plain file I/O.  Covers:

1. Sources without an include directive pass through byte-identical.
2. ``//#include "<name>"`` expands to the included file's text wrapped in
   begin/end marker comments, verbatim, at the directive's position.
3. Missing include → FileNotFoundError naming includer and missing path.
4. Nested include (included file contains a directive) → ValueError.
"""

from __future__ import annotations

import pathlib

import pytest

from fire_engine.core.shader_source import load_glsl

_REPO = pathlib.Path(__file__).resolve().parents[1]


def _write_shaders(tmp_path: pathlib.Path, files: dict[str, str]) -> str:
    """Lay out ``files`` under ``<tmp_path>/shaders/`` and return the anchor
    (a fake module path beside that directory) for ``load_glsl``."""
    shader_dir = tmp_path / "shaders"
    shader_dir.mkdir(exist_ok=True)
    for name, text in files.items():
        (shader_dir / name).write_text(text, encoding="utf-8")
    return str(tmp_path / "fake_module.py")


class TestPassthrough:
    def test_passthrough_without_directive_is_byte_identical(self, tmp_path):
        src = "#version 330 core\nvoid main() { /* no includes */ }\n"
        anchor = _write_shaders(tmp_path, {"plain.frag": src})
        assert load_glsl(anchor, "plain.frag") == src

    def test_plain_comment_mentioning_include_is_not_a_directive(self, tmp_path):
        # Only the exact `//#include "..."` line shape fires; prose doesn't.
        src = '#version 330 core\n// see the //#include "x" directive docs\n'
        anchor = _write_shaders(tmp_path, {"plain.frag": src})
        assert load_glsl(anchor, "plain.frag") == src

    def test_real_repo_shaders_still_load(self):
        # The loader change must not disturb existing sidecar files.
        anchor = str(_REPO / "fire_engine" / "render" / "fake.py")
        text = load_glsl(anchor, "terrain.frag")
        assert text.startswith("#version")


class TestIncludeExpansion:
    def test_include_expands_verbatim(self, tmp_path):
        lib = "float litOne() { return 1.0; }\n"
        src = (
            "#version 330 core\n"
            '//#include "lib.glsl"\n'
            "void main() {}\n"
        )
        anchor = _write_shaders(tmp_path, {"lib.glsl": lib, "user.frag": src})
        out = load_glsl(anchor, "user.frag")
        assert lib in out                          # included text verbatim
        assert '//#include "lib.glsl"' not in out  # directive consumed
        # Order preserved: version line, then library, then main.
        assert out.index("#version") < out.index("litOne") < out.index("main")

    def test_include_markers_present(self, tmp_path):
        anchor = _write_shaders(
            tmp_path,
            {"lib.glsl": "float x;\n",
             "user.frag": '#version 330 core\n//#include "lib.glsl"\n'},
        )
        out = load_glsl(anchor, "user.frag")
        assert "// --- begin include: lib.glsl ---" in out
        assert "// --- end include: lib.glsl ---" in out

    def test_include_without_trailing_newline_stays_well_formed(self, tmp_path):
        anchor = _write_shaders(
            tmp_path,
            {"lib.glsl": "float x;",  # no trailing newline
             "user.frag": '#version 330 core\n//#include "lib.glsl"\nvoid main() {}\n'},
        )
        out = load_glsl(anchor, "user.frag")
        # The end marker must land on its own line, not glued to `float x;`.
        assert "float x;\n// --- end include: lib.glsl ---" in out

    def test_indented_directive_fires(self, tmp_path):
        anchor = _write_shaders(
            tmp_path,
            {"lib.glsl": "float x;\n",
             "user.frag": '#version 330 core\n    //#include "lib.glsl"\n'},
        )
        assert "float x;" in load_glsl(anchor, "user.frag")


class TestIncludeErrors:
    def test_missing_include_raises_filenotfound(self, tmp_path):
        anchor = _write_shaders(
            tmp_path, {"user.frag": '#version 330 core\n//#include "nope.glsl"\n'}
        )
        with pytest.raises(FileNotFoundError, match=r'user\.frag.*nope\.glsl'):
            load_glsl(anchor, "user.frag")

    def test_nested_include_raises_valueerror(self, tmp_path):
        anchor = _write_shaders(
            tmp_path,
            {"inner.glsl": "float y;\n",
             "lib.glsl": '//#include "inner.glsl"\nfloat x;\n',
             "user.frag": '#version 330 core\n//#include "lib.glsl"\n'},
        )
        with pytest.raises(ValueError, match="nested"):
            load_glsl(anchor, "user.frag")
