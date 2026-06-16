"""
tests/render/vegetation/test_tree_shaders.py — Headless tests for render/vegetation/tree_shaders.py.

No panda3d imports.
"""

from __future__ import annotations

import fire_engine.render.vegetation.tree_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("TREE_VERTEX", _mod.TREE_VERTEX),
    ("TREE_FRAGMENT", _mod.TREE_FRAGMENT),
    ("TREE_IMPOSTOR_VERTEX", _mod.TREE_IMPOSTOR_VERTEX),
]


class TestTreeShaderExports:
    def test_all_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)

    def test_non_empty_strings(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert isinstance(val, str), f"{name} not str"
            assert len(val) > 50, f"{name} too short"


class TestTreeShaderGlslTokens:
    def test_version_and_main_present(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "#version" in val, f"{name} missing #version"
            assert "void main" in val, f"{name} missing void main"

    def test_tree_vertex_uses_texel_fetch(self) -> None:
        """tree.vert reads per-instance transform via texelFetch from RGBA32F data texture."""
        assert "texelFetch" in _mod.TREE_VERTEX

    def test_tree_impostor_vertex_uses_instance_id(self) -> None:
        """Far-LOD billboard reads the same data texture using gl_InstanceID."""
        assert "gl_InstanceID" in _mod.TREE_IMPOSTOR_VERTEX

    def test_tree_fragment_has_version_and_main(self) -> None:
        assert "#version" in _mod.TREE_FRAGMENT
        assert "void main" in _mod.TREE_FRAGMENT


class TestTreeShaderDeterminism:
    def test_tree_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "tree.vert")
        b = load_glsl(_mod.__file__, "tree.vert")
        assert a == b

    def test_tree_impostor_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "tree_impostor.vert")
        b = load_glsl(_mod.__file__, "tree_impostor.vert")
        assert a == b

    def test_tree_fragment_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "tree.frag")
        b = load_glsl(_mod.__file__, "tree.frag")
        assert a == b
