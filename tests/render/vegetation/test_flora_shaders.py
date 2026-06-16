"""
tests/render/vegetation/test_flora_shaders.py — Headless tests for
render/vegetation/flora_shaders.py.

No panda3d imports.
"""

from __future__ import annotations

import fire_engine.render.vegetation.flora_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("FLORA_VERTEX", _mod.FLORA_VERTEX),
    ("FLORA_FRAGMENT", _mod.FLORA_FRAGMENT),
]


class TestFloraShaderExports:
    def test_all_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)

    def test_non_empty_strings(self) -> None:
        for _name, val in _ALL_CONSTANTS:
            assert isinstance(val, str)
            assert len(val) > 50


class TestFloraShaderGlslTokens:
    def test_version_and_main_present(self) -> None:
        for _name, val in _ALL_CONSTANTS:
            assert "#version" in val
            assert "void main" in val

    def test_vertex_uses_instance_id(self) -> None:
        """Flora uses gl_InstanceID for zero-CPU placement."""
        assert "gl_InstanceID" in _mod.FLORA_VERTEX

    def test_fragment_has_fog_enabled_uniform(self) -> None:
        assert "u_fog_enabled" in _mod.FLORA_FRAGMENT


class TestFloraShaderDeterminism:
    def test_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "flora.vert")
        b = load_glsl(_mod.__file__, "flora.vert")
        assert a == b

    def test_fragment_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "flora.frag")
        b = load_glsl(_mod.__file__, "flora.frag")
        assert a == b
