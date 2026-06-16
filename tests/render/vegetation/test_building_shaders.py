"""
tests/render/vegetation/test_building_shaders.py — Headless tests for
render/vegetation/building_shaders.py.

No panda3d imports.
"""

from __future__ import annotations

import fire_engine.render.vegetation.building_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("BUILDING_VERTEX", _mod.BUILDING_VERTEX),
    ("BUILDING_FRAGMENT", _mod.BUILDING_FRAGMENT),
]


class TestBuildingShaderExports:
    def test_all_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)

    def test_non_empty_strings(self) -> None:
        for _name, val in _ALL_CONSTANTS:
            assert isinstance(val, str)
            assert len(val) > 50


class TestBuildingShaderGlslTokens:
    def test_version_and_main_present(self) -> None:
        for _name, val in _ALL_CONSTANTS:
            assert "#version" in val
            assert "void main" in val

    def test_fragment_uses_albedo_texture(self) -> None:
        """building.frag samples from p3d_Texture0 (the albedo)."""
        assert "p3d_Texture0" in _mod.BUILDING_FRAGMENT

    def test_vertex_uses_model_matrix(self) -> None:
        """building.vert uses p3d_ModelMatrix for world-space placement."""
        assert "p3d_ModelMatrix" in _mod.BUILDING_VERTEX


class TestBuildingShaderDeterminism:
    def test_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "building.vert")
        b = load_glsl(_mod.__file__, "building.vert")
        assert a == b

    def test_fragment_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "building.frag")
        b = load_glsl(_mod.__file__, "building.frag")
        assert a == b
