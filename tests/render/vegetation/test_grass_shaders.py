"""
tests/render/vegetation/test_grass_shaders.py — Headless tests for
render/vegetation/grass_shaders.py.

No panda3d imports.
"""

from __future__ import annotations

import fire_engine.render.vegetation.grass_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("GRASS_VERTEX", _mod.GRASS_VERTEX),
    ("GRASS_FRAGMENT", _mod.GRASS_FRAGMENT),
]


class TestGrassShaderExports:
    def test_all_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)

    def test_non_empty_strings(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert isinstance(val, str), f"{name} not str"
            assert len(val) > 50


class TestGrassShaderGlslTokens:
    def test_version_and_main_present(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "#version" in val, f"{name} missing #version"
            assert "void main" in val, f"{name} missing void main"

    def test_vertex_has_height_field_uniform(self) -> None:
        assert "u_height_field" in _mod.GRASS_VERTEX

    def test_vertex_has_wind_dir_uniform(self) -> None:
        assert "u_wind_dir" in _mod.GRASS_VERTEX

    def test_vertex_has_time_s_uniform(self) -> None:
        assert "u_time_s" in _mod.GRASS_VERTEX

    def test_fragment_has_fog_enabled_uniform(self) -> None:
        assert "u_fog_enabled" in _mod.GRASS_FRAGMENT


class TestGrassShaderDeterminism:
    def test_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "grass.vert")
        b = load_glsl(_mod.__file__, "grass.vert")
        assert a == b

    def test_fragment_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "grass.frag")
        b = load_glsl(_mod.__file__, "grass.frag")
        assert a == b
