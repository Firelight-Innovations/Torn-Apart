"""
tests/render/bridges/test_post_shaders.py — Headless tests for render/bridges/post_shaders.py.

Verifies the seven HDR post-processing GLSL shader constants (no panda3d).
"""

from __future__ import annotations

import fire_engine.render.bridges.post_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("POST_FULLSCREEN_VERTEX", _mod.POST_FULLSCREEN_VERTEX),
    ("BLOOM_DOWN_FRAGMENT", _mod.BLOOM_DOWN_FRAGMENT),
    ("BLOOM_UP_FRAGMENT", _mod.BLOOM_UP_FRAGMENT),
    ("LENS_FLARE_FRAGMENT", _mod.LENS_FLARE_FRAGMENT),
    ("GOD_RAYS_FRAGMENT", _mod.GOD_RAYS_FRAGMENT),
    ("FXAA_FRAGMENT", _mod.FXAA_FRAGMENT),
    ("COMPOSITE_FRAGMENT", _mod.COMPOSITE_FRAGMENT),
]


class TestPostShaderExports:
    def test_all_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)

    def test_non_empty_strings(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert isinstance(val, str), f"{name} not str"
            assert len(val) > 50, f"{name} too short"


class TestPostShaderGlslTokens:
    def test_every_shader_has_version_and_main(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "#version" in val, f"{name} missing #version"
            assert "void main" in val, f"{name} missing void main"

    def test_composite_has_scene_uniform(self) -> None:
        """Composite reads the linear HDR scene texture via u_scene."""
        assert "u_scene" in _mod.COMPOSITE_FRAGMENT

    def test_composite_has_bloom_uniform(self) -> None:
        assert "u_bloom" in _mod.COMPOSITE_FRAGMENT

    def test_fxaa_has_tex_uniform(self) -> None:
        """FXAA reads the tonemapped LDR image via u_tex."""
        assert "u_tex" in _mod.FXAA_FRAGMENT

    def test_bloom_strength_in_composite(self) -> None:
        assert "u_bloom_strength" in _mod.COMPOSITE_FRAGMENT

    def test_godray_has_uniform(self) -> None:
        assert "u_godray" in _mod.COMPOSITE_FRAGMENT


class TestPostShaderDeterminism:
    def test_fullscreen_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "fullscreen.vert")
        b = load_glsl(_mod.__file__, "fullscreen.vert")
        assert a == b

    def test_bloom_down_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "bloom_down.frag")
        b = load_glsl(_mod.__file__, "bloom_down.frag")
        assert a == b

    def test_composite_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "composite.frag")
        b = load_glsl(_mod.__file__, "composite.frag")
        assert a == b

    def test_fxaa_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "fxaa.frag")
        b = load_glsl(_mod.__file__, "fxaa.frag")
        assert a == b
