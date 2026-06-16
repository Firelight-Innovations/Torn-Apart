"""
tests/render/sky/test_rain_shaders.py — Headless tests for render/sky/rain_shaders.py.

Verifies the four GLSL rain shader constants (no panda3d).
"""

from __future__ import annotations

import fire_engine.render.sky.rain_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("RAIN_PARTICLE_VERTEX", _mod.RAIN_PARTICLE_VERTEX),
    ("RAIN_PARTICLE_FRAGMENT", _mod.RAIN_PARTICLE_FRAGMENT),
    ("RAIN_CYLINDER_VERTEX", _mod.RAIN_CYLINDER_VERTEX),
    ("RAIN_CYLINDER_FRAGMENT", _mod.RAIN_CYLINDER_FRAGMENT),
]


class TestRainShaderExports:
    def test_all_exports_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)

    def test_all_constants_are_non_empty_strings(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert isinstance(val, str), f"{name} is not a str"
            assert len(val) > 50, f"{name} is too short"


class TestRainShaderGlslTokens:
    def test_every_shader_has_version_directive(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "#version" in val, f"{name} missing #version"

    def test_every_shader_has_void_main(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "void main" in val, f"{name} missing void main"

    def test_particle_vertex_uses_instance_id(self) -> None:
        assert "gl_InstanceID" in _mod.RAIN_PARTICLE_VERTEX

    def test_cylinder_fragment_has_discard_or_frag_color(self) -> None:
        src = _mod.RAIN_CYLINDER_FRAGMENT
        assert "discard" in src or "FragColor" in src or "gl_FragColor" in src


class TestRainShaderDeterminism:
    def test_particle_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "rain_particles.vert")
        b = load_glsl(_mod.__file__, "rain_particles.vert")
        assert a == b

    def test_cylinder_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "rain_cylinder.vert")
        b = load_glsl(_mod.__file__, "rain_cylinder.vert")
        assert a == b
