"""
tests/render/sky/test_lightning_shaders.py — Headless tests for render/sky/lightning_shaders.py.

Verifies the two GLSL lightning shader constants (no panda3d).
"""

from __future__ import annotations

import fire_engine.render.sky.lightning_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("LIGHTNING_VERTEX", _mod.LIGHTNING_VERTEX),
    ("LIGHTNING_FRAGMENT", _mod.LIGHTNING_FRAGMENT),
]


class TestLightningShaderExports:
    def test_all_exports_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)
            assert isinstance(getattr(_mod, name), str)

    def test_all_constants_are_non_empty_strings(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert isinstance(val, str), f"{name} is not a str"
            assert len(val) > 50, f"{name} too short"


class TestLightningShaderGlslTokens:
    def test_every_shader_has_version(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "#version" in val, f"{name} missing #version"

    def test_every_shader_has_main(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "void main" in val, f"{name} missing void main"

    def test_vertex_has_u_flash_uniform(self) -> None:
        assert "u_flash" in _mod.LIGHTNING_VERTEX

    def test_vertex_has_u_reveal_uniform(self) -> None:
        assert "u_reveal" in _mod.LIGHTNING_VERTEX


class TestLightningShaderDeterminism:
    def test_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "lightning.vert")
        b = load_glsl(_mod.__file__, "lightning.vert")
        assert a == b

    def test_fragment_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "lightning.frag")
        b = load_glsl(_mod.__file__, "lightning.frag")
        assert a == b
