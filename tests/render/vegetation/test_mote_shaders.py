"""
tests/render/vegetation/test_mote_shaders.py — Headless tests for render/vegetation/mote_shaders.py.

No panda3d imports.
"""

from __future__ import annotations

import fire_engine.render.vegetation.mote_shaders as _mod
from fire_engine.core.shader_source import load_glsl

_ALL_CONSTANTS = [
    ("DUST_VERTEX", _mod.DUST_VERTEX),
    ("DUST_FRAGMENT", _mod.DUST_FRAGMENT),
    ("LEAF_VERTEX", _mod.LEAF_VERTEX),
    ("LEAF_FRAGMENT", _mod.LEAF_FRAGMENT),
]


class TestMoteShaderExports:
    def test_all_in_all(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name)

    def test_non_empty_strings(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert isinstance(val, str), f"{name} not str"
            assert len(val) > 50, f"{name} too short"


class TestMoteShaderGlslTokens:
    def test_version_and_main_present(self) -> None:
        for name, val in _ALL_CONSTANTS:
            assert "#version" in val, f"{name} missing #version"
            assert "void main" in val, f"{name} missing void main"

    def test_dust_vertex_has_hash_seed_uniform(self) -> None:
        """Dust instances are seeded via u_hash_seed."""
        assert "u_hash_seed" in _mod.DUST_VERTEX

    def test_dust_vertex_has_time_s_uniform(self) -> None:
        assert "u_time_s" in _mod.DUST_VERTEX

    def test_dust_vertex_has_wind_tex_uniform(self) -> None:
        assert "u_wind_tex" in _mod.DUST_VERTEX

    def test_leaf_vertex_has_fog_enabled_uniform_or_lit(self) -> None:
        """Leaf fragment inherits the lighting contract (fog + cascade)."""
        assert "u_fog_enabled" in _mod.LEAF_FRAGMENT


class TestMoteShaderDeterminism:
    def test_dust_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "mote_dust.vert")
        b = load_glsl(_mod.__file__, "mote_dust.vert")
        assert a == b

    def test_leaf_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "mote_leaf.vert")
        b = load_glsl(_mod.__file__, "mote_leaf.vert")
        assert a == b
