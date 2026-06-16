"""
tests/render/sky/test_sky_shaders.py — Headless tests for render/sky/sky_shaders.py.

Verifies GLSL string constants (no panda3d; pure string analysis + determinism).
"""

from __future__ import annotations

import fire_engine.render.sky.sky_shaders as _mod
from fire_engine.core.shader_source import load_glsl


class TestSkyShaderExports:
    """All __all__ exports are non-empty strings."""

    def test_all_exports_defined(self) -> None:
        for name in _mod.__all__:
            assert hasattr(_mod, name), f"Missing export: {name}"
            val = getattr(_mod, name)
            assert isinstance(val, str), f"{name} is not a str"
            assert len(val) > 0, f"{name} is empty"

    def test_sky_dome_vertex_is_str(self) -> None:
        assert isinstance(_mod.SKY_DOME_VERTEX, str)
        assert len(_mod.SKY_DOME_VERTEX) > 50

    def test_sky_dome_fragment_is_str(self) -> None:
        assert isinstance(_mod.SKY_DOME_FRAGMENT, str)
        assert len(_mod.SKY_DOME_FRAGMENT) > 50

    def test_cloud_volumetric_vertex_is_str(self) -> None:
        assert isinstance(_mod.CLOUD_VOLUMETRIC_VERTEX, str)
        assert len(_mod.CLOUD_VOLUMETRIC_VERTEX) > 50

    def test_cloud_volumetric_fragment_is_str(self) -> None:
        assert isinstance(_mod.CLOUD_VOLUMETRIC_FRAGMENT, str)
        assert len(_mod.CLOUD_VOLUMETRIC_FRAGMENT) > 50


class TestSkyShaderGlslTokens:
    """Each constant contains the mandatory GLSL structural tokens."""

    def test_sky_dome_vertex_has_version_and_main(self) -> None:
        assert "#version" in _mod.SKY_DOME_VERTEX
        assert "void main" in _mod.SKY_DOME_VERTEX

    def test_sky_dome_fragment_has_version_and_main(self) -> None:
        assert "#version" in _mod.SKY_DOME_FRAGMENT
        assert "void main" in _mod.SKY_DOME_FRAGMENT

    def test_sky_dome_fragment_has_sun_dir_uniform(self) -> None:
        assert "u_sun_dir" in _mod.SKY_DOME_FRAGMENT

    def test_sky_dome_fragment_has_moon_dir_uniform(self) -> None:
        assert "u_moon_dir" in _mod.SKY_DOME_FRAGMENT

    def test_sky_dome_fragment_has_star_visibility_uniform(self) -> None:
        assert "u_star_visibility" in _mod.SKY_DOME_FRAGMENT

    def test_sky_dome_fragment_has_daylight_uniform(self) -> None:
        assert "u_daylight" in _mod.SKY_DOME_FRAGMENT

    def test_cloud_fragment_has_version_and_main(self) -> None:
        assert "#version" in _mod.CLOUD_VOLUMETRIC_FRAGMENT
        assert "void main" in _mod.CLOUD_VOLUMETRIC_FRAGMENT

    def test_cloud_fragment_has_coverage_uniform(self) -> None:
        assert "u_coverage" in _mod.CLOUD_VOLUMETRIC_FRAGMENT

    def test_cloud_fragment_has_altitude_uniform(self) -> None:
        assert "u_altitude" in _mod.CLOUD_VOLUMETRIC_FRAGMENT

    def test_cloud_vertex_has_version_and_main(self) -> None:
        assert "#version" in _mod.CLOUD_VOLUMETRIC_VERTEX
        assert "void main" in _mod.CLOUD_VOLUMETRIC_VERTEX


class TestSkyShaderDeterminism:
    """load_glsl returns byte-identical results on repeated calls."""

    def test_sky_dome_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "sky_dome.vert")
        b = load_glsl(_mod.__file__, "sky_dome.vert")
        assert a == b

    def test_sky_dome_fragment_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "sky_dome.frag")
        b = load_glsl(_mod.__file__, "sky_dome.frag")
        assert a == b

    def test_cloud_vertex_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "cloud_volumetric.vert")
        b = load_glsl(_mod.__file__, "cloud_volumetric.vert")
        assert a == b

    def test_cloud_fragment_deterministic(self) -> None:
        a = load_glsl(_mod.__file__, "cloud_volumetric.frag")
        b = load_glsl(_mod.__file__, "cloud_volumetric.frag")
        assert a == b
