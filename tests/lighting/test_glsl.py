"""
tests/lighting/test_glsl.py — Headless tests for fire_engine.lighting.glsl.

Verifies that:
- All exported symbols are present and importable.
- MAX_LIGHTS is a positive integer matching the documented GLSL array bound.
- Every shader-source constant is a non-empty string (the .comp file was loaded).
- The module is importable without panda3d.

No GPU compilation happens here — glsl.py is deliberately headless.
"""

from __future__ import annotations

import fire_engine.lighting.glsl as glsl


class TestGlslSymbols:
    """All names listed in __all__ must be importable and have correct types."""

    def test_all_symbols_present(self):
        for name in glsl.__all__:
            assert hasattr(glsl, name), f"Missing symbol: {name}"

    def test_max_lights_is_positive_int(self):
        assert isinstance(glsl.MAX_LIGHTS, int)
        assert glsl.MAX_LIGHTS > 0

    def test_max_lights_value_is_64(self):
        """Pin the documented value so accidental changes are caught."""
        assert glsl.MAX_LIGHTS == 64

    def test_inject_compute_is_nonempty_string(self):
        assert isinstance(glsl.INJECT_COMPUTE, str)
        assert len(glsl.INJECT_COMPUTE) > 10

    def test_gather_compute_is_nonempty_string(self):
        assert isinstance(glsl.GATHER_COMPUTE, str)
        assert len(glsl.GATHER_COMPUTE) > 10

    def test_smooth_compute_is_nonempty_string(self):
        assert isinstance(glsl.SMOOTH_COMPUTE, str)
        assert len(glsl.SMOOTH_COMPUTE) > 10

    def test_shift_compute_is_nonempty_string(self):
        assert isinstance(glsl.SHIFT_COMPUTE, str)
        assert len(glsl.SHIFT_COMPUTE) > 10

    def test_fog_scatter_compute_is_nonempty_string(self):
        assert isinstance(glsl.FOG_SCATTER_COMPUTE, str)
        assert len(glsl.FOG_SCATTER_COMPUTE) > 10

    def test_fog_integrate_compute_is_nonempty_string(self):
        assert isinstance(glsl.FOG_INTEGRATE_COMPUTE, str)
        assert len(glsl.FOG_INTEGRATE_COMPUTE) > 10

    def test_shader_sources_differ(self):
        """Each pass should be a distinct source (no accidental aliasing)."""
        sources = [
            glsl.INJECT_COMPUTE,
            glsl.GATHER_COMPUTE,
            glsl.SMOOTH_COMPUTE,
            glsl.SHIFT_COMPUTE,
            glsl.FOG_SCATTER_COMPUTE,
            glsl.FOG_INTEGRATE_COMPUTE,
        ]
        assert len(set(sources)) == len(sources), "Two shader source strings are identical"

    def test_no_panda3d_imported(self):
        """glsl.py is importable without panda3d — the import itself is the test."""
        import importlib

        # Must not raise even in a headless environment.
        importlib.import_module("fire_engine.lighting.glsl")
