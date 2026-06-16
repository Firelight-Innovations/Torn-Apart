"""
tests/lighting/test_palette.py — Headless tests for fire_engine.lighting.palette.

Covers:
- MaterialPalette construction and field shapes.
- MaterialPalette.with_emission: copy semantics, value set correctly.
- build_default_palette: air is zero, determinism, emission is zero.

No panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.rng import set_world_seed
from fire_engine.lighting.palette import MaterialPalette, build_default_palette


class TestMaterialPalette:
    def test_default_constructor_shapes(self):
        pal = MaterialPalette()
        assert pal.albedo.shape == (256, 3)
        assert pal.emission.shape == (256, 3)

    def test_default_constructor_dtype(self):
        pal = MaterialPalette()
        assert pal.albedo.dtype == np.float32
        assert pal.emission.dtype == np.float32

    def test_default_albedo_is_zero(self):
        pal = MaterialPalette()
        assert (pal.albedo == 0).all()

    def test_default_emission_is_zero(self):
        pal = MaterialPalette()
        assert (pal.emission == 0).all()

    def test_with_emission_sets_correct_row(self):
        pal = MaterialPalette()
        pal2 = pal.with_emission(7, (1.0, 2.0, 3.0))
        np.testing.assert_allclose(pal2.emission[7], (1.0, 2.0, 3.0))

    def test_with_emission_is_a_copy_not_mutating_original(self):
        pal = MaterialPalette()
        _ = pal.with_emission(7, (1.0, 2.0, 3.0))
        assert (pal.emission[7] == 0).all()

    def test_with_emission_albedo_unchanged(self):
        albedo = np.ones((256, 3), dtype=np.float32) * 0.5
        pal = MaterialPalette(albedo=albedo.copy(), emission=np.zeros((256, 3), dtype=np.float32))
        pal2 = pal.with_emission(3, (2.0, 0.5, 0.1))
        np.testing.assert_allclose(pal2.albedo, albedo)

    def test_fancy_index_materials(self):
        """palette.albedo[materials] must work for a full uint8 material array."""
        pal = MaterialPalette()
        materials = np.zeros((32, 32, 32), dtype=np.uint8)
        materials[0, 0, 0] = 1
        result = pal.albedo[materials]
        assert result.shape == (32, 32, 32, 3)
        assert result.dtype == np.float32


class TestBuildDefaultPalette:
    def test_air_row_is_zero(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401

        pal = build_default_palette()
        assert (pal.albedo[0] == 0).all()

    def test_emission_is_all_zero(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401

        pal = build_default_palette()
        assert (pal.emission == 0).all()

    def test_material_1_and_2_albedo_differ(self):
        """Dirt and grass must have distinct albedos."""
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401

        pal = build_default_palette()
        assert not np.allclose(pal.albedo[1], pal.albedo[2])

    def test_deterministic_same_seed(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401

        a = build_default_palette().albedo.tobytes()
        b = build_default_palette().albedo.tobytes()
        assert a == b

    def test_solid_materials_have_nonzero_albedo(self):
        """At minimum material ids 1 and 2 must have non-zero albedo."""
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401

        pal = build_default_palette()
        assert not (pal.albedo[1] == 0).all()
        assert not (pal.albedo[2] == 0).all()
