"""
tests/procedural/textures/sky/test__night_sky_helpers.py
— Tests for fire_engine/procedural/textures/sky/_night_sky_helpers.py.

Covers the shared constants and helper functions used by both night_sky and
night_sky_cube.  Headless — no panda3d imports.
"""

from __future__ import annotations

import math

import numpy as np


class TestSharedConstants:
    def test_galaxy_inclination_non_zero(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _GALAXY_INCLINATION_RAD,
        )

        assert _GALAXY_INCLINATION_RAD > 0.0
        assert math.pi > _GALAXY_INCLINATION_RAD

    def test_galaxy_core_sigma_less_than_halo(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _GALAXY_CORE_SIGMA,
            _GALAXY_HALO_SIGMA,
        )

        assert 0.0 < _GALAXY_CORE_SIGMA < _GALAXY_HALO_SIGMA

    def test_sky_floor_shape_and_positive(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import _SKY_FLOOR

        assert _SKY_FLOOR.shape == (3,)
        assert (_SKY_FLOOR > 0).all(), "Sky floor must be positive (non-black)"

    def test_band_star_fraction_in_range(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _BAND_STAR_FRACTION,
        )

        assert 0.0 < _BAND_STAR_FRACTION < 1.0

    def test_galaxy_ramp_keys_ascending(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import _GALAXY_RAMP_KEYS

        diffs = np.diff(_GALAXY_RAMP_KEYS)
        assert (diffs > 0).all(), "Galaxy ramp keys must be strictly ascending"

    def test_galaxy_ramp_rgb_shape_matches_keys(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _GALAXY_RAMP_KEYS,
            _GALAXY_RAMP_RGB,
        )

        assert _GALAXY_RAMP_RGB.shape == (len(_GALAXY_RAMP_KEYS), 3)

    def test_galaxy_ramp_rgb_in_unit_range(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import _GALAXY_RAMP_RGB

        assert float(_GALAXY_RAMP_RGB.min()) >= 0.0
        assert float(_GALAXY_RAMP_RGB.max()) <= 1.0


class TestCubeFaceDirections:
    def test_output_shape(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import cube_face_directions

        res = 8
        dirs = cube_face_directions(res)
        assert dirs.shape == (6, res, res, 3), f"Expected (6,{res},{res},3), got {dirs.shape}"

    def test_unit_vectors(self):
        """All direction vectors must be unit-length (within float tolerance)."""
        from fire_engine.procedural.textures.sky._night_sky_helpers import cube_face_directions

        dirs = cube_face_directions(16)
        norms = np.linalg.norm(dirs, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"cube_face_directions must return unit vectors; max err={np.abs(norms - 1).max():.6f}"
        )

    def test_six_distinct_face_centres(self):
        """The six face-centre directions must be the ±X, ±Y, ±Z axes."""
        from fire_engine.procedural.textures.sky._night_sky_helpers import cube_face_directions

        dirs = cube_face_directions(3)  # small grid — centre pixel is exact
        centres = dirs[:, 1, 1]  # middle texel of each face
        norms = np.linalg.norm(centres, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        # Each face-centre direction must have exactly one ±1 component
        # (the other two being ≈0).
        for fc in centres:
            abs_fc = np.abs(fc)
            assert abs_fc.max() > 0.9, "Face centre should align with an axis"
            assert (abs_fc > 0.9).sum() == 1, "Exactly one dominant axis per face"


class TestRampRgb:
    def test_at_zero_returns_first_key_colour(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _GALAXY_RAMP_KEYS,
            _GALAXY_RAMP_RGB,
            _ramp_rgb,
        )

        t = np.array([_GALAXY_RAMP_KEYS[0]])
        result = _ramp_rgb(t, _GALAXY_RAMP_KEYS, _GALAXY_RAMP_RGB)
        np.testing.assert_allclose(result[0], _GALAXY_RAMP_RGB[0], atol=1e-5)

    def test_at_one_returns_last_key_colour(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _GALAXY_RAMP_KEYS,
            _GALAXY_RAMP_RGB,
            _ramp_rgb,
        )

        t = np.array([_GALAXY_RAMP_KEYS[-1]])
        result = _ramp_rgb(t, _GALAXY_RAMP_KEYS, _GALAXY_RAMP_RGB)
        np.testing.assert_allclose(result[0], _GALAXY_RAMP_RGB[-1], atol=1e-5)

    def test_output_shape(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _GALAXY_RAMP_KEYS,
            _GALAXY_RAMP_RGB,
            _ramp_rgb,
        )

        t = np.linspace(0, 1, 50)
        result = _ramp_rgb(t, _GALAXY_RAMP_KEYS, _GALAXY_RAMP_RGB)
        assert result.shape == (50, 3)

    def test_output_in_unit_range(self):
        from fire_engine.procedural.textures.sky._night_sky_helpers import (
            _GALAXY_RAMP_KEYS,
            _GALAXY_RAMP_RGB,
            _ramp_rgb,
        )

        t = np.linspace(0, 1, 100)
        result = _ramp_rgb(t, _GALAXY_RAMP_KEYS, _GALAXY_RAMP_RGB)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0
