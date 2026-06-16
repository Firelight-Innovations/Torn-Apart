"""
tests/world/sky/test_atmosphere.py — Mirror tests for fire_engine/world/sky/atmosphere.py.

Covers the physically-based single-scattering atmosphere model:
 - Constants are correct physical values
 - sun_radiance: correctness, determinism, twilight fade, scalar vs. array
 - sky_radiance: shape, sign, blue zenith, vectorization
 - transmittance: shape, range, monotonicity, planet-hit zero
 - sky_ambient: shape, range, determinism

No panda3d imports. All tests headless and deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.world.sky import atmosphere

NOON_Z = 0.94  # sin(elevation) at v0 noon arc peak
SUN_NOON = np.array([0.34, 0.0, 0.94])
ZENITH_DIR = np.array([[0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Physical constants — smoke-test that they exist and have expected magnitudes
# ---------------------------------------------------------------------------


class TestConstants:
    def test_beta_rayleigh_shape_and_blue_dominant(self):
        """Blue channel of BETA_RAYLEIGH must be largest (≈5.7× red)."""
        b = atmosphere.BETA_RAYLEIGH
        assert b.shape == (3,)
        assert b[2] > b[1] > b[0]

    def test_planet_radius_order_of_magnitude(self):
        assert 6e6 < atmosphere.PLANET_RADIUS_M < 7e6

    def test_atmosphere_top_order_of_magnitude(self):
        assert 5e4 < atmosphere.ATMOSPHERE_TOP_M < 1e5

    def test_mie_g_anisotropy_in_range(self):
        assert 0.0 < atmosphere.MIE_G < 1.0

    def test_sun_toa_radiance_positive(self):
        assert atmosphere.SUN_TOA_RADIANCE > 0.0

    def test_sun_fade_lo_z_is_negative(self):
        """SUN_FADE_LO_Z should be sin(-4°) ≈ -0.0698."""
        assert atmosphere.SUN_FADE_LO_Z < 0.0
        assert abs(atmosphere.SUN_FADE_LO_Z - math.sin(math.radians(-4.0))) < 1e-6


# ---------------------------------------------------------------------------
# sun_radiance — correctness, determinism, twilight
# ---------------------------------------------------------------------------


class TestSunRadiance:
    def test_noon_near_contract_target(self):
        """Clear noon (sun_z ≈ 0.94) must be near (3.2, 3.0, 2.6)."""
        r = atmosphere.sun_radiance(NOON_Z)
        assert abs(r[0] - 3.2) < 0.5
        assert abs(r[1] - 3.0) < 0.5
        assert abs(r[2] - 2.6) < 0.5

    def test_scalar_returns_shape_3(self):
        r = atmosphere.sun_radiance(0.5)
        assert r.shape == (3,)

    def test_array_returns_n3(self):
        zs = np.array([0.94, 0.5, 0.1])
        r = atmosphere.sun_radiance(zs)
        assert r.shape == (3, 3)

    def test_scalar_matches_array_row(self):
        z = 0.5
        scalar = atmosphere.sun_radiance(z)
        arr = atmosphere.sun_radiance(np.array([z]))
        np.testing.assert_array_equal(scalar, arr[0])

    def test_exactly_zero_below_fade_cutoff(self):
        z_below = math.sin(math.radians(-5.0))
        r = atmosphere.sun_radiance(z_below)
        assert np.all(r == 0.0)

    def test_twilight_tail_nonzero_at_minus_two_degrees(self):
        z_twilight = math.sin(math.radians(-2.0))
        r = atmosphere.sun_radiance(z_twilight)
        assert r.sum() > 0.0

    def test_sunset_redder_than_noon(self):
        noon = atmosphere.sun_radiance(NOON_Z)
        dusk = atmosphere.sun_radiance(0.03)
        noon_rb = noon[0] / noon[2]
        dusk_rb = dusk[0] / max(dusk[2], 1e-9)
        assert dusk_rb > noon_rb * 3.0

    def test_noon_brighter_than_dusk(self):
        noon = atmosphere.sun_radiance(NOON_Z)
        dusk = atmosphere.sun_radiance(0.03)
        assert noon.sum() > dusk.sum()

    def test_non_negative_and_finite_across_range(self):
        zs = np.linspace(atmosphere.SUN_FADE_LO_Z - 0.05, 1.0, 30)
        r = atmosphere.sun_radiance(zs)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)

    def test_deterministic(self):
        a = atmosphere.sun_radiance(0.3)
        b = atmosphere.sun_radiance(0.3)
        np.testing.assert_array_equal(a, b)

    def test_boundary_zero_finite(self):
        r = atmosphere.sun_radiance(0.0)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)

    def test_boundary_one_finite(self):
        r = atmosphere.sun_radiance(1.0)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)


# ---------------------------------------------------------------------------
# sky_radiance — shape, sign, physical blue-zenith
# ---------------------------------------------------------------------------


class TestSkyRadiance:
    def test_zenith_is_blue_at_midday(self):
        L = atmosphere.sky_radiance(ZENITH_DIR, SUN_NOON)
        assert L[0, 2] > L[0, 1] > L[0, 0]

    def test_returns_n3_for_batch(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.7, 0.0, 0.4],
            ]
        )
        L = atmosphere.sky_radiance(dirs, SUN_NOON)
        assert L.shape == (3, 3)

    def test_1d_input_returns_1x3(self):
        L = atmosphere.sky_radiance(np.array([0.0, 0.0, 1.0]), SUN_NOON)
        assert L.shape == (1, 3)

    def test_all_non_negative(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.7, 0.0, 0.4],
            ]
        )
        L = atmosphere.sky_radiance(dirs, SUN_NOON)
        assert np.all(L >= 0.0)

    def test_all_finite(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.707],
            ]
        )
        L = atmosphere.sky_radiance(dirs, SUN_NOON)
        assert np.all(np.isfinite(L))

    def test_deterministic(self):
        dirs = np.array([[0.0, 0.0, 1.0], [0.7, 0.0, 0.4]])
        L1 = atmosphere.sky_radiance(dirs, SUN_NOON)
        L2 = atmosphere.sky_radiance(dirs, SUN_NOON)
        np.testing.assert_array_equal(L1, L2)

    def test_single_matches_batch_row(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.7, 0.0, 0.4],
            ]
        )
        batch = atmosphere.sky_radiance(dirs, SUN_NOON)
        for i, d in enumerate(dirs):
            single = atmosphere.sky_radiance(np.array([d]), SUN_NOON)
            np.testing.assert_allclose(single[0], batch[i], atol=0.0, rtol=0.0)

    def test_night_sun_produces_near_zero(self):
        """When the sun is far below the horizon, sky radiance should be very dim."""
        sun_below = np.array([0.0, 0.0, -1.0])
        L = atmosphere.sky_radiance(ZENITH_DIR, sun_below)
        assert L.sum() < 1e-3


# ---------------------------------------------------------------------------
# transmittance — shape, range, monotonicity
# ---------------------------------------------------------------------------


class TestTransmittance:
    def test_returns_n3_for_batch(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.7, 0.0, 0.3],
                [0.5, 0.5, 0.707],
            ]
        )
        T = atmosphere.transmittance(dirs)
        assert T.shape == (3, 3)

    def test_zenith_in_range_0_1(self):
        T = atmosphere.transmittance(np.array([[0.0, 0.0, 1.0]]))[0]
        assert np.all(T >= 0.0) and np.all(T <= 1.0)

    def test_red_greater_than_blue_at_zenith(self):
        """Rayleigh scattering removes more blue — red transmits better."""
        T = atmosphere.transmittance(np.array([[0.0, 0.0, 1.0]]))[0]
        assert T[0] > T[2]

    def test_below_horizon_returns_zero(self):
        T = atmosphere.transmittance(np.array([[0.0, 0.0, -1.0]]))[0]
        assert np.all(T == 0.0)

    def test_all_finite_for_upward_rays(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.866],
                [0.866, 0.0, 0.5],
            ]
        )
        T = atmosphere.transmittance(dirs)
        assert np.all(np.isfinite(T))

    def test_decreases_toward_horizon(self):
        t_zen = atmosphere.transmittance(np.array([[0.0, 0.0, 1.0]]))[0]
        t_mid = atmosphere.transmittance(np.array([[0.866, 0.0, 0.5]]))[0]
        t_low = atmosphere.transmittance(np.array([[0.99, 0.0, 0.14]]))[0]
        assert np.all(t_zen >= t_mid)
        assert np.all(t_mid >= t_low)

    def test_deterministic(self):
        dirs = np.array([[0.0, 0.0, 1.0], [0.7, 0.0, 0.3]])
        np.testing.assert_array_equal(
            atmosphere.transmittance(dirs),
            atmosphere.transmittance(dirs),
        )


# ---------------------------------------------------------------------------
# sky_ambient — shape, range, determinism
# ---------------------------------------------------------------------------


class TestSkyAmbient:
    def test_returns_shape_3(self):
        r = atmosphere.sky_ambient(NOON_Z)
        assert r.shape == (3,)

    def test_noon_blue_dominant(self):
        r = atmosphere.sky_ambient(NOON_Z)
        assert r[2] > r[1] > r[0]

    def test_noon_in_contract_range(self):
        r = atmosphere.sky_ambient(NOON_Z)
        assert 0.15 < r[0] < 0.45
        assert 0.25 < r[1] < 0.60
        assert 0.45 < r[2] < 0.95

    def test_night_near_zero(self):
        r = atmosphere.sky_ambient(-0.3)
        assert np.all(r < 1e-4)

    def test_noon_brighter_than_sunset(self):
        noon = atmosphere.sky_ambient(NOON_Z)
        sunset = atmosphere.sky_ambient(0.02)
        assert noon.sum() > sunset.sum()

    def test_non_negative_finite_daytime(self):
        r = atmosphere.sky_ambient(NOON_Z)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)

    def test_non_negative_finite_below_horizon(self):
        r = atmosphere.sky_ambient(-0.5)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)

    def test_deterministic(self):
        a = atmosphere.sky_ambient(0.5)
        b = atmosphere.sky_ambient(0.5)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("sun_z", [-0.5, -0.1, 0.0, 0.3, 0.7, 1.0])
    def test_finite_non_negative_across_range(self, sun_z):
        r = atmosphere.sky_ambient(sun_z)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)
