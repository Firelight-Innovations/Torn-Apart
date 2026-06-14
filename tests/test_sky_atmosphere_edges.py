"""
tests/test_sky_atmosphere_edges.py — Edge-case / characterization tests for
``sky/atmosphere.py``.

Pins current behaviour as a golden-master. Do NOT fix bugs here; any
deviation from a pinned assert is a regression signal. Suspected anomalies
are noted inline via ``# SUSPECT:`` comments.

Does NOT duplicate cases from tests/test_sky_atmosphere.py (determinism at
fixed sun_z, noon radiance magnitudes, sunset hue, below-horizon extinguish,
zenith blue, transmittance red-bias, ambient noon range + night zero).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.world.sky import atmosphere


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOON_Z = 0.94  # sin(elevation) at v0 noon arc peak
HORIZON_Z = 0.0  # sin(0°) — exactly at the geometric horizon
ZENITH_DIR = np.array([[0.0, 0.0, 1.0]])
SUN_NOON = np.array([0.34, 0.0, 0.94])  # matches existing fixture


# ---------------------------------------------------------------------------
# sky_radiance — vectorization correctness
# ---------------------------------------------------------------------------


class TestSkyRadianceVectorization:
    """Batch calls must behave identically to individual calls."""

    VIEW_DIRS_BATCH = np.array(
        [
            [0.0, 0.0, 1.0],  # zenith
            [1.0, 0.0, 0.0],  # horizon east
            [0.0, 1.0, 0.0],  # horizon north
            [0.0, -1.0, 0.0],  # horizon south
            [0.7, 0.0, 0.4],  # mid-elevation (used in existing determinism test)
            [-0.6, 0.0, 0.8],  # upper hemisphere, away from sun
            [0.0, 0.5, 0.866],  # NE upper
        ]
    )

    def test_batch_returns_N3_shape(self):
        """An (N, 3) batch of view dirs must yield (N, 3) output."""
        result = atmosphere.sky_radiance(self.VIEW_DIRS_BATCH, SUN_NOON)
        assert result.shape == (len(self.VIEW_DIRS_BATCH), 3)

    def test_batch_all_finite(self):
        """All batch outputs must be finite (no NaN or Inf)."""
        result = atmosphere.sky_radiance(self.VIEW_DIRS_BATCH, SUN_NOON)
        assert np.all(np.isfinite(result)), "sky_radiance produced non-finite values"

    def test_batch_all_non_negative(self):
        """Radiance is energy — no negative values allowed."""
        result = atmosphere.sky_radiance(self.VIEW_DIRS_BATCH, SUN_NOON)
        assert np.all(result >= 0.0), "sky_radiance produced negative values"

    def test_single_dir_matches_batch_row(self):
        """Each single-direction call must equal the matching row of a batch call."""
        batch = atmosphere.sky_radiance(self.VIEW_DIRS_BATCH, SUN_NOON)
        for i, d in enumerate(self.VIEW_DIRS_BATCH):
            single = atmosphere.sky_radiance(np.array([d]), SUN_NOON)
            np.testing.assert_allclose(
                single[0],
                batch[i],
                rtol=0.0,
                atol=0.0,
                err_msg=f"dir[{i}] single vs batch mismatch",
            )

    def test_1d_input_coerced_to_N1(self):
        """A bare (3,) view dir must also return (1, 3), not raise."""
        result = atmosphere.sky_radiance(np.array([0.0, 0.0, 1.0]), SUN_NOON)
        assert result.shape == (1, 3)

    def test_determinism_batch(self):
        """Calling sky_radiance twice with identical args is bit-equal (pure)."""
        first = atmosphere.sky_radiance(self.VIEW_DIRS_BATCH, SUN_NOON)
        second = atmosphere.sky_radiance(self.VIEW_DIRS_BATCH, SUN_NOON)
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Rayleigh blue-dominance in clear sky
# ---------------------------------------------------------------------------


class TestRayleighBlueDominance:
    """The doc promises B > R for clear-day zenith/sky — pin that direction."""

    def test_zenith_blue_over_red_daytime(self):
        """At clear noon, zenith B channel must dominate R (Rayleigh physics)."""
        L = atmosphere.sky_radiance(ZENITH_DIR, SUN_NOON)
        assert L[0, 2] > L[0, 0], (
            f"Zenith: B={L[0, 2]:.4f} not > R={L[0, 0]:.4f} — Rayleigh blue expected"
        )

    def test_zenith_blue_over_green_daytime(self):
        """Standard Rayleigh: B > G at zenith under noon sun."""
        L = atmosphere.sky_radiance(ZENITH_DIR, SUN_NOON)
        assert L[0, 2] > L[0, 1], f"Zenith: B={L[0, 2]:.4f} not > G={L[0, 1]:.4f}"

    def test_upper_hemisphere_batch_mostly_blue_dominant(self):
        """In a diverse batch, the majority of clear-sky samples should be B > R."""
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.866],
                [-0.5, 0.0, 0.866],
                [0.0, 0.5, 0.866],
                [0.3, 0.3, 0.9],
            ]
        )
        L = atmosphere.sky_radiance(dirs, SUN_NOON)
        blue_dom = np.sum(L[:, 2] > L[:, 0])
        assert blue_dom >= len(dirs) - 1, (
            f"Expected Rayleigh blue-dom in most directions, got {blue_dom}/{len(dirs)}"
        )

    def test_sky_ambient_noon_blue_over_red(self):
        """sky_ambient at noon should be blue-dominant (pinned from docs contract)."""
        amb = atmosphere.sky_ambient(NOON_Z)
        assert amb[2] > amb[0], f"sky_ambient noon: B={amb[2]:.4f} not > R={amb[0]:.4f}"


# ---------------------------------------------------------------------------
# transmittance — shape, range, and monotonicity
# ---------------------------------------------------------------------------


class TestTransmittance:
    def test_returns_N3_for_batch(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.7, 0.0, 0.3],
                [0.5, 0.5, 0.707],
            ]
        )
        T = atmosphere.transmittance(dirs)
        assert T.shape == (3, 3)

    def test_all_components_in_0_1(self):
        """Transmittance must be in (0, 1] for upward rays."""
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.866],
                [0.866, 0.0, 0.5],
            ]
        )
        T = atmosphere.transmittance(dirs)
        assert np.all(T >= 0.0), "transmittance has negative components"
        assert np.all(T <= 1.0), "transmittance exceeds 1.0"

    def test_finite_non_negative(self):
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.01],  # near-grazing upward
            ]
        )
        T = atmosphere.transmittance(dirs)
        assert np.all(np.isfinite(T))
        assert np.all(T >= 0.0)

    def test_transmittance_decreases_toward_horizon(self):
        """Longer atmospheric path at lower elevations must reduce transmittance."""
        t_zenith = atmosphere.transmittance(np.array([[0.0, 0.0, 1.0]]))[0]
        t_mid = atmosphere.transmittance(np.array([[0.866, 0.0, 0.5]]))[0]
        t_low = atmosphere.transmittance(np.array([[0.99, 0.0, 0.14]]))[0]
        # Each channel must monotonically decrease toward the horizon
        assert np.all(t_zenith >= t_mid), "transmittance not monotone zenith→mid"
        assert np.all(t_mid >= t_low), "transmittance not monotone mid→low"

    def test_zenith_transmittance_in_range(self):
        """
        Pin observed zenith transmittance range.

        Actual values (pinned): R≈0.947, G≈0.887, B≈0.751.
        The heavy BETA_RAYLEIGH blue coefficient (33.1e-6 vs 5.8e-6 for red)
        means blue is strongly attenuated even at zenith — the common intuition
        of ">0.9 for all channels" does NOT hold here.  Pin the observed range
        instead.

        # SUSPECT: Real Earth zenith transmittance at ~550 nm is ~0.95+; B here
        # is ~0.75 which seems aggressively low for a single zenith pass.  Could
        # indicate BETA_RAYLEIGH Blue is tuned above physical to exaggerate sky
        # colour, or that the 60 km atmosphere shell integrates more optical depth
        # than a 10 km boundary-layer model would.  Flag but do not fix.
        """
        T = atmosphere.transmittance(np.array([[0.0, 0.0, 1.0]]))[0]
        # Red channel must be fairly high (short Rayleigh path)
        assert T[0] > 0.90, f"Zenith R transmittance too low: {T[0]:.4f}"
        # Green is intermediate
        assert T[1] > 0.80, f"Zenith G transmittance too low: {T[1]:.4f}"
        # Blue is significantly lower (strong Rayleigh at short λ)
        assert T[2] > 0.60, f"Zenith B transmittance too low: {T[2]:.4f}"
        # All still less than 1
        assert np.all(T <= 1.0), f"Transmittance exceeded 1.0: {T}"

    def test_below_horizon_zero(self):
        """Rays that strike the planet must return exactly 0."""
        # Straight down — guaranteed planet hit
        T = atmosphere.transmittance(np.array([[0.0, 0.0, -1.0]]))[0]
        assert np.all(T == 0.0), f"Below-horizon ray returned non-zero transmittance: {T}"

    def test_determinism(self):
        dirs = np.array([[0.0, 0.0, 1.0], [0.7, 0.0, 0.3]])
        np.testing.assert_array_equal(
            atmosphere.transmittance(dirs),
            atmosphere.transmittance(dirs),
        )


# ---------------------------------------------------------------------------
# sun_radiance — shape, sign, and monotonicity
# ---------------------------------------------------------------------------


class TestSunRadiance:
    def test_scalar_returns_shape_3(self):
        out = atmosphere.sun_radiance(NOON_Z)
        assert out.shape == (3,)

    def test_array_input_returns_N3(self):
        zs = np.array([0.94, 0.5, 0.1])
        out = atmosphere.sun_radiance(zs)
        assert out.shape == (3, 3)

    def test_non_negative_finite_across_range(self):
        zs = np.linspace(atmosphere.SUN_FADE_LO_Z - 0.05, 1.0, 30)
        out = atmosphere.sun_radiance(zs)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_high_sun_brighter_than_horizon(self):
        """Clear noon must be brighter in total than near-horizon sun."""
        noon = atmosphere.sun_radiance(NOON_Z)
        horizon = atmosphere.sun_radiance(0.02)  # ~1° elevation
        assert noon.sum() > horizon.sum(), "sun_radiance: noon not brighter than horizon"

    def test_monotone_dimming_toward_horizon(self):
        """Decreasing sun elevation from noon to near-horizon → dimmer each step."""
        zs = [NOON_Z, 0.5, 0.2, 0.05]
        totals = [atmosphere.sun_radiance(z).sum() for z in zs]
        for i in range(len(totals) - 1):
            assert totals[i] > totals[i + 1], (
                f"sun_radiance not monotone at z[{i}]={zs[i]}: "
                f"{totals[i]:.4f} vs {totals[i + 1]:.4f}"
            )

    def test_exactly_zero_below_fade_cutoff(self):
        """Below SUN_FADE_LO_Z (−4°) the output must be exactly 0."""
        # −5° is safely below the cutoff
        z_below = math.sin(math.radians(-5.0))
        out = atmosphere.sun_radiance(z_below)
        assert np.all(out == 0.0), f"sun_radiance at −5°={z_below:.4f} not zero: {out}"

    def test_fade_zone_between_minus4_and_zero(self):
        """Between SUN_FADE_LO_Z and 0° the sun should be non-zero but dimming."""
        z_twilight = math.sin(math.radians(-2.0))  # −2°, within fade zone
        out = atmosphere.sun_radiance(z_twilight)
        # Must be non-zero (twilight tail) but strictly less than the horizon value
        assert out.sum() > 0.0, "Twilight tail at −2° should be non-zero"
        horizon = atmosphere.sun_radiance(0.0)
        assert out.sum() < horizon.sum(), "−2° sun brighter than 0° horizon — fade direction wrong"

    def test_boundary_sun_z_zero_finite(self):
        """sun_z=0 (exactly at horizon) must be finite and non-negative."""
        out = atmosphere.sun_radiance(0.0)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_boundary_sun_z_one_finite(self):
        """sun_z=1 (sun at zenith) must be finite and non-negative."""
        out = atmosphere.sun_radiance(1.0)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_array_single_row_matches_scalar(self):
        """sun_radiance([z]) row must equal sun_radiance(z) scalar."""
        z = 0.5
        scalar = atmosphere.sun_radiance(z)
        arr = atmosphere.sun_radiance(np.array([z]))
        np.testing.assert_array_equal(scalar, arr[0])

    def test_determinism(self):
        a = atmosphere.sun_radiance(0.3)
        b = atmosphere.sun_radiance(0.3)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# sky_ambient — sign, shape, and sun elevation sensitivity
# ---------------------------------------------------------------------------


class TestSkyAmbient:
    def test_returns_shape_3(self):
        out = atmosphere.sky_ambient(NOON_Z)
        assert out.shape == (3,)

    def test_non_negative_finite_daytime(self):
        out = atmosphere.sky_ambient(NOON_Z)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_non_negative_finite_near_horizon(self):
        out = atmosphere.sky_ambient(HORIZON_Z)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_non_negative_finite_below_horizon(self):
        """Deep below-horizon sun_z must still produce finite, non-negative output."""
        out = atmosphere.sky_ambient(-0.5)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_noon_brighter_than_sunset(self):
        """Noon sky ambient must exceed a near-sunset value in total luminance."""
        noon = atmosphere.sky_ambient(NOON_Z)
        sunset = atmosphere.sky_ambient(0.02)
        assert noon.sum() > sunset.sum(), "sky_ambient: noon not brighter than near-sunset"

    def test_determinism(self):
        a = atmosphere.sky_ambient(0.5)
        b = atmosphere.sky_ambient(0.5)
        np.testing.assert_array_equal(a, b)

    def test_boundary_zenith_sun_finite(self):
        """sun_z=1.0 (sun at zenith) edge case — must be finite and non-negative."""
        out = atmosphere.sky_ambient(1.0)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)
