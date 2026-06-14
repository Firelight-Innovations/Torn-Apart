"""
tests/test_weather_synoptic_integral.py — Integral consistency and scalar/vector
equivalence for fire_engine/weather/synoptic.py.

Pins CURRENT behaviour (golden-master / characterisation tests).
Does NOT fix bugs — any inconsistency is flagged in docstrings/comments.

Coverage:
  * Displacement integral consistency — central finite difference dD/dt ≈ wind
  * Scalar vs vector equivalence — wind() / wind_vec() / displacement() /
    displacement_vec() are consistent for isolated t values
  * Speed band — speed from wind() stays within [v_min, v_max] across dense t
  * Determinism — same seed → identical; different seed → different
  * Boundary / edge — t=0, large t, negative t (pins current behaviour)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather.synoptic import Synoptic

DAY = 86400.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_syn(seed: int = 42, **cfg_overrides) -> Synoptic:
    """Build a Synoptic with a fixed world seed and optional Config overrides."""
    set_world_seed(seed)
    return Synoptic(Config(**cfg_overrides))


def _cfg(seed: int = 42) -> Config:
    set_world_seed(seed)
    return Config()


# ---------------------------------------------------------------------------
# Displacement integral consistency
# ---------------------------------------------------------------------------

class TestDisplacementIntegral:
    """
    Central finite difference: (D(t+h) - D(t-h)) / (2h)  should equal wind_vec(t).

    The module docstring guarantees dD/dt ≡ W(t) to machine precision.
    We verify this numerically with h=0.5 s and expect atol=1e-6.
    """

    # Fine step – tight tolerance (analytic integral, not numerical)
    _H_FINE = 0.5          # game seconds
    _ATOL_FINE = 1e-6

    def _fd(self, syn: Synoptic, t: float, h: float = _H_FINE) -> np.ndarray:
        """Central finite difference of displacement at scalar t."""
        return (syn.displacement(t + h) - syn.displacement(t - h)) / (2.0 * h)

    def test_fd_at_t0(self):
        """t=0: both sides of the FD are defined (D(-h) is computable)."""
        syn = _make_syn()
        fd = self._fd(syn, 0.0)
        wv = syn.wind_vec(0.0)
        np.testing.assert_allclose(fd, wv, atol=self._ATOL_FINE)

    def test_fd_at_one_hour(self):
        syn = _make_syn()
        t = 3600.0
        np.testing.assert_allclose(self._fd(syn, t), syn.wind_vec(t), atol=self._ATOL_FINE)

    def test_fd_at_two_days(self):
        syn = _make_syn()
        t = 2.0 * DAY + 4321.0
        np.testing.assert_allclose(self._fd(syn, t), syn.wind_vec(t), atol=self._ATOL_FINE)

    def test_fd_at_large_t(self):
        """100 days in game-seconds (8_640_000 s) — must still be finite + consistent."""
        syn = _make_syn()
        t = 100.0 * DAY
        fd = self._fd(syn, t)
        assert np.all(np.isfinite(fd)), "FD result non-finite at large t"
        np.testing.assert_allclose(fd, syn.wind_vec(t), atol=self._ATOL_FINE)

    def test_fd_vectorised_range(self):
        """
        Check FD ≈ wind for 40 log-spaced t values spanning 0.1 h to 30 days.
        Vectorised: build all t values at once, compare arrays.
        """
        syn = _make_syn(seed=7)
        h = self._H_FINE
        t = np.logspace(math.log10(360.0), math.log10(30.0 * DAY), 40)
        fd = (syn.displacement(t + h) - syn.displacement(t - h)) / (2.0 * h)
        wv = syn.wind_vec(t)
        np.testing.assert_allclose(
            fd, wv, atol=self._ATOL_FINE,
            err_msg="dD/dt != wind_vec — displacement and wind are NOT consistent"
        )

    def test_fd_midpoints_of_each_day_for_7_days(self):
        """
        Mid-day sample of each of 7 game days — verifies the integral across
        multi-day spans, not just a single instant.
        """
        syn = _make_syn(seed=99)
        h = self._H_FINE
        midpoints = np.array([d * DAY + 12.0 * 3600.0 for d in range(7)])
        fd = (syn.displacement(midpoints + h) - syn.displacement(midpoints - h)) / (2.0 * h)
        wv = syn.wind_vec(midpoints)
        np.testing.assert_allclose(fd, wv, atol=self._ATOL_FINE)

    def test_fd_negative_t_current_behaviour(self):
        """
        Pin current behaviour for t < 0 (not documented; just ensure no crash
        and that FD is still consistent if results are finite).

        SUSPICION: negative t is well-defined mathematically (cos/sin are
        periodic) but is not guaranteed by the API contract.  This test pins
        whatever the code currently does — if it diverges in future that is
        an intentional contract change.
        """
        syn = _make_syn()
        t = -3600.0  # 1 hour before epoch
        d_neg = syn.displacement(t)
        wv_neg = syn.wind_vec(t)
        h = self._H_FINE
        fd = (syn.displacement(t + h) - syn.displacement(t - h)) / (2.0 * h)
        # Pin: result must be finite (the code doesn't clamp negative t)
        assert np.all(np.isfinite(d_neg)), "displacement(negative t) returned non-finite"
        assert np.all(np.isfinite(wv_neg)), "wind_vec(negative t) returned non-finite"
        # FD should still match wind_vec (analytic formula doesn't care about sign)
        np.testing.assert_allclose(fd, wv_neg, atol=self._ATOL_FINE)


# ---------------------------------------------------------------------------
# Scalar vs vector equivalence
# ---------------------------------------------------------------------------

class TestScalarVectorEquivalence:
    """
    wind(t) scalar form, wind_vec(scalar t), wind_vec([t]) row, and
    displacement(t) / displacement([t]) row must all agree.
    """

    _PROBE_TIMES = [0.0, 1.0, 3600.0, DAY, 2.5 * DAY + 1234.5, 7.0 * DAY]

    def test_wind_scalar_matches_wind_vec_scalar(self):
        """wind(t)[1] (speed) must equal |wind_vec(t)| and direction must match."""
        syn = _make_syn(seed=11)
        for t in self._PROBE_TIMES:
            (ux, uy), spd = syn.wind(t)
            wv = syn.wind_vec(float(t))          # scalar → (2,)
            assert wv.shape == (2,)
            np.testing.assert_allclose(spd, math.hypot(wv[0], wv[1]), atol=1e-12)
            np.testing.assert_allclose([ux, uy], wv / spd, atol=1e-9)

    def test_wind_scalar_matches_wind_vec_array_row(self):
        """wind_vec(np.array([t]))[0] must equal wind_vec(t) for each probe t."""
        syn = _make_syn(seed=22)
        for t in self._PROBE_TIMES:
            row = syn.wind_vec(np.array([float(t)]))[0]   # shape (2,)
            scalar = syn.wind_vec(float(t))               # shape (2,)
            np.testing.assert_allclose(row, scalar, atol=0.0, rtol=0.0)

    def test_displacement_scalar_matches_vec_array_row(self):
        """displacement(t) must match displacement(np.array([t]))[0]."""
        syn = _make_syn(seed=33)
        for t in self._PROBE_TIMES:
            row = syn.displacement(np.array([float(t)]))[0]
            scalar = syn.displacement(float(t))
            np.testing.assert_allclose(row, scalar, atol=0.0, rtol=0.0)

    def test_batch_wind_vec_matches_per_element(self):
        """wind_vec(t_array) equals stacking individual wind_vec(t) results."""
        syn = _make_syn(seed=44)
        ts = np.array(self._PROBE_TIMES)
        batch = syn.wind_vec(ts)                          # (M, 2)
        stacked = np.stack([syn.wind_vec(float(t)) for t in ts])
        np.testing.assert_allclose(batch, stacked, atol=0.0, rtol=0.0)

    def test_batch_displacement_matches_per_element(self):
        """displacement(t_array) equals stacking individual displacement(t)."""
        syn = _make_syn(seed=55)
        ts = np.array(self._PROBE_TIMES)
        batch = syn.displacement(ts)
        stacked = np.stack([syn.displacement(float(t)) for t in ts])
        np.testing.assert_allclose(batch, stacked, atol=0.0, rtol=0.0)

    def test_wind_vec_scalar_shape(self):
        """wind_vec(scalar) returns shape (2,) not (1, 2)."""
        syn = _make_syn()
        for t in self._PROBE_TIMES:
            assert syn.wind_vec(float(t)).shape == (2,)

    def test_wind_vec_array_shape(self):
        """wind_vec((M,) array) returns shape (M, 2)."""
        syn = _make_syn()
        ts = np.linspace(0.0, DAY, 17)
        assert syn.wind_vec(ts).shape == (17, 2)

    def test_displacement_scalar_shape(self):
        """displacement(scalar) returns shape (2,)."""
        syn = _make_syn()
        for t in self._PROBE_TIMES:
            assert syn.displacement(float(t)).shape == (2,)

    def test_displacement_zero_at_t0(self):
        """D(0) = (0, 0) — documented invariant."""
        syn = _make_syn()
        np.testing.assert_allclose(syn.displacement(0.0), [0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Speed band
# ---------------------------------------------------------------------------

class TestSpeedBand:
    """
    wind() speed must stay in [v_min, v_max] for ALL sampled times.
    The amplitude budget in __init__ guarantees this analytically; we verify
    it with a dense sample.
    """

    def test_speed_within_band_dense_30_days(self):
        """
        Dense sample: 1-game-minute resolution over 30 days.
        Uses the configured band values from Config.
        """
        set_world_seed(42)
        cfg = Config()
        syn = Synoptic(cfg)
        v_min = cfg.weather_synoptic_speed_min_ms
        v_max = cfg.weather_synoptic_speed_max_ms

        t = np.linspace(0.0, 30.0 * DAY, 30 * 24 * 60 + 1)
        speeds = np.linalg.norm(syn.wind_vec(t), axis=1)

        assert speeds.min() >= v_min - 1e-9, (
            f"speed fell below v_min={v_min}: min was {speeds.min()}"
        )
        assert speeds.max() <= v_max + 1e-9, (
            f"speed exceeded v_max={v_max}: max was {speeds.max()}"
        )

    def test_speed_within_band_large_t(self):
        """Speed band still holds at 100-day mark."""
        set_world_seed(42)
        cfg = Config()
        syn = Synoptic(cfg)
        v_min = cfg.weather_synoptic_speed_min_ms
        v_max = cfg.weather_synoptic_speed_max_ms

        t = np.linspace(90.0 * DAY, 100.0 * DAY, 10_000)
        speeds = np.linalg.norm(syn.wind_vec(t), axis=1)
        assert speeds.min() >= v_min - 1e-9
        assert speeds.max() <= v_max + 1e-9

    def test_wind_scalar_speed_within_band(self):
        """wind() convenience form returns speed inside band for probe times."""
        set_world_seed(42)
        cfg = Config()
        syn = Synoptic(cfg)
        v_min = cfg.weather_synoptic_speed_min_ms
        v_max = cfg.weather_synoptic_speed_max_ms

        probe = np.linspace(0.0, 10.0 * DAY, 200)
        for t in probe:
            _, spd = syn.wind(float(t))
            assert v_min - 1e-9 <= spd <= v_max + 1e-9, (
                f"wind() speed {spd} outside [{v_min}, {v_max}] at t={t}"
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed → identical; different seed → different (non-trivially)."""

    def test_same_seed_wind_vec_identical(self):
        t = np.linspace(0.0, 10.0 * DAY, 5000)
        a = _make_syn(seed=1337)
        b = _make_syn(seed=1337)
        np.testing.assert_array_equal(a.wind_vec(t), b.wind_vec(t))

    def test_same_seed_displacement_identical(self):
        t = np.linspace(0.0, 10.0 * DAY, 5000)
        a = _make_syn(seed=1337)
        b = _make_syn(seed=1337)
        np.testing.assert_array_equal(a.displacement(t), b.displacement(t))

    def test_different_seed_wind_differs(self):
        t = np.linspace(0.0, DAY, 500)
        wa = _make_syn(seed=1).wind_vec(t)
        wb = _make_syn(seed=2).wind_vec(t)
        assert not np.allclose(wa, wb), "Different seeds produced identical wind — suspicious"

    def test_different_seed_displacement_differs(self):
        t = np.linspace(0.0, DAY, 500)
        da = _make_syn(seed=1).displacement(t)
        db = _make_syn(seed=2).displacement(t)
        assert not np.allclose(da, db), "Different seeds produced identical displacement"

    def test_repeated_calls_same_instance_identical(self):
        """Calling wind_vec and displacement twice on the same instance is idempotent."""
        syn = _make_syn(seed=77)
        t = np.linspace(0.0, 5.0 * DAY, 1000)
        np.testing.assert_array_equal(syn.wind_vec(t), syn.wind_vec(t))
        np.testing.assert_array_equal(syn.displacement(t), syn.displacement(t))


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """
    t=0, large t, negative t — pin current behaviour.
    """

    def test_t0_wind_vec_finite(self):
        syn = _make_syn()
        wv = syn.wind_vec(0.0)
        assert np.all(np.isfinite(wv))

    def test_t0_displacement_zero(self):
        syn = _make_syn()
        np.testing.assert_allclose(syn.displacement(0.0), [0.0, 0.0], atol=1e-12)

    def test_large_t_wind_vec_finite(self):
        """100 game-days — no overflow or NaN."""
        syn = _make_syn()
        wv = syn.wind_vec(100.0 * DAY)
        assert np.all(np.isfinite(wv))

    def test_large_t_displacement_finite(self):
        """100 game-days — displacement is finite (no runaway)."""
        syn = _make_syn()
        d = syn.displacement(100.0 * DAY)
        assert np.all(np.isfinite(d))

    def test_negative_t_wind_vec_current_behaviour(self):
        """
        Negative t: the code uses sin/cos which are defined for all reals,
        so wind_vec(-h) should return finite values.

        PINNED: current implementation does NOT clamp/error on t<0.
        If this test starts failing it means a guard was added.
        """
        syn = _make_syn()
        wv_neg = syn.wind_vec(-1.0 * DAY)
        assert np.all(np.isfinite(wv_neg)), (
            "wind_vec(negative t) returned non-finite — behaviour changed"
        )

    def test_negative_t_displacement_current_behaviour(self):
        """
        Negative t: displacement should be finite (analytic formula still applies).

        PINNED: current implementation does NOT clamp/error on t<0.
        D(0)=(0,0) but D(-h) is non-zero by the integral formula.
        """
        syn = _make_syn()
        d_neg = syn.displacement(-1.0 * DAY)
        assert np.all(np.isfinite(d_neg)), (
            "displacement(negative t) returned non-finite — behaviour changed"
        )
        # Also verify D(-h) != (0, 0) — it's at a different position than the
        # epoch, not clamped to zero.
        assert not np.allclose(d_neg, [0.0, 0.0]), (
            "displacement(negative t) == (0,0); this would indicate clamping, "
            "which is a behaviour change from the analytic formula"
        )

    def test_wind_speed_never_zero_at_dense_probe(self):
        """
        The v_min > 0 guarantee means speed must never be zero.
        Guards against the degenerate branch in wind() (speed < 1e-9).
        """
        syn = _make_syn()
        t = np.linspace(0.0, 30.0 * DAY, 60 * 24 * 30)
        speeds = np.linalg.norm(syn.wind_vec(t), axis=1)
        assert (speeds > 1e-9).all(), "wind speed reached near-zero unexpectedly"

    def test_single_element_array_wind_vec(self):
        """np.array([t]) (single-element 1-D) returns shape (1, 2)."""
        syn = _make_syn()
        out = syn.wind_vec(np.array([3600.0]))
        assert out.shape == (1, 2)

    def test_single_element_array_displacement(self):
        """np.array([t]) returns shape (1, 2)."""
        syn = _make_syn()
        out = syn.displacement(np.array([3600.0]))
        assert out.shape == (1, 2)
