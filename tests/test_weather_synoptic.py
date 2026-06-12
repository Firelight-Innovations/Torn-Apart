"""
tests/test_weather_synoptic.py — Synoptic flow (fire_engine/weather/synoptic.py).

Covers the M1 contract:
* determinism — same seed → identical flow across instances; different seed
  → different flow,
* speed band — |W(t)| guaranteed inside the configured [min, max],
* smooth drift — direction never jumps (bounded angular rate),
* consistency — displacement D(t) is the exact integral of the wind
  (finite-difference check),
* sky integration — WeatherSystem wind is synoptic-driven, unit-length and
  continuous across segment/midnight boundaries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.sky.weather import WeatherSystem
from fire_engine.weather.synoptic import Synoptic

DAY = 86400.0


def _make_syn(seed: int = 1337, **overrides) -> Synoptic:
    set_world_seed(seed)
    return Synoptic(Config(**overrides))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical_flow(self):
        t = np.linspace(0.0, 10.0 * DAY, 5000)
        a = _make_syn(seed=1337)
        wa, da = a.wind_vec(t), a.displacement(t)
        b = _make_syn(seed=1337)
        wb, db = b.wind_vec(t), b.displacement(t)
        np.testing.assert_array_equal(wa, wb)
        np.testing.assert_array_equal(da, db)

    def test_different_seed_different_flow(self):
        t = np.linspace(0.0, DAY, 100)
        wa = _make_syn(seed=1).wind_vec(t)
        wb = _make_syn(seed=2).wind_vec(t)
        assert not np.allclose(wa, wb)

    def test_scalar_matches_vectorized(self):
        syn = _make_syn()
        ts = [0.0, 12345.6, 3.0 * DAY + 7.5 * 3600.0]
        vec = syn.wind_vec(np.array(ts))
        for i, t in enumerate(ts):
            np.testing.assert_allclose(syn.wind_vec(t), vec[i], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Speed band + smooth drift
# ---------------------------------------------------------------------------

class TestBandAndSmoothness:
    def test_speed_within_configured_band(self):
        syn = _make_syn()
        cfg = Config()
        t = np.linspace(0.0, 30.0 * DAY, 30 * 24 * 60)      # 1 game-min steps
        speed = np.linalg.norm(syn.wind_vec(t), axis=1)
        assert speed.min() >= cfg.weather_synoptic_speed_min_ms - 1e-9
        assert speed.max() <= cfg.weather_synoptic_speed_max_ms + 1e-9

    def test_direction_actually_drifts(self):
        # The whole point: direction must move over hours (not a constant).
        syn = _make_syn()
        t = np.linspace(0.0, 3.0 * DAY, 3 * 24 * 12)        # 5-min steps
        w = syn.wind_vec(t)
        ang = np.arctan2(w[:, 1], w[:, 0])
        swing = np.ptp(np.unwrap(ang))
        assert swing > math.radians(20.0), (
            f"direction swing over 3 days only {math.degrees(swing):.1f} deg"
        )

    def test_no_direction_jumps(self):
        # Angular rate bound: < 15 degrees per game minute, everywhere.
        syn = _make_syn()
        t = np.linspace(0.0, 10.0 * DAY, 10 * 24 * 60)
        w = syn.wind_vec(t)
        ang = np.unwrap(np.arctan2(w[:, 1], w[:, 0]))
        step = np.abs(np.diff(ang))
        assert step.max() < math.radians(15.0)

    def test_speed_changes_smoothly(self):
        syn = _make_syn()
        t = np.linspace(0.0, 10.0 * DAY, 10 * 24 * 60)
        speed = np.linalg.norm(syn.wind_vec(t), axis=1)
        assert np.abs(np.diff(speed)).max() < 0.5            # m/s per game min


# ---------------------------------------------------------------------------
# D(t) is the exact integral of W(t)
# ---------------------------------------------------------------------------

class TestDisplacementConsistency:
    def test_displacement_zero_at_origin(self):
        syn = _make_syn()
        np.testing.assert_allclose(syn.displacement(0.0), [0.0, 0.0], atol=1e-9)

    def test_dDdt_equals_wind(self):
        syn = _make_syn()
        h = 1.0                                              # game seconds
        for t in [0.0, 3600.0, DAY * 2 + 4321.0, DAY * 9.7]:
            fd = (syn.displacement(t + h) - syn.displacement(t - h)) / (2.0 * h)
            np.testing.assert_allclose(fd, syn.wind_vec(t), atol=1e-4)

    def test_displacement_accumulates(self):
        # Over a day the air mass must actually go somewhere (speed >= 1.5
        # m/s guarantees >= ~130 km of travel; ripple can't cancel the
        # prevailing term).
        syn = _make_syn()
        d = syn.displacement(DAY) - syn.displacement(0.0)
        assert np.linalg.norm(d) > 10_000.0


# ---------------------------------------------------------------------------
# WeatherSystem integration
# ---------------------------------------------------------------------------

class TestSkyIntegration:
    def _make_ws(self, seed: int = 1337) -> WeatherSystem:
        set_world_seed(seed)
        return WeatherSystem(Config())

    def test_wind_dir_is_synoptic_dir(self):
        ws = self._make_ws()
        # Sample mid-segment (past any blend window) so params sit at the
        # current target, whose direction must be the synoptic direction.
        day, tod = 2, 5.0 * 3600.0
        p = ws.update(day, tod)
        (ux, uy), _speed = ws.synoptic.wind(day * DAY + tod)
        assert math.isclose(p.wind_dir[0], ux, abs_tol=1e-9)
        assert math.isclose(p.wind_dir[1], uy, abs_tol=1e-9)

    def test_wind_unit_and_positive(self):
        ws = self._make_ws()
        for day in range(2):
            for i in range(48):
                p = ws.update(day, i * 1800.0)
                assert abs(math.hypot(*p.wind_dir) - 1.0) < 1e-6
                assert p.wind_speed > 0.0

    def test_wind_continuous_across_midnight(self):
        # The old per-day wind snapped direction at midnight (hidden by the
        # blend); synoptic wind must be continuous through it outright.
        ws = self._make_ws()
        p1 = ws.update(1, DAY - 30.0)
        p2 = ws.update(2, 30.0)
        d_ang = abs(
            math.atan2(p1.wind_dir[1], p1.wind_dir[0])
            - math.atan2(p2.wind_dir[1], p2.wind_dir[0])
        )
        d_ang = min(d_ang, 2.0 * math.pi - d_ang)
        assert d_ang < math.radians(2.0)

    def test_update_deterministic_across_instances(self):
        a = self._make_ws(seed=42)
        pa = [a.update(1, i * 600.0) for i in range(144)]
        b = self._make_ws(seed=42)
        pb = [b.update(1, i * 600.0) for i in range(144)]
        assert pa == pb


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_invalid_band_raises(self):
        set_world_seed(0)
        with pytest.raises(ValueError):
            Synoptic(Config(
                weather_synoptic_speed_min_ms=5.0,
                weather_synoptic_speed_max_ms=3.0,
            ))
