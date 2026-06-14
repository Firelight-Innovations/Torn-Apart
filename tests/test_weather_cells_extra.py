"""
tests/test_weather_cells_extra.py — Golden-master / pin-down characterization
tests for weather/cells.py.

Coverage gaps addressed (not in test_weather_cells.py):
- regime_ambient: golden mapping for all three Regime members + range checks.
- contribution: zero beyond one radius; peak near center during plateau; shape
  matches the leading dimension of the input array; vectorised (no per-element
  loops in the test assertions).
- active: boundary inclusivity at exactly spawn_time and spawn_time+duration.
- intensity envelope: 0 outside lifetime; fraction pins for grow/plateau/decay
  transition points; birth/death clamp behaviour.
- day_regime: deterministic across seed resets; spans more than one Regime
  value across many days (not all one value).
- natural_cells: ids and kinds are stable across re-draws of the same day+seed;
  different days produce different cells; peak_intensity band; drift_bias
  magnitude; spawn_pos inside the declared domain.

Headless only.  No panda3d imports.  Fixed seed throughout.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather.cells import (
    CellKind,
    Regime,
    StormCell,
    day_regime,
    natural_cells,
    regime_ambient,
)
from fire_engine.world.weather.synoptic import Synoptic

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DAY = 24 * 3600.0
_FIXED_SEED = 2025


def _cell(**kw) -> StormCell:
    """Minimal StormCell factory — mirrors test_weather_cells.py style."""
    base = dict(
        id="s:0",
        kind=CellKind.SHOWER,
        spawn_time=0.0,
        spawn_pos=(0.0, 0.0),
        duration_s=4000.0,
        radius_m=500.0,
        peak_intensity=0.8,
        drift_bias=(0.0, 0.0),
    )
    base.update(kw)
    return StormCell(**base)


# ---------------------------------------------------------------------------
# regime_ambient golden mapping
# ---------------------------------------------------------------------------

class TestRegimeAmbient:
    """Pin the exact (coverage, density) pair returned for each Regime."""

    # Golden values read from _REGIME_AMBIENT in cells.py — pinning current
    # behaviour so any silent edit trips the test.
    GOLDEN = {
        Regime.HIGH_PRESSURE: (0.08, 0.30),
        Regime.MIXED:         (0.40, 0.52),
        Regime.FRONTAL:       (0.75, 0.72),
    }

    @pytest.mark.parametrize("regime", list(Regime))
    def test_golden_values(self, regime):
        cov, den = regime_ambient(regime)
        exp_cov, exp_den = self.GOLDEN[regime]
        assert cov == pytest.approx(exp_cov)
        assert den == pytest.approx(exp_den)

    @pytest.mark.parametrize("regime", list(Regime))
    def test_coverage_in_range(self, regime):
        cov, _ = regime_ambient(regime)
        assert 0.0 <= cov <= 1.0

    @pytest.mark.parametrize("regime", list(Regime))
    def test_density_in_range(self, regime):
        _, den = regime_ambient(regime)
        assert 0.0 <= den <= 1.0

    @pytest.mark.parametrize("regime", list(Regime))
    def test_both_finite(self, regime):
        cov, den = regime_ambient(regime)
        assert math.isfinite(cov)
        assert math.isfinite(den)

    def test_all_regimes_covered(self):
        """regime_ambient must not raise for any Regime member."""
        for r in Regime:
            regime_ambient(r)  # must not raise

    def test_high_pressure_below_mixed_coverage(self):
        """HIGH_PRESSURE coverage < MIXED coverage < FRONTAL coverage."""
        hp_cov, _ = regime_ambient(Regime.HIGH_PRESSURE)
        mx_cov, _ = regime_ambient(Regime.MIXED)
        fr_cov, _ = regime_ambient(Regime.FRONTAL)
        assert hp_cov < mx_cov < fr_cov

    def test_high_pressure_classifies_as_clear(self):
        """HIGH_PRESSURE ambient sits inside the CLEAR bucket (coverage < 0.3)."""
        cov, _ = regime_ambient(Regime.HIGH_PRESSURE)
        assert cov < 0.3

    def test_frontal_classifies_as_overcast(self):
        """FRONTAL ambient sits inside the OVERCAST bucket (coverage > 0.7)."""
        cov, _ = regime_ambient(Regime.FRONTAL)
        assert cov > 0.7


# ---------------------------------------------------------------------------
# active() boundary inclusivity
# ---------------------------------------------------------------------------

class TestActiveBoundary:
    """
    Pin the boundary behaviour of active():
      cells.py line 213: `return 0.0 < (t - self.spawn_time) < self.duration_s`
    Both endpoints are EXCLUSIVE — suspect / pin.
    """

    def test_exactly_at_spawn_time_is_inactive(self):
        c = _cell(spawn_time=1000.0, duration_s=2000.0)
        # u = 0.0 → strictly-less check: 0 < 0 is False
        assert c.active(1000.0) is False

    def test_just_after_spawn_time_is_active(self):
        c = _cell(spawn_time=1000.0, duration_s=2000.0)
        assert c.active(1000.0 + 1e-9) is True

    def test_exactly_at_end_is_inactive(self):
        c = _cell(spawn_time=1000.0, duration_s=2000.0)
        # spawn_time + duration = 3000; u = 1.0 → strictly-less: False
        assert c.active(3000.0) is False

    def test_just_before_end_is_active(self):
        c = _cell(spawn_time=1000.0, duration_s=2000.0)
        assert c.active(3000.0 - 1e-9) is True

    def test_well_before_spawn_is_inactive(self):
        c = _cell(spawn_time=5000.0, duration_s=2000.0)
        assert c.active(0.0) is False

    def test_well_after_end_is_inactive(self):
        c = _cell(spawn_time=0.0, duration_s=2000.0)
        assert c.active(1e9) is False


# ---------------------------------------------------------------------------
# intensity() envelope — birth/death fraction pins
# ---------------------------------------------------------------------------

class TestIntensityEnvelope:
    """
    The envelope is smoothstep(u,0,0.2)*[1-smoothstep(u,0.7,1.0)].
    Pin the values at the ramp transition points.
    """

    def test_intensity_zero_at_spawn(self):
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=1.0)
        # u=0 → outside life → 0
        assert c.intensity(0.0) == 0.0

    def test_intensity_zero_at_end(self):
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=1.0)
        assert c.intensity(1000.0) == 0.0

    def test_intensity_zero_before_spawn(self):
        c = _cell(spawn_time=500.0, duration_s=1000.0, peak_intensity=1.0)
        assert c.intensity(0.0) == 0.0

    def test_intensity_zero_after_end(self):
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=1.0)
        assert c.intensity(2000.0) == 0.0

    def test_intensity_at_grow_midpoint(self):
        """u=0.10 is mid-grow (smoothstep 0→0.2) → grow < 1, decay = 1."""
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=1.0)
        v = c.intensity(100.0)  # u=0.10
        # Must be strictly between 0 and peak_intensity, not yet plateau.
        assert 0.0 < v < 1.0

    def test_intensity_at_grow_end_is_peak(self):
        """u=0.20 is exactly end of grow: smoothstep(0.2,0,0.2)=1 → plateau."""
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=0.9)
        v = c.intensity(200.0)  # u=0.20
        assert v == pytest.approx(0.9, abs=1e-9)

    def test_intensity_plateau_is_peak(self):
        """u=0.45 (mid-plateau) → both grow=1 and decay=1 → peak."""
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=0.75)
        v = c.intensity(450.0)
        assert v == pytest.approx(0.75)

    def test_intensity_at_decay_start_is_peak(self):
        """u=0.70 is exactly start of decay: 1-smoothstep(0.7,0.7,1.0)=1 → peak."""
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=1.0)
        v = c.intensity(700.0)  # u=0.70
        assert v == pytest.approx(1.0, abs=1e-9)

    def test_intensity_at_decay_midpoint(self):
        """u=0.85 is mid-decay → intensity between 0 and peak."""
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=1.0)
        v = c.intensity(850.0)
        assert 0.0 < v < 1.0

    def test_intensity_scales_with_peak(self):
        """intensity is linearly proportional to peak_intensity."""
        c1 = _cell(peak_intensity=0.5, spawn_time=0.0, duration_s=1000.0)
        c2 = _cell(peak_intensity=1.0, spawn_time=0.0, duration_s=1000.0)
        t = 450.0
        assert c1.intensity(t) == pytest.approx(c2.intensity(t) * 0.5)


# ---------------------------------------------------------------------------
# contribution() — vectorised position array behaviour
# ---------------------------------------------------------------------------

class TestContributionVectorised:
    """
    contribution(points_xy, t, syn) must:
      - return an array whose shape == (N,) when input is (N, 2)
      - peak at the cell center during plateau
      - be ~zero at positions beyond one cell radius from center
      - be exactly zero when the cell is inactive
    """

    def setup_method(self):
        set_world_seed(_FIXED_SEED)
        self.syn = Synoptic(load_config())
        self.c = _cell(
            spawn_time=0.0, duration_s=4000.0, radius_m=500.0,
            peak_intensity=0.8, drift_bias=(0.0, 0.0),
        )
        self.t = 2000.0  # plateau (u=0.5)

    def test_output_shape_matches_input_leading_dim(self):
        pts = np.zeros((7, 2))
        out = self.c.contribution(pts, self.t, self.syn)
        assert out.shape == (7,)

    def test_single_point_returns_length_1(self):
        pts = np.zeros((1, 2))
        out = self.c.contribution(pts, self.t, self.syn)
        assert out.shape == (1,)

    def test_peak_at_center_equals_intensity(self):
        center = self.c.center(self.t, self.syn)
        out = self.c.contribution(center[None, :], self.t, self.syn)
        assert out[0] == pytest.approx(self.c.intensity(self.t), rel=1e-9)

    def test_near_zero_at_far_positions(self):
        """Points 50 × radius away should contribute essentially nothing."""
        center = self.c.center(self.t, self.syn)
        r = self.c.radius(self.t)
        far_pts = np.array([
            center + np.array([50.0 * r, 0.0]),
            center + np.array([0.0, 50.0 * r]),
            center + np.array([-50.0 * r, 0.0]),
        ])
        out = self.c.contribution(far_pts, self.t, self.syn)
        assert np.all(out < 1e-9)

    def test_contributions_decrease_with_distance(self):
        """Radial samples must be monotonically decreasing away from center."""
        center = self.c.center(self.t, self.syn)
        distances = np.array([0.0, 0.2, 0.5, 1.0, 2.0]) * self.c.radius(self.t)
        pts = np.column_stack([center[0] + distances,
                               np.full(len(distances), center[1])])
        out = self.c.contribution(pts, self.t, self.syn)
        # Each successive value must be <= the previous (non-increasing).
        assert np.all(np.diff(out) <= 0.0)

    def test_value_at_one_radius_approx_one_fiftieth(self):
        center = self.c.center(self.t, self.syn)
        edge = center + np.array([self.c.radius(self.t), 0.0])
        out = self.c.contribution(edge[None, :], self.t, self.syn)[0]
        expected = self.c.intensity(self.t) / 50.0
        assert out == pytest.approx(expected, rel=1e-3)

    def test_dead_cell_all_zeros(self):
        """A dead cell returns an all-zero array regardless of positions."""
        c = _cell(spawn_time=0.0, duration_s=1000.0)
        pts = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [-200.0, 50.0],
        ])
        out = c.contribution(pts, 9999.0, self.syn)
        assert out.shape == (3,)
        assert np.all(out == 0.0)

    def test_all_values_finite_and_nonneg(self):
        """contribution must always return finite non-negative values."""
        center = self.c.center(self.t, self.syn)
        r = self.c.radius(self.t)
        pts = np.array([
            center,
            center + [r, 0.0],
            center + [0.0, r],
            center + [10 * r, 0.0],
        ])
        out = self.c.contribution(pts, self.t, self.syn)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# day_regime: determinism + coverage across many days
# ---------------------------------------------------------------------------

class TestDayRegimeDeterminism:

    def test_same_seed_same_day_same_regime(self):
        set_world_seed(_FIXED_SEED)
        results_a = [day_regime(d) for d in range(30)]
        set_world_seed(_FIXED_SEED)
        results_b = [day_regime(d) for d in range(30)]
        assert results_a == results_b

    def test_different_seed_can_differ(self):
        set_world_seed(1)
        a = [day_regime(d) for d in range(20)]
        set_world_seed(2)
        b = [day_regime(d) for d in range(20)]
        # With different seeds the sequences may differ.
        assert a != b

    def test_not_all_same_regime_over_many_days(self):
        """Over 60 days we must see at least 2 distinct regimes."""
        set_world_seed(_FIXED_SEED)
        seen = {day_regime(d) for d in range(60)}
        assert len(seen) >= 2, f"Only one regime seen in 60 days: {seen}"

    def test_all_three_regimes_seen_over_many_days(self):
        """Across 200 days all three Regime values should appear at least once."""
        set_world_seed(_FIXED_SEED)
        seen = {day_regime(d) for d in range(200)}
        assert seen == set(Regime), f"Missing regime(s): {set(Regime) - seen}"

    def test_returns_regime_member(self):
        set_world_seed(_FIXED_SEED)
        for d in range(10):
            r = day_regime(d)
            assert isinstance(r, Regime)

    def test_query_order_independent(self):
        """day_regime(5) must be the same whether or not days 0-4 were queried."""
        set_world_seed(_FIXED_SEED)
        for d in range(5):
            day_regime(d)
        after = day_regime(5)

        set_world_seed(_FIXED_SEED)
        direct = day_regime(5)
        assert after == direct


# ---------------------------------------------------------------------------
# natural_cells: determinism + inter-day variability
# ---------------------------------------------------------------------------

class TestNaturalCellsDeterminism:

    def test_same_seed_same_day_same_ids(self):
        cfg = load_config()
        set_world_seed(_FIXED_SEED)
        a = natural_cells(3, cfg)
        set_world_seed(_FIXED_SEED)
        b = natural_cells(3, cfg)
        assert [c.id for c in a] == [c.id for c in b]

    def test_same_seed_same_day_same_kinds(self):
        cfg = load_config()
        set_world_seed(_FIXED_SEED)
        a = natural_cells(3, cfg)
        set_world_seed(_FIXED_SEED)
        b = natural_cells(3, cfg)
        assert [c.kind for c in a] == [c.kind for c in b]

    def test_same_seed_same_day_same_spawn_pos(self):
        cfg = load_config()
        set_world_seed(_FIXED_SEED)
        a = natural_cells(3, cfg)
        set_world_seed(_FIXED_SEED)
        b = natural_cells(3, cfg)
        a_pos = [(c.spawn_pos[0], c.spawn_pos[1]) for c in a]
        b_pos = [(c.spawn_pos[0], c.spawn_pos[1]) for c in b]
        assert a_pos == b_pos

    def test_different_days_differ(self):
        """Days 0..9 should not all produce identical cell lists."""
        cfg = load_config()
        set_world_seed(_FIXED_SEED)
        lists = [tuple(c.id for c in natural_cells(d, cfg)) for d in range(10)]
        assert len(set(lists)) > 1, "All 10 days produced identical cell lists"

    def test_peak_intensity_in_band(self):
        """natural_cells draws peak from uniform(0.6, 1.0)."""
        cfg = load_config()
        set_world_seed(_FIXED_SEED)
        intensities = []
        for d in range(20):
            for c in natural_cells(d, cfg):
                intensities.append(c.peak_intensity)
        if not intensities:
            pytest.skip("No cells spawned in 20 days — seed is too calm")
        assert all(0.6 <= v <= 1.0 for v in intensities)

    def test_drift_bias_magnitude_in_band(self):
        """natural_cells draws drift_mag from uniform(0, 0.6)."""
        cfg = load_config()
        set_world_seed(_FIXED_SEED)
        for d in range(20):
            for c in natural_cells(d, cfg):
                mag = math.hypot(c.drift_bias[0], c.drift_bias[1])
                assert 0.0 <= mag <= 0.6, f"drift magnitude {mag} outside [0, 0.6]"

    def test_spawn_pos_inside_domain(self):
        """spawn_pos must be within ±weather_domain_m of origin."""
        cfg = load_config()
        domain = cfg.weather_domain_m
        set_world_seed(_FIXED_SEED)
        for d in range(20):
            for c in natural_cells(d, cfg):
                assert -domain <= c.spawn_pos[0] <= domain
                assert -domain <= c.spawn_pos[1] <= domain

    def test_high_pressure_day_can_be_empty(self):
        """
        On a HIGH_PRESSURE day (spawn_prob=0.12) it's plausible to see 0 cells.
        We simply check the call doesn't raise and returns a list.
        (Actual emptiness is probabilistic and seed-dependent.)
        """
        cfg = load_config()
        # Find a HIGH_PRESSURE day in first 50 days.
        from fire_engine.world.weather.cells import Regime
        set_world_seed(_FIXED_SEED)
        for d in range(50):
            if day_regime(d) is Regime.HIGH_PRESSURE:
                set_world_seed(_FIXED_SEED)
                result = natural_cells(d, cfg)
                assert isinstance(result, list)
                return
        pytest.skip("No HIGH_PRESSURE day found in 50 days")
