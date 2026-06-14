"""
tests/test_weather_cells.py — Storm cells and the natural spawn schedule.

No panda3d imports anywhere in this file (weather/ is a headless package).

Coverage
--------
- Cell envelope: 0 outside life, smoothstep grow/plateau/decay, peak-scaled.
- radius(t): 55 % at birth/death, full radius_m at plateau.
- contribution(): shape (N,), peak at center, decays to ~1/50 at one radius,
  zero when the cell is dead.
- center() rides D(t): a cell with no drift_bias moves exactly with the
  synoptic displacement; drift_bias bends the track linearly in time.
- Natural spawn is a pure function of (seed, day); different seeds differ;
  THUNDERSTORM radii are biased to the upper half of the band.
- No panda3d import leaks into fire_engine/world/weather/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather.cells import (
    CellKind,
    StormCell,
    day_regime,
    natural_cells,
)
from fire_engine.world.weather.synoptic import Synoptic

DAY = 24 * 3600.0


def _cell(**kw) -> StormCell:
    base = dict(
        id="s:0",
        kind=CellKind.SHOWER,
        spawn_time=1000.0,
        spawn_pos=(0.0, 0.0),
        duration_s=4000.0,
        radius_m=500.0,
        peak_intensity=0.8,
        drift_bias=(0.0, 0.0),
    )
    base.update(kw)
    return StormCell(**base)


# ---------------------------------------------------------------------------
# Envelope / intensity / radius
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_zero_outside_life(self):
        c = _cell(spawn_time=1000.0, duration_s=2000.0)
        assert c.intensity(999.0) == 0.0
        assert c.intensity(1000.0) == 0.0  # boundary is exclusive
        assert c.intensity(3000.0) == 0.0
        assert c.intensity(5000.0) == 0.0
        assert not c.active(999.0)
        assert c.active(2000.0)

    def test_plateau_is_peak(self):
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=0.8)
        # Mid-life (u = 0.5) is well inside grow (≤0.2) and decay (≥0.7).
        assert c.intensity(500.0) == pytest.approx(0.8)

    def test_grows_and_decays_monotonically(self):
        c = _cell(spawn_time=0.0, duration_s=1000.0, peak_intensity=1.0)
        grow = [c.intensity(u * 1000.0) for u in (0.02, 0.08, 0.15, 0.20)]
        assert grow == sorted(grow)  # non-decreasing through grow
        decay = [c.intensity(u * 1000.0) for u in (0.70, 0.85, 0.95, 0.99)]
        assert decay == sorted(decay, reverse=True)  # non-increasing in decay

    def test_radius_grows_to_full(self):
        c = _cell(spawn_time=0.0, duration_s=1000.0, radius_m=500.0)
        assert c.radius(5.0) == pytest.approx(0.55 * 500.0, abs=2.0)  # birth
        assert c.radius(500.0) == pytest.approx(500.0)  # plateau


# ---------------------------------------------------------------------------
# contribution()
# ---------------------------------------------------------------------------


class TestContribution:
    def test_shape_and_peak_at_center(self):
        syn = Synoptic(load_config())
        c = _cell(spawn_time=0.0, duration_s=1000.0, drift_bias=(0.0, 0.0))
        t = 500.0
        center = c.center(t, syn)
        pts = np.array([center, center + np.array([1e6, 0.0])])
        out = c.contribution(pts, t, syn)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(c.intensity(t))  # peak at the center
        assert out[1] == pytest.approx(0.0, abs=1e-9)  # far away → ~0

    def test_falls_to_one_fiftieth_at_one_radius(self):
        syn = Synoptic(load_config())
        c = _cell(spawn_time=0.0, duration_s=1000.0, drift_bias=(0.0, 0.0))
        t = 500.0
        center = c.center(t, syn)
        edge = center + np.array([c.radius(t), 0.0])
        out = c.contribution(edge[None, :], t, syn)[0]
        assert out == pytest.approx(c.intensity(t) / 50.0, rel=1e-3)

    def test_dead_cell_contributes_zero(self):
        syn = Synoptic(load_config())
        c = _cell(spawn_time=1000.0, duration_s=1000.0)
        pts = np.zeros((4, 2))
        out = c.contribution(pts, 5000.0, syn)  # well after death
        assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# center() rides the synoptic displacement
# ---------------------------------------------------------------------------


class TestTrack:
    def test_no_drift_rides_displacement_exactly(self):
        set_world_seed(1337)
        syn = Synoptic(load_config())
        spawn_t = 2000.0
        c = _cell(
            spawn_time=spawn_t, spawn_pos=(100.0, -50.0), duration_s=20000.0, drift_bias=(0.0, 0.0)
        )
        t = 8000.0
        expected = np.array([100.0, -50.0]) + syn.displacement(t) - syn.displacement(spawn_t)
        assert np.allclose(c.center(t, syn), expected)

    def test_drift_bias_bends_track_linearly(self):
        set_world_seed(1337)
        syn = Synoptic(load_config())
        spawn_t = 0.0
        drift = (2.0, -1.0)
        c0 = _cell(
            spawn_time=spawn_t, spawn_pos=(0.0, 0.0), duration_s=20000.0, drift_bias=(0.0, 0.0)
        )
        c1 = _cell(spawn_time=spawn_t, spawn_pos=(0.0, 0.0), duration_s=20000.0, drift_bias=drift)
        t = 3000.0
        diff = c1.center(t, syn) - c0.center(t, syn)
        assert np.allclose(diff, np.array(drift) * (t - spawn_t))


# ---------------------------------------------------------------------------
# Natural spawn schedule
# ---------------------------------------------------------------------------


class TestNaturalSchedule:
    def test_pure_function_of_seed_and_day(self):
        cfg = load_config()
        set_world_seed(42)
        a = {d: natural_cells(d, cfg) for d in range(6)}
        set_world_seed(42)
        b = {d: natural_cells(d, cfg) for d in range(6)}
        assert a == b

    def test_query_order_independent(self):
        """Day 5 is identical whether or not earlier days were queried."""
        cfg = load_config()
        set_world_seed(7)
        for d in range(5):
            natural_cells(d, cfg)
        after = natural_cells(5, cfg)
        set_world_seed(7)
        direct = natural_cells(5, cfg)  # jump straight to day 5
        assert after == direct

    def test_different_seed_different_cells(self):
        cfg = load_config()
        set_world_seed(1)
        a = [tuple(natural_cells(d, cfg)) for d in range(8)]
        set_world_seed(2)
        b = [tuple(natural_cells(d, cfg)) for d in range(8)]
        assert a != b

    def test_spawn_times_inside_their_day(self):
        cfg = load_config()
        set_world_seed(99)
        for d in range(10):
            for c in natural_cells(d, cfg):
                assert d * DAY <= c.spawn_time < (d + 1) * DAY
                assert c.id.startswith(f"n:{d}:")

    def test_thunderstorm_radius_biased_large(self):
        cfg = load_config()
        r_min = cfg.weather_cell_radius_min_m
        r_max = cfg.weather_cell_radius_max_m
        mid = 0.5 * (r_min + r_max)
        set_world_seed(13)
        seen = False
        for d in range(60):  # frontal days carry thunder
            for c in natural_cells(d, cfg):
                if c.kind is CellKind.THUNDERSTORM:
                    seen = True
                    assert mid <= c.radius_m <= r_max
        assert seen, "no thunderstorms in 60 days (suspicious)"

    def test_regime_is_deterministic(self):
        set_world_seed(5)
        a = [day_regime(d) for d in range(10)]
        set_world_seed(5)
        b = [day_regime(d) for d in range(10)]
        assert a == b


# ---------------------------------------------------------------------------
# Hard-rule guard: no panda3d under fire_engine/world/weather/
# ---------------------------------------------------------------------------


class TestNoPanda3D:
    def test_no_panda3d_import_in_weather_package(self):
        import ast

        weather_dir = Path(__file__).parent.parent / "fire_engine" / "world" / "weather"
        offenders = []
        for src in weather_dir.glob("*.py"):
            tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    if name.split(".")[0] in ("panda3d", "direct"):
                        offenders.append(f"{src.name}: import {name}")
        assert not offenders, f"panda3d leaked into weather/: {offenders}"
