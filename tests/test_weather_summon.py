"""
tests/test_weather_summon.py — M8 summon API, Saveable delta, gust-front coupling.

Headless (no panda3d).  Covers:
- a summoned cell is placed UPWIND and drifts toward the player;
- ETA read-out math is finite/sane for an approaching cell;
- `get_delta() == {}` for pure natural weather, non-empty after a summon;
- save→load mid-storm reproduces identical future samples (load-resume invariant);
- `clear_all()` persists across `update()`;
- the GustFront modifier register/remove is balanced — no accumulation/leak;
- a legacy/garbage delta is ignored without error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core import EventBus, load_config, set_world_seed
from fire_engine.world.weather import CellKind, WeatherSystem


@pytest.fixture(autouse=True)
def _seed():
    set_world_seed(1337)


def _fresh() -> WeatherSystem:
    return WeatherSystem(load_config(), EventBus())


# ---------------------------------------------------------------------------
# Summon placement + drift
# ---------------------------------------------------------------------------


def test_summon_places_cell_upwind_and_drifts_toward_player():
    ws = _fresh()
    t0 = 2 * 86400.0 + 9.5 * 3600.0
    player = (0.0, 0.0)
    cid = ws.summon_rainstorm(time_abs=t0, player_pos=player)
    assert cid.startswith("s:")

    cell = next(c for c in ws._summoned if c.id == cid)
    # Spawn is upwind: in the OPPOSITE direction to the wind blow direction.
    (ux, uy), _ = ws.synoptic.wind(t0)
    spawn = np.array(cell.spawn_pos)
    origin = np.array(player)
    to_spawn = spawn - origin
    # The spawn vector points against the wind (negative dot with wind dir).
    assert float(np.dot(to_spawn, (ux, uy))) < 0.0

    # As time advances the cell center moves toward the player.
    d_start = float(np.hypot(*(cell.center(t0 + 1.0, ws.synoptic) - origin)))
    d_later = float(np.hypot(*(cell.center(t0 + 600.0, ws.synoptic) - origin)))
    assert d_later < d_start


def test_eta_is_finite_and_sane_for_approaching_cell():
    ws = _fresh()
    t0 = 1 * 86400.0 + 12.0 * 3600.0
    player = (0.0, 0.0)
    cid = ws.summon_thunderstorm(time_abs=t0, player_pos=player)
    cell = next(c for c in ws._summoned if c.id == cid)
    eta = ws.cell_eta_s(cell, t0 + 1.0, player)
    assert math.isfinite(eta)
    assert eta > 0.0
    # Sanity: edge distance / max plausible closing speed gives a rough lower
    # bound; the ETA should be well under the cell's lifetime.
    assert eta < cell.duration_s


# ---------------------------------------------------------------------------
# Saveable delta
# ---------------------------------------------------------------------------


def test_delta_empty_for_pure_natural_weather():
    ws = _fresh()
    ws.update(2, 9.5 * 3600.0, player_pos=(0.0, 0.0))
    assert ws.get_delta() == {}


def test_delta_nonempty_after_summon():
    ws = _fresh()
    t0 = 2 * 86400.0 + 9.5 * 3600.0
    ws.summon_rainstorm(time_abs=t0, player_pos=(0.0, 0.0))
    delta = ws.get_delta()
    assert "summoned" in delta and len(delta["summoned"]) == 1
    # All primitives — no live object refs.
    cell_d = delta["summoned"][0]
    assert isinstance(cell_d["spawn_pos"], list)
    assert all(isinstance(v, (int, float, str, list)) for v in cell_d.values())


def test_save_load_midstorm_reproduces_identical_future():
    ws = _fresh()
    t0 = 3 * 86400.0 + 8.0 * 3600.0
    player = (120.0, -40.0)
    ws.summon_thunderstorm(time_abs=t0, player_pos=player)
    ws.summon_fog_bank(time_abs=t0 + 300.0, player_pos=player)
    ws.update(3, 8.2 * 3600.0, player_pos=player)

    delta = ws.get_delta()

    # Fresh system (same seed) loads the delta and must match the future.
    ws2 = _fresh()
    ws2.apply_delta(delta)

    sample_pts = [
        ((0.0, 0.0), t0 + 1800.0),
        (player, t0 + 1200.0),
        ((300.0, 200.0), t0 + 3600.0),
    ]
    for pos, t in sample_pts:
        a = ws.sample_local(pos, t)
        b = ws2.sample_local(pos, t)
        assert a.rain_intensity == b.rain_intensity
        assert a.cloud_coverage == b.cloud_coverage
        assert a.fog_density == b.fog_density
        assert a.wind_speed == b.wind_speed

    # The would-be strike-driving cell params must round-trip bit-exact.
    orig = {c.id: c for c in ws._summoned}
    for c in ws2._summoned:
        o = orig[c.id]
        assert c.spawn_pos == o.spawn_pos
        assert c.spawn_time == o.spawn_time
        assert c.radius_m == o.radius_m
        assert c.peak_intensity == o.peak_intensity
        assert c.kind == o.kind


def test_legacy_and_garbage_delta_ignored():
    ws = _fresh()
    # Old Markov-style override delta still loads.
    ws.apply_delta({"override": "storm"})
    assert ws.current.value == "storm"

    # Garbage shapes do not crash.
    ws2 = _fresh()
    ws2.apply_delta({"summoned": [{"id": "s:0"}]})  # missing fields
    ws2.apply_delta({"summoned": "not a list"})
    ws2.apply_delta({"suppressed": 12345})
    ws2.apply_delta({})
    ws2.apply_delta({"unknown_key": [1, 2, 3]})
    # No summoned cell survived the malformed entry.
    assert ws2._summoned == []


# ---------------------------------------------------------------------------
# clear_all / suppression persistence
# ---------------------------------------------------------------------------


def test_clear_all_persists_across_update():
    ws = _fresh()
    # Find a day with natural rain so clearing is observable.
    day, tod, pos = _find_rainy_sample(ws)
    ws.update(day, tod, player_pos=pos)
    before = ws.sample_local(pos, day * 86400.0 + tod).rain_intensity
    assert before > 0.0

    ws.clear_all()
    # After clearing, the same instant reads dry — and stays dry after update().
    after = ws.sample_local(pos, day * 86400.0 + tod).rain_intensity
    assert after == 0.0
    ws.update(day, tod, player_pos=pos)
    assert ws.sample_local(pos, day * 86400.0 + tod).rain_intensity == 0.0


# ---------------------------------------------------------------------------
# GustFront coupling — register/remove balance (no leak)
# ---------------------------------------------------------------------------


class _FakeWindField:
    """Minimal stand-in for WindField: records add/remove_modifier calls."""

    def __init__(self) -> None:
        self.modifiers: list = []

    def add_modifier(self, m) -> None:
        self.modifiers.append(m)

    def remove_modifier(self, m) -> None:
        try:
            self.modifiers.remove(m)
        except ValueError:
            pass


def test_gustfront_registers_when_cell_near_and_no_leak():
    ws = _fresh()
    wf = _FakeWindField()
    ws.attach_wind_field(wf)

    t0 = 1 * 86400.0 + 10.0 * 3600.0
    player = (0.0, 0.0)
    # Summon right on top of the player so its edge is immediately in range.
    ws.summon_rainstorm(time_abs=t0, player_pos=player, upwind_m=0.0)

    # Update a second after spawn (a cell is "active" only for spawn_time < t).
    ws.update(1, 10.0 * 3600.0 + 1.0, player_pos=player)
    assert len(wf.modifiers) == 1  # front registered

    # Many updates while still near must NOT accumulate fronts.
    for k in range(2, 31):
        ws.update(1, 10.0 * 3600.0 + k, player_pos=player)
    assert len(wf.modifiers) == 1  # exactly one, no leak

    # Clearing the cell removes the front cleanly.
    ws.clear_all()
    ws.update(1, 10.0 * 3600.0 + 31.0, player_pos=player)
    assert wf.modifiers == []


def test_detach_wind_field_clears_fronts():
    ws = _fresh()
    wf = _FakeWindField()
    ws.attach_wind_field(wf)
    t0 = 1 * 86400.0 + 10.0 * 3600.0
    ws.summon_rainstorm(time_abs=t0, player_pos=(0.0, 0.0), upwind_m=0.0)
    ws.update(1, 10.0 * 3600.0 + 1.0, player_pos=(0.0, 0.0))
    assert len(wf.modifiers) == 1
    ws.attach_wind_field(None)
    assert wf.modifiers == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_rainy_sample(ws: WeatherSystem):
    """Scan a few days for a (day, tod, pos) that samples natural rain > 0."""
    for day in range(0, 12):
        cells = ws._cells_for_day(day)
        for c in cells:
            if c.kind in (CellKind.SHOWER, CellKind.THUNDERSTORM):
                t = c.spawn_time + 0.5 * c.duration_s  # mid-life (plateau)
                pos = tuple(c.center(t, ws.synoptic))
                lw = ws.sample_local(pos, t)
                if lw.rain_intensity > 0.0:
                    tod = t - day * 86400.0
                    return day, tod, (float(pos[0]), float(pos[1]))
    raise AssertionError("no natural rain found in first 12 days for seed 1337")
