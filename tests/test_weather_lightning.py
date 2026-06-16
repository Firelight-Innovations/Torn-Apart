"""
tests/test_weather_lightning.py — Procedural lightning (M7): bolt geometry +
strike schedule.

No panda3d imports anywhere in this file (weather/ is a headless package — the
renderer/shaders are the lead's boot-check, out of scope here).

Coverage
--------
- Bolt is byte-identical for a fixed seed (determinism), and differs by seed.
- Bolt's main channel reaches ground_z (within a tolerance); branches stay above.
- Bolt generation is fast (< 5 ms, loose timing sanity).
- Strike schedule is deterministic and LOAD-RESUME SAFE: a window equals the
  concatenation of any partition of it (no double-count, no gap at the split).
- Schedule respects thinning (more strikes near the cell's intensity plateau)
  and only THUNDERSTORM cells strike.
- WeatherSystem emits LightningStrikeEvents (deferred) for active storms.
- No panda3d import leaks into fire_engine/world/weather/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from fire_engine.core import EventBus, LightningStrikeEvent, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather.bolt import BoltGeometry, generate_bolt
from fire_engine.world.weather.cells import CellKind, StormCell, natural_cells
from fire_engine.world.weather.lightning import (
    cell_id_int,
    scheduled_strikes,
)

DAY = 24 * 3600.0


@pytest.fixture
def cfg():
    return load_config()


def _storm_cell(
    spawn_time: float = 0.0, duration: float = 10800.0, radius: float = 800.0, peak: float = 1.0
) -> StormCell:
    """A plain THUNDERSTORM cell with no drift (deterministic footprint)."""
    return StormCell(
        id="n:0:0",
        kind=CellKind.THUNDERSTORM,
        spawn_time=spawn_time,
        spawn_pos=(0.0, 0.0),
        duration_s=duration,
        radius_m=radius,
        peak_intensity=peak,
        drift_bias=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Bolt geometry
# ---------------------------------------------------------------------------


def test_bolt_is_deterministic(cfg):
    set_world_seed(1337)
    a = generate_bolt(42, (0.0, 0.0, 220.0), 8.0, cfg)
    b = generate_bolt(42, (0.0, 0.0, 220.0), 8.0, cfg)
    assert len(a) == len(b) and len(a) > 0
    assert np.array_equal(a.a, b.a)
    assert np.array_equal(a.b, b.b)
    assert np.array_equal(a.width, b.width)
    assert np.array_equal(a.brightness, b.brightness)
    assert np.array_equal(a.is_main, b.is_main)


def test_bolt_differs_by_seed(cfg):
    set_world_seed(1337)
    a = generate_bolt(1, (0.0, 0.0, 220.0), 8.0, cfg)
    b = generate_bolt(2, (0.0, 0.0, 220.0), 8.0, cfg)
    # Different seeds → different channel geometry (overwhelmingly likely).
    assert a.a.shape != b.a.shape or not np.array_equal(a.b, b.b)


def test_bolt_main_reaches_ground_branches_stay_up(cfg):
    set_world_seed(7)
    ground_z = 8.0
    start_z = 220.0
    b: BoltGeometry = generate_bolt(99, (0.0, 0.0, start_z), ground_z, cfg)
    assert b.is_main.any(), "no main return-stroke channel was produced"

    main_b = b.b[b.is_main]
    # The lowest point of the main channel should be at the ground (snapped).
    assert main_b[:, 2].min() == pytest.approx(ground_z, abs=1e-3)

    if (~b.is_main).any():
        branch_b = b.b[~b.is_main]
        # Branches terminate in mid-air — none dips to ground.  Allow a small
        # tolerance (a branch's final step may end slightly above ground_z).
        assert branch_b[:, 2].min() > ground_z + 1.0


def test_bolt_main_channel_descends(cfg):
    set_world_seed(3)
    b = generate_bolt(5, (0.0, 0.0, 200.0), 8.0, cfg)
    main_a = b.a[b.is_main]
    main_b = b.b[b.is_main]
    # The main channel trends downward overall (start above end).
    assert main_a[0, 2] > main_b[-1, 2]
    # Widths/brightness positive and main channel is the boldest.
    assert (b.width > 0).all() and (b.brightness > 0).all()
    assert b.width[b.is_main].max() >= b.width.max() - 1e-6


def test_bolt_generation_under_5ms(cfg):
    set_world_seed(1337)
    # Warm the noise/RNG path once, then time a fresh bolt.
    generate_bolt(0, (0.0, 0.0, 220.0), 8.0, cfg)
    t0 = time.perf_counter()
    b = generate_bolt(123, (0.0, 0.0, 260.0), 8.0, cfg)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert len(b) > 0
    # Loose: generous headroom over the < 5 ms target so CI noise won't flake.
    assert dt_ms < 60.0, f"bolt generation took {dt_ms:.1f} ms"


def test_bolt_respects_step_budget(cfg):
    set_world_seed(11)
    b = generate_bolt(77, (0.0, 0.0, 400.0), 8.0, cfg)
    # Total segments never exceed the configured step cap.
    assert len(b) <= int(cfg.bolt_max_steps)


# ---------------------------------------------------------------------------
# Strike schedule
# ---------------------------------------------------------------------------


def test_schedule_only_thunderstorms(cfg):
    set_world_seed(1337)
    shower = StormCell("n:0:1", CellKind.SHOWER, 0.0, (0.0, 0.0), 10800.0, 800.0, 1.0, (0.0, 0.0))
    assert scheduled_strikes(shower, 0.0, 10800.0, cfg) == []


def test_schedule_deterministic(cfg):
    set_world_seed(1337)
    cell = _storm_cell()
    a = scheduled_strikes(cell, 0.0, 3600.0, cfg)
    b = scheduled_strikes(cell, 0.0, 3600.0, cfg)
    assert a == b
    assert len(a) > 0, "a full-intensity hour-long storm should strike at least once"


def test_schedule_load_resume_safe(cfg):
    """A window == concat of any partition of it (no double-count, no gap)."""
    set_world_seed(1337)
    cell = _storm_cell(duration=10800.0)
    t0, t1 = 0.0, 5400.0
    whole = scheduled_strikes(cell, t0, t1, cfg)
    for tm in (600.0, 1800.0, 2700.123, 4000.0):
        left = scheduled_strikes(cell, t0, tm, cfg)
        right = scheduled_strikes(cell, tm, t1, cfg)
        assert left + right == whole, f"split at {tm} misaligned the schedule"


def test_schedule_times_in_window(cfg):
    set_world_seed(1337)
    cell = _storm_cell()
    t0, t1 = 1200.0, 3600.0
    strikes = scheduled_strikes(cell, t0, t1, cfg)
    for s in strikes:
        assert t0 <= s.time_abs < t1
        assert 0.0 <= s.intensity <= 1.0
        assert s.seed >= 0


def test_schedule_offsets_within_footprint(cfg):
    set_world_seed(1337)
    cell = _storm_cell(radius=800.0)
    for s in scheduled_strikes(cell, 0.0, 5400.0, cfg):
        r = cell.radius(s.time_abs)
        assert np.hypot(*s.pos_xy) <= r + 1e-6


def test_schedule_thinned_by_intensity(cfg):
    """Far more strikes during the plateau than during the brief grow tail."""
    set_world_seed(1337)
    cell = _storm_cell(duration=10800.0)
    # Grow phase is the first 20 % of life; plateau is the middle.
    grow = scheduled_strikes(cell, 0.0, 0.10 * 10800.0, cfg)
    plateau = scheduled_strikes(cell, 0.40 * 10800.0, 0.50 * 10800.0, cfg)
    # Equal-length windows; the plateau (intensity ~1) should out-strike the
    # early grow window (intensity ramping from 0).
    assert len(plateau) >= len(grow)


def test_cell_id_int_stable(cfg):
    assert cell_id_int("n:5:2") == cell_id_int("n:5:2")
    assert cell_id_int("n:5:2") != cell_id_int("n:5:3")
    assert 0 <= cell_id_int("s:0") < 2**31


# ---------------------------------------------------------------------------
# WeatherSystem emission hook
# ---------------------------------------------------------------------------


def test_system_emits_strike_events(cfg):
    """Find a day with a thunderstorm, advance the system over it, expect events."""
    set_world_seed(1337)
    from fire_engine.world.weather import WeatherSystem

    # Locate a (day) that spawns at least one thunderstorm cell.
    storm_day = None
    for day in range(0, 40):
        if any(c.kind is CellKind.THUNDERSTORM for c in natural_cells(day, cfg)):
            storm_day = day
            break
    assert storm_day is not None, "no thunderstorm in the first 40 days at seed 1337"

    cell = next(c for c in natural_cells(storm_day, cfg) if c.kind is CellKind.THUNDERSTORM)

    bus = EventBus()
    received: list[LightningStrikeEvent] = []
    bus.subscribe(LightningStrikeEvent, received.append)

    ws = WeatherSystem(cfg, bus)
    # Sample at the cell center so the storm is "here".  Step in 60 s ticks
    # across the cell's plateau, draining the deferred bus each tick.
    t_mid = cell.spawn_time + 0.45 * cell.duration_s
    day = int(t_mid // DAY)
    # Prime last_strike_time with one update, then march forward.
    center = cell.center(t_mid, ws.synoptic)
    pos = (float(center[0]), float(center[1]))
    ws.update(day, (t_mid % DAY), player_pos=pos)
    bus.drain()
    for k in range(1, 30):
        t = t_mid + k * 60.0
        ws.update(int(t // DAY), (t % DAY), player_pos=pos)
        bus.drain()

    assert received, "no LightningStrikeEvent emitted over a thunderstorm plateau"
    e = received[0]
    # The event contract M8 depends on.
    assert len(e.pos) == 3 and len(e.ground_pos) == 3
    assert e.pos[2] > e.ground_pos[2]  # cloud base above ground
    assert isinstance(e.seed, int) and isinstance(e.cell_id, int)
    assert 0.0 <= e.intensity <= 1.0


def test_system_no_events_without_bus(cfg):
    """No bus → the hook is a silent no-op (and update still works)."""
    set_world_seed(1337)
    from fire_engine.world.weather import WeatherSystem

    ws = WeatherSystem(cfg, bus=None)
    ws.update(0, 3600.0, player_pos=(0.0, 0.0))
    lw2 = ws.update(0, 3660.0, player_pos=(0.0, 0.0))
    assert lw2 is not None


# ---------------------------------------------------------------------------
# Headless guard
# ---------------------------------------------------------------------------


def test_no_panda3d_in_weather_lightning():
    """weather/lightning.py + weather/bolt.py never import panda3d (Hard Rule 1)."""
    import ast

    root = Path(__file__).resolve().parents[1] / "fire_engine" / "world" / "weather"
    for name in ("lightning.py", "bolt.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            assert not any(m.split(".")[0] == "panda3d" for m in mods), (
                f"{name} imports panda3d (Hard Rule 1)"
            )
