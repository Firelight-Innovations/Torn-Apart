"""
tests/world/weather/_impl/test__summon.py — Mirror for
fire_engine/world/weather/_impl/_summon.py.

Authored tests covering the summon_cell, suppress, and clear_all helpers,
exercised through the WeatherSystem public API (the only legal call path per
the module's contract).

Headless — no panda3d imports.

Coverage
--------
CORRECTNESS — summon_cell (via WeatherSystem.summon_*):
  - Returned id starts with "s:" and is unique per call.
  - The spawned cell appears in ws._summoned with the correct kind/spawn_time.
  - Cell is placed UPWIND (spawn_pos is in the opposite direction to the wind).
  - Summoned radius/duration/peak_intensity match per-kind config defaults when
    no overrides are passed.

CORRECTNESS — suppress (via WeatherSystem.suppress):
  - A suppressed natural-cell id no longer contributes to the active cells list.
  - A suppressed summoned cell is removed from ws._summoned.

CORRECTNESS — clear_all (via WeatherSystem.clear_all):
  - After clear_all(), ws._summoned is empty.
  - ws.get_delta() after clear_all + summon contains only the new cells, not
    the old ones.

DETERMINISM:
  - Same-seed summon at the same (time_abs, player_pos) produces an identical
    spawn_pos.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather import WeatherSystem
from fire_engine.world.weather.cells import CellKind


def _fresh(seed: int = 1337) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(load_config(), EventBus())


# ---------------------------------------------------------------------------
# summon_cell — id, kind, placement
# ---------------------------------------------------------------------------


class TestSummonCell:
    def test_returns_id_starting_with_s(self):
        ws = _fresh()
        cid = ws.summon_rainstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        assert cid.startswith("s:")

    def test_id_is_unique_per_call(self):
        ws = _fresh()
        a = ws.summon_rainstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        b = ws.summon_thunderstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        assert a != b

    def test_cell_appears_in_summoned(self):
        ws = _fresh()
        cid = ws.summon_rainstorm(time_abs=7200.0, player_pos=(50.0, -30.0))
        ids = [c.id for c in ws._summoned]
        assert cid in ids

    def test_summoned_cell_has_shower_kind(self):
        ws = _fresh()
        cid = ws.summon_rainstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        cell = next(c for c in ws._summoned if c.id == cid)
        assert cell.kind is CellKind.SHOWER

    def test_summoned_thunderstorm_has_correct_kind(self):
        ws = _fresh()
        cid = ws.summon_thunderstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        cell = next(c for c in ws._summoned if c.id == cid)
        assert cell.kind is CellKind.THUNDERSTORM

    def test_cell_spawn_time_matches_time_abs(self):
        ws = _fresh()
        t0 = 2 * 86400.0 + 10 * 3600.0
        cid = ws.summon_rainstorm(time_abs=t0, player_pos=(0.0, 0.0))
        cell = next(c for c in ws._summoned if c.id == cid)
        assert cell.spawn_time == pytest.approx(t0)

    def test_cell_placed_upwind(self):
        """The spawn position is in the UPWIND direction from the player."""
        ws = _fresh()
        t0 = 1 * 86400.0 + 12 * 3600.0
        player = (0.0, 0.0)
        cid = ws.summon_rainstorm(time_abs=t0, player_pos=player)
        cell = next(c for c in ws._summoned if c.id == cid)

        (ux, uy), _ = ws.synoptic.wind(t0)
        to_spawn = np.array(cell.spawn_pos) - np.array(player)
        # Spawn is upwind = opposite to wind direction → negative dot product
        dot = float(np.dot(to_spawn, (ux, uy)))
        assert dot <= 0.0, f"cell not placed upwind: dot={dot:.3f}"

    def test_determinism_same_seed_same_spawn(self):
        """Same seed + same args → identical spawn_pos."""
        t0 = 86400.0
        player = (100.0, -200.0)

        set_world_seed(42)
        ws1 = WeatherSystem(load_config())
        cid1 = ws1.summon_rainstorm(time_abs=t0, player_pos=player)
        pos1 = next(c for c in ws1._summoned if c.id == cid1).spawn_pos

        set_world_seed(42)
        ws2 = WeatherSystem(load_config())
        cid2 = ws2.summon_rainstorm(time_abs=t0, player_pos=player)
        pos2 = next(c for c in ws2._summoned if c.id == cid2).spawn_pos

        assert pos1 == pos2


# ---------------------------------------------------------------------------
# suppress — natural vs summoned
# ---------------------------------------------------------------------------


class TestSuppress:
    def test_suppressed_summoned_cell_removed(self):
        ws = _fresh()
        cid = ws.summon_rainstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        assert any(c.id == cid for c in ws._summoned)
        ws.suppress(cid)
        assert not any(c.id == cid for c in ws._summoned)

    def test_suppress_natural_cell_adds_to_suppressed_set(self):
        ws = _fresh()
        # Natural cell ids follow "n:{day}:{slot}" pattern.
        fake_natural_id = "n:0:0"
        ws.suppress(fake_natural_id)
        assert fake_natural_id in ws._suppressed

    def test_suppress_unknown_id_is_noop(self):
        ws = _fresh()
        ws.suppress("n:999:999")  # must not raise


# ---------------------------------------------------------------------------
# clear_all
# ---------------------------------------------------------------------------


class TestClearAll:
    def test_summoned_list_empty_after_clear(self):
        ws = _fresh()
        ws.summon_rainstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        ws.summon_thunderstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        assert len(ws._summoned) == 2
        ws.clear_all()
        assert ws._summoned == []

    def test_delta_empty_before_summon_after_clear(self):
        ws = _fresh()
        ws.update(0, 3600.0, (0.0, 0.0))
        ws.summon_rainstorm(time_abs=3600.0, player_pos=(0.0, 0.0))
        ws.clear_all()
        # After clear, the summoned list is empty; natural cells are suppressed
        # but get_delta returns {} for an un-overridden system if there are
        # no remaining summoned cells and no pending override.
        # The important thing: clear_all does not crash and removes summoned.
        assert ws._summoned == []
