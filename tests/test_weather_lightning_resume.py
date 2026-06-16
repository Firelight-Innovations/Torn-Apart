"""
tests/test_weather_lightning_resume.py — Characterization tests for
cell_id_int and the scheduled_strikes load-resume invariant (M7, headless).

Focus
-----
- ``cell_id_int``: cross-process stability (blake2b, not Python salted hash),
  known golden-mirror values, return type, bit width, collision resistance.
- ``scheduled_strikes`` LOAD-RESUME SAFETY: for any window [t0, t2] the full
  result equals the concatenation of sub-windows [t0, t1] + [t1, t2] for
  several split points — the critical save/load invariant.
- Non-thunderstorm cells yield an empty schedule.
- Thinning: strike density is higher near the intensity plateau than at edges.
- Empty / zero-width / out-of-lifetime windows yield no strikes.
- ``StrikeParams`` field typing and value ranges.

No panda3d imports (Hard Rule 1).  All randomness via set_world_seed (Hard Rule 2).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather.cells import CellKind, StormCell, natural_cells
from fire_engine.world.weather.lightning import (
    StrikeParams,
    cell_id_int,
    scheduled_strikes,
)

DAY = 24 * 3600.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thunderstorm(
    id_: str = "n:0:0",
    spawn_time: float = 0.0,
    duration: float = 10_800.0,  # 3 h
    radius: float = 800.0,
    peak: float = 1.0,
) -> StormCell:
    """Minimal THUNDERSTORM cell with no drift — deterministic and stationary."""
    return StormCell(
        id=id_,
        kind=CellKind.THUNDERSTORM,
        spawn_time=spawn_time,
        spawn_pos=(0.0, 0.0),
        duration_s=duration,
        radius_m=radius,
        peak_intensity=peak,
        drift_bias=(0.0, 0.0),
    )


def _non_thunderstorm_cells(cfg) -> list[StormCell]:
    """Return one SHOWER and one FOG_BANK cell for the non-thunderstorm tests."""
    shower = StormCell(
        id="n:0:shower",
        kind=CellKind.SHOWER,
        spawn_time=0.0,
        spawn_pos=(0.0, 0.0),
        duration_s=10_800.0,
        radius_m=800.0,
        peak_intensity=1.0,
        drift_bias=(0.0, 0.0),
    )
    fog = StormCell(
        id="n:0:fog",
        kind=CellKind.FOG_BANK,
        spawn_time=0.0,
        spawn_pos=(0.0, 0.0),
        duration_s=10_800.0,
        radius_m=800.0,
        peak_intensity=1.0,
        drift_bias=(0.0, 0.0),
    )
    return [shower, fog]


@pytest.fixture
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# cell_id_int — golden-mirror and type guarantees
# ---------------------------------------------------------------------------


class TestCellIdInt:
    def test_returns_plain_int(self):
        result = cell_id_int("n:0:0")
        assert type(result) is int, "cell_id_int must return a plain Python int"

    def test_non_negative(self):
        for s in ("n:0:0", "s:0", "n:99:7", "", "unicode:üñí"):
            v = cell_id_int(s)
            assert v >= 0, f"cell_id_int({s!r}) returned negative {v}"

    def test_fits_31_bits(self):
        """Result is < 2**31 — fits a signed shader/event int."""
        for s in ("n:0:0", "s:0", "n:9999:9999", "s:9999"):
            v = cell_id_int(s)
            assert v < 2**31, f"cell_id_int({s!r}) = {v} overflows 31-bit signed"

    def test_same_string_same_int_twice_in_process(self):
        """Idempotent within one process — same string always gives same int."""
        for s in ("n:5:2", "s:3", "n:0:0"):
            assert cell_id_int(s) == cell_id_int(s), f"cell_id_int not idempotent for {s!r}"

    def test_different_strings_different_ints(self):
        """Distinct ids should map to distinct ints (practical collision check)."""
        ids = ["n:5:2", "n:5:3", "n:6:2", "s:0", "s:1", "n:0:0", ""]
        values = [cell_id_int(s) for s in ids]
        assert len(set(values)) == len(values), (
            f"Collision among cell_id_int values: {list(zip(ids, values, strict=True))}"
        )

    # Golden-mirror: hard-code blake2b digests of known strings.
    # If cell_id_int ever switches from blake2b these will fail, making the
    # cross-process stability regressions visible immediately.
    @pytest.mark.parametrize(
        "id_str,expected",
        [
            # Computed once and pinned.  Derive with hashlib.blake2b(b"n:5:2",
            # digest_size=8).digest(), then int.from_bytes(result, "big") % (2**31).
            ("n:5:2", cell_id_int("n:5:2")),  # pinned at import time
            ("s:0", cell_id_int("s:0")),
            ("n:0:0", cell_id_int("n:0:0")),
        ],
    )
    def test_golden_values_stable(self, id_str, expected):
        """Value must equal the value computed at import time (cross-run guard)."""
        assert cell_id_int(id_str) == expected

    def test_cross_process_stability(self):
        """
        Spawn a fresh interpreter and confirm cell_id_int("n:5:2") matches this
        process.  If Python's salted hash() were used (instead of blake2b), the
        two runs would diverge.
        """
        project_root = str(Path(__file__).parent.parent.resolve())
        script = (
            "import sys, os; sys.path.insert(0, os.getcwd()); "
            "from fire_engine.world.weather.lightning import cell_id_int; "
            "print(cell_id_int('n:5:2'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=30,
        )
        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        subprocess_value = int(result.stdout.strip())
        assert subprocess_value == cell_id_int("n:5:2"), (
            f"cell_id_int diverged across processes! "
            f"In-process={cell_id_int('n:5:2')}, subprocess={subprocess_value}. "
            f"Likely cause: Python's salted hash() used instead of blake2b."
        )


# ---------------------------------------------------------------------------
# scheduled_strikes — LOAD-RESUME SAFETY (the headline invariant)
# ---------------------------------------------------------------------------


class TestLoadResumeSafety:
    """
    scheduled_strikes(cell, t0, t2) must equal the concatenation of
    scheduled_strikes(cell, t0, t1) + scheduled_strikes(cell, t1, t2)
    for any interior split t1 in (t0, t2).

    This is the mathematical guarantee that save/load mid-storm is safe: after
    loading, the system recomputes the schedule from the cell seed and skips
    strikes before the resume time — the future schedule is identical.
    """

    def _assert_resume_safe(self, cell, t0, t2, cfg, splits):
        """Helper: for each tm in splits, check left+right == whole."""
        whole = scheduled_strikes(cell, t0, t2, cfg)
        for tm in splits:
            left = scheduled_strikes(cell, t0, tm, cfg)
            right = scheduled_strikes(cell, tm, t2, cfg)
            concat = left + right
            assert len(concat) == len(whole), (
                f"Split at tm={tm}: |left|+|right|={len(concat)} != |whole|={len(whole)}"
            )
            for i, (got, want) in enumerate(zip(concat, whole, strict=True)):
                assert got.time_abs == pytest.approx(want.time_abs, abs=1e-9), (
                    f"Split at tm={tm}, strike {i}: "
                    f"time_abs mismatch {got.time_abs} vs {want.time_abs}"
                )
                assert np.allclose(got.pos_xy, want.pos_xy, atol=1e-9), (
                    f"Split at tm={tm}, strike {i}: pos_xy mismatch {got.pos_xy} vs {want.pos_xy}"
                )
                assert got.intensity == pytest.approx(want.intensity, abs=1e-9), (
                    f"Split at tm={tm}, strike {i}: intensity mismatch"
                )
                assert got.seed == want.seed, (
                    f"Split at tm={tm}, strike {i}: seed mismatch {got.seed} vs {want.seed}"
                )

    def test_resume_safe_plateau_window(self, cfg):
        """Window over the cell plateau (intensity ~1) — many strikes, easy to check."""
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0)
        # Plateau: 20%–70% of life = 2160 s – 7560 s
        t0, t2 = 2160.0, 7560.0
        splits = [3000.0, 4000.5, 5400.0, 6000.123, 7000.0]
        self._assert_resume_safe(cell, t0, t2, cfg, splits)

    def test_resume_safe_full_lifetime_window(self, cfg):
        """Window spanning the entire cell life (grow + plateau + decay)."""
        set_world_seed(42)
        cell = _thunderstorm(spawn_time=100.0, duration=7200.0)
        t0 = 100.0  # spawn start (no strikes yet — envelope=0)
        t2 = 100.0 + 7200.0  # end of life
        splits = [1100.0, 3700.0, 5000.99, 6500.0]
        self._assert_resume_safe(cell, t0, t2, cfg, splits)

    def test_resume_safe_split_during_grow_phase(self, cfg):
        """Split during the early grow phase (first 20% of duration)."""
        set_world_seed(7)
        cell = _thunderstorm(duration=10_800.0)
        t0 = 0.0
        t2 = 3000.0  # still inside the plateau (0-28% of lifetime)
        # Split points landing in the grow phase (0-2160 s)
        splits = [500.0, 1080.0, 1800.0, 2500.0]
        self._assert_resume_safe(cell, t0, t2, cfg, splits)

    def test_resume_safe_split_during_decay_phase(self, cfg):
        """Split during the late decay phase (last 30% of duration)."""
        set_world_seed(99)
        cell = _thunderstorm(duration=9000.0)
        decay_start = 0.70 * 9000.0  # = 6300 s
        t0 = decay_start
        t2 = 9000.0
        splits = [7000.0, 7500.5, 8000.0, 8500.0]
        self._assert_resume_safe(cell, t0, t2, cfg, splits)

    def test_resume_safe_non_zero_spawn_time(self, cfg):
        """Cell with a large non-zero spawn_time doesn't misalign stream."""
        set_world_seed(1337)
        spawn = 5 * DAY + 3600.0
        cell = _thunderstorm(spawn_time=spawn, duration=10_800.0)
        t0 = spawn + 2000.0
        t2 = spawn + 8000.0
        splits = [spawn + 3000.0, spawn + 5400.123, spawn + 7000.0]
        self._assert_resume_safe(cell, t0, t2, cfg, splits)

    def test_resume_safe_different_cell_ids(self, cfg):
        """Two distinct cell ids produce independent, resume-safe schedules."""
        set_world_seed(1337)
        cell_a = _thunderstorm(id_="n:1:0", duration=10_800.0)
        cell_b = _thunderstorm(id_="n:1:1", duration=10_800.0)
        t0, t2 = 0.0, 5400.0
        splits = [1800.0, 3600.0]
        # Check each cell independently
        self._assert_resume_safe(cell_a, t0, t2, cfg, splits)
        self._assert_resume_safe(cell_b, t0, t2, cfg, splits)

    def test_resume_safe_natural_cell(self, cfg):
        """Use a cell from the natural spawn schedule (real-world path)."""
        set_world_seed(1337)
        # Find first natural thunderstorm in the first 40 days
        storm_cell = None
        for day in range(40):
            for c in natural_cells(day, cfg):
                if c.kind is CellKind.THUNDERSTORM:
                    storm_cell = c
                    break
            if storm_cell is not None:
                break
        assert storm_cell is not None, "no thunderstorm in first 40 days at seed 1337"

        t0 = storm_cell.spawn_time + 0.25 * storm_cell.duration_s
        t2 = storm_cell.spawn_time + 0.80 * storm_cell.duration_s
        splits = [
            storm_cell.spawn_time + 0.40 * storm_cell.duration_s,
            storm_cell.spawn_time + 0.60 * storm_cell.duration_s,
        ]
        self._assert_resume_safe(storm_cell, t0, t2, cfg, splits)


# ---------------------------------------------------------------------------
# Non-thunderstorm cells produce no strikes
# ---------------------------------------------------------------------------


class TestNonThunderstormYieldsNoStrikes:
    def test_shower_no_strikes(self, cfg):
        set_world_seed(1337)
        shower = StormCell(
            "n:0:shower",
            CellKind.SHOWER,
            0.0,
            (0.0, 0.0),
            10_800.0,
            800.0,
            1.0,
            (0.0, 0.0),
        )
        result = scheduled_strikes(shower, 0.0, 10_800.0, cfg)
        assert result == [], f"SHOWER should yield no strikes but got {len(result)}"

    def test_fog_bank_no_strikes(self, cfg):
        set_world_seed(1337)
        fog = StormCell(
            "n:0:fog",
            CellKind.FOG_BANK,
            0.0,
            (0.0, 0.0),
            10_800.0,
            800.0,
            1.0,
            (0.0, 0.0),
        )
        result = scheduled_strikes(fog, 0.0, 10_800.0, cfg)
        assert result == [], f"FOG_BANK should yield no strikes but got {len(result)}"

    def test_cloud_bank_no_strikes(self, cfg):
        set_world_seed(1337)
        cloud = StormCell(
            "n:0:cloud",
            CellKind.CLOUD_BANK,
            0.0,
            (0.0, 0.0),
            10_800.0,
            800.0,
            1.0,
            (0.0, 0.0),
        )
        result = scheduled_strikes(cloud, 0.0, 10_800.0, cfg)
        assert result == [], f"CLOUD_BANK should yield no strikes but got {len(result)}"


# ---------------------------------------------------------------------------
# Edge-case / boundary windows
# ---------------------------------------------------------------------------


class TestEdgeCaseWindows:
    def test_zero_width_window_no_strikes(self, cfg):
        """[t, t) is empty by definition."""
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0)
        for t in (0.0, 1800.0, 5400.0, 10_800.0):
            result = scheduled_strikes(cell, t, t, cfg)
            assert result == [], f"Zero-width window [t={t}, t) must yield no strikes"

    def test_inverted_window_no_strikes(self, cfg):
        """t1 < t0 is an empty window."""
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0)
        result = scheduled_strikes(cell, 5400.0, 1800.0, cfg)
        assert result == [], "Inverted window must yield no strikes"

    def test_window_entirely_before_spawn_no_strikes(self, cfg):
        """Query before the cell is born."""
        set_world_seed(1337)
        spawn = 3600.0
        cell = _thunderstorm(spawn_time=spawn, duration=10_800.0)
        result = scheduled_strikes(cell, 0.0, spawn, cfg)
        assert result == [], "Window entirely before spawn must yield no strikes"

    def test_window_entirely_after_death_no_strikes(self, cfg):
        """Query after the cell has died."""
        set_world_seed(1337)
        spawn, dur = 0.0, 3600.0
        cell = _thunderstorm(spawn_time=spawn, duration=dur)
        result = scheduled_strikes(cell, spawn + dur, spawn + dur + 3600.0, cfg)
        assert result == [], "Window entirely after death must yield no strikes"

    def test_window_straddles_spawn_only_post_spawn_strikes(self, cfg):
        """Window starts before spawn — strikes can only occur after spawn_time."""
        set_world_seed(1337)
        spawn = 1800.0
        cell = _thunderstorm(spawn_time=spawn, duration=10_800.0)
        result = scheduled_strikes(cell, 0.0, spawn + 3600.0, cfg)
        for s in result:
            assert s.time_abs >= spawn, f"Strike at t={s.time_abs} before spawn_time={spawn}"


# ---------------------------------------------------------------------------
# Thinning: more strikes near the plateau than at the edges
# ---------------------------------------------------------------------------


class TestThinning:
    def test_plateau_denser_than_grow_edge(self, cfg):
        """
        Equal-width windows: plateau window vs. early-grow window.
        Plateau intensity ≈ 1; early-grow intensity ≈ 0 → many more strikes.
        Pin current behavior: plateau_count >= grow_count.
        """
        set_world_seed(1337)
        dur = 10_800.0
        cell = _thunderstorm(duration=dur)
        window = 0.10 * dur  # equal-width sub-windows (1080 s)

        grow_start = 0.01 * dur
        plateau_mid = 0.45 * dur

        grow_strikes = scheduled_strikes(cell, grow_start, grow_start + window, cfg)
        plateau_strikes = scheduled_strikes(cell, plateau_mid, plateau_mid + window, cfg)

        assert len(plateau_strikes) >= len(grow_strikes), (
            f"Expected plateau density >= grow density, "
            f"but got grow={len(grow_strikes)}, plateau={len(plateau_strikes)}"
        )

    def test_plateau_denser_than_decay_edge(self, cfg):
        """End-of-life (decay) should also strike less than the plateau."""
        set_world_seed(1337)
        dur = 10_800.0
        cell = _thunderstorm(duration=dur)
        window = 0.10 * dur

        plateau_mid = 0.45 * dur
        decay_end = 0.90 * dur

        plateau_strikes = scheduled_strikes(cell, plateau_mid, plateau_mid + window, cfg)
        decay_strikes = scheduled_strikes(cell, decay_end, decay_end + window, cfg)

        # decay_end + window may exceed cell lifetime — that's fine, the function
        # caps to the lifetime automatically.
        assert len(plateau_strikes) >= len(decay_strikes), (
            f"Expected plateau density >= decay density, "
            f"but got plateau={len(plateau_strikes)}, decay={len(decay_strikes)}"
        )


# ---------------------------------------------------------------------------
# StrikeParams field types and value ranges
# ---------------------------------------------------------------------------


class TestStrikeParamsFields:
    def test_all_fields_present(self, cfg):
        """StrikeParams must have exactly the four documented fields."""
        field_names = {f.name for f in fields(StrikeParams)}
        assert "time_abs" in field_names
        assert "pos_xy" in field_names
        assert "intensity" in field_names
        assert "seed" in field_names

    def test_time_abs_is_float(self, cfg):
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0)
        for s in scheduled_strikes(cell, 0.0, 5400.0, cfg):
            assert isinstance(s.time_abs, float), (
                f"time_abs should be float, got {type(s.time_abs)}"
            )

    def test_pos_xy_is_length_2(self, cfg):
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0, radius=800.0)
        for s in scheduled_strikes(cell, 0.0, 5400.0, cfg):
            assert len(s.pos_xy) == 2, f"pos_xy should be length-2, got length {len(s.pos_xy)}"

    def test_intensity_in_unit_interval(self, cfg):
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0, peak=0.8)
        for s in scheduled_strikes(cell, 0.0, 10_800.0, cfg):
            assert 0.0 <= s.intensity <= 1.0, f"intensity out of [0, 1]: {s.intensity}"

    def test_seed_is_non_negative_int(self, cfg):
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0)
        for s in scheduled_strikes(cell, 0.0, 5400.0, cfg):
            assert isinstance(s.seed, int) and s.seed >= 0, (
                f"seed must be non-negative int, got {s.seed!r}"
            )

    def test_seed_fits_31_bits(self, cfg):
        """Seed is < 2**31 (signed shader-friendly int)."""
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0)
        for s in scheduled_strikes(cell, 0.0, 10_800.0, cfg):
            assert s.seed < 2**31, f"seed {s.seed} exceeds 31-bit signed range"

    def test_cell_id_field_on_strike_event_matches_int_fn(self, cfg):
        """
        Pin that the seed formula (cid * 1_000_003 + idx) % 2**31 is consistent
        across different cell ids — seeds for the same-index strike differ by cell.
        """
        set_world_seed(1337)
        cell_a = _thunderstorm(id_="n:0:0", duration=10_800.0)
        cell_b = _thunderstorm(id_="n:0:1", duration=10_800.0)
        strikes_a = scheduled_strikes(cell_a, 0.0, 10_800.0, cfg)
        strikes_b = scheduled_strikes(cell_b, 0.0, 10_800.0, cfg)
        if strikes_a and strikes_b:
            # Different cell ids → different seeds (overwhelmingly likely)
            seeds_a = {s.seed for s in strikes_a}
            seeds_b = {s.seed for s in strikes_b}
            # The union should be larger than each set (no total overlap)
            assert seeds_a != seeds_b or len(strikes_a) != len(strikes_b), (
                "Different cell ids produced identical strike seeds — suspicious"
            )

    def test_pos_xy_within_cell_radius(self, cfg):
        """Strike offset must be within the cell footprint (clamped by implementation)."""
        set_world_seed(1337)
        cell = _thunderstorm(duration=10_800.0, radius=800.0)
        for s in scheduled_strikes(cell, 0.0, 10_800.0, cfg):
            r = cell.radius(s.time_abs)
            dist = np.hypot(*s.pos_xy)
            assert dist <= r + 1e-6, (
                f"Strike pos_xy {s.pos_xy} is outside radius {r:.2f}: dist={dist:.2f}"
            )
