"""
tests/world/wind/test_gusts.py — Characterization (golden-master) tests for
fire_engine/world/wind/gusts.py: GustModes dataclass, build_modes, eval_gusts.

DO NOT FIX BUGS — pin current behaviour and report suspicions.
Headless only.  Fixed seeds via set_world_seed / Config.world_seed.
All assertions use numpy bulk operations; no per-element Python loops.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.wind.gusts import GustModes, build_modes, eval_gusts

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SEED_A = 1337  # canonical seed used throughout test_wind.py
SEED_B = 9999  # a second distinct seed


def _modes(seed: int = SEED_A, cfg: Config | None = None) -> GustModes:
    """Build GustModes from a fixed seed."""
    set_world_seed(seed)
    return build_modes(cfg or Config())


def _grid(cells: int = 16, cell_m: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) meshgrid of cell-centre world positions, shape (cells, cells)."""
    xs = (np.arange(cells) + 0.5) * cell_m
    return np.meshgrid(xs, xs, indexing="ij")


# ---------------------------------------------------------------------------
# build_modes — GustModes dataclass structure
# ---------------------------------------------------------------------------


class TestBuildModesStructure:
    """Pin the shape and dtype of every array in GustModes."""

    def test_all_arrays_have_mode_count_length(self):
        cfg = Config()
        modes = _modes(SEED_A, cfg)
        M = int(cfg.wind_gust_modes)
        for attr in ("kx", "ky", "omega", "phase0", "pux", "puy", "amp"):
            arr = getattr(modes, attr)
            assert arr.shape == (M,), f"{attr} shape {arr.shape} != ({M},)"

    def test_all_arrays_are_float32(self):
        modes = _modes(SEED_A)
        for attr in ("kx", "ky", "omega", "phase0", "pux", "puy", "amp"):
            arr = getattr(modes, attr)
            assert arr.dtype == np.float32, f"{attr} dtype is {arr.dtype}"

    def test_amp_sums_to_one(self):
        """Red-noise amplitudes must be normalised to sum == 1."""
        modes = _modes(SEED_A)
        assert float(modes.amp.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_amp_all_positive(self):
        """All amplitudes are strictly positive (1/wavelength normalised)."""
        modes = _modes(SEED_A)
        assert np.all(modes.amp > 0)

    def test_push_dirs_are_unit_vectors(self):
        """(pux, puy) must lie on the unit circle for each mode."""
        modes = _modes(SEED_A)
        lengths = np.hypot(modes.pux, modes.puy)
        np.testing.assert_allclose(lengths, np.ones_like(lengths), atol=1e-6)

    def test_omega_within_config_bounds(self):
        """omega values drawn from [omega_min, omega_max]."""
        cfg = Config()
        modes = _modes(SEED_A, cfg)
        assert np.all(modes.omega >= cfg.wind_gust_omega_min - 1e-6)
        assert np.all(modes.omega <= cfg.wind_gust_omega_max + 1e-6)


# ---------------------------------------------------------------------------
# build_modes — determinism
# ---------------------------------------------------------------------------


class TestBuildModesDeterminism:
    """Same seed → identical GustModes; different seed → different modes."""

    def test_same_seed_identical_arrays(self):
        a = _modes(SEED_A)
        b = _modes(SEED_A)
        for attr in ("kx", "ky", "omega", "phase0", "pux", "puy", "amp"):
            assert np.array_equal(getattr(a, attr), getattr(b, attr)), (
                f"{attr} differs between two calls with the same seed"
            )

    def test_different_seed_different_arrays(self):
        a = _modes(SEED_A)
        b = _modes(SEED_B)
        # At least one field should differ; almost certainly all do.
        diffs = [
            not np.array_equal(getattr(a, attr), getattr(b, attr))
            for attr in ("kx", "ky", "omega", "phase0")
        ]
        assert any(diffs), "Different seeds produced identical GustModes — suspect RNG"

    def test_default_config_world_seed_is_canonical(self):
        """Config.world_seed default must match SEED_A so tests stay in sync."""
        assert Config().world_seed == SEED_A


# ---------------------------------------------------------------------------
# eval_gusts — output shape
# ---------------------------------------------------------------------------


class TestEvalGustsShape:
    """Output arrays have the same shape as the input X, Y grids."""

    def test_square_grid_shape(self):
        modes = _modes()
        X, Y = _grid(cells=16)
        gx, gy = eval_gusts(modes, X, Y, t_eff=0.0, mean=(2.0, 0.0))
        assert gx.shape == (16, 16)
        assert gy.shape == (16, 16)

    def test_rectangular_grid_shape(self):
        modes = _modes()
        xs = np.arange(0.0, 32.0, 4.0)  # 8 points
        ys = np.arange(0.0, 64.0, 4.0)  # 16 points
        X, Y = np.meshgrid(xs, ys, indexing="ij")  # (8, 16)
        gx, gy = eval_gusts(modes, X, Y, t_eff=1.0, mean=(0.0, 0.0))
        assert gx.shape == (8, 16)
        assert gy.shape == (8, 16)

    def test_1d_positions_shape(self):
        """eval_gusts also works on 1-D position vectors."""
        modes = _modes()
        X = np.linspace(0.0, 200.0, 50)
        Y = np.zeros(50)
        gx, gy = eval_gusts(modes, X, Y, t_eff=5.0, mean=(3.0, 1.0))
        assert gx.shape == (50,)
        assert gy.shape == (50,)

    def test_output_is_float32(self):
        modes = _modes()
        X, Y = _grid(8)
        gx, gy = eval_gusts(modes, X, Y, t_eff=0.0, mean=(0.0, 0.0))
        assert gx.dtype == np.float32, f"gx dtype {gx.dtype}"
        assert gy.dtype == np.float32, f"gy dtype {gy.dtype}"

    def test_scalar_positions(self):
        """Single position (0-d-broadcastable) must not crash and return scalar."""
        modes = _modes()
        gx, _gy = eval_gusts(modes, np.float32(0.0), np.float32(0.0), t_eff=0.0, mean=(0.0, 0.0))
        assert np.ndim(gx) == 0 or gx.shape == ()


# ---------------------------------------------------------------------------
# eval_gusts — finite / magnitude sanity
# ---------------------------------------------------------------------------


class TestEvalGustsFinite:
    """Pin that eval_gusts never produces NaN/Inf and stays in a sane range."""

    def test_no_nan_or_inf(self):
        modes = _modes()
        X, Y = _grid(64, 4.0)
        gx, gy = eval_gusts(modes, X, Y, t_eff=100.0, mean=(5.0, 3.0))
        assert np.isfinite(gx).all(), "gx contains NaN or Inf"
        assert np.isfinite(gy).all(), "gy contains NaN or Inf"

    def test_dimensionless_magnitude_bounded(self):
        """Gust shape amplitudes sum to 1, so |gx|+|gy| per cell <= 2 theoretically.
        Pin what the current implementation actually produces (expect < 2)."""
        modes = _modes()
        X, Y = _grid(64, 4.0)
        for t in (0.0, 50.0, 1000.0):
            gx, gy = eval_gusts(modes, X, Y, t_eff=t, mean=(5.0, 0.0))
            mag = np.abs(gx) + np.abs(gy)
            # Pin: combined amplitude never exceeds 2 (modes sum to 1, sin is bounded).
            assert float(mag.max()) < 2.0, f"magnitude {mag.max():.4f} exceeded 2.0 at t={t}"

    def test_zero_mean_still_finite(self):
        """zero mean wind (calm) must still produce a finite field."""
        modes = _modes()
        X, Y = _grid(32, 4.0)
        gx, gy = eval_gusts(modes, X, Y, t_eff=10.0, mean=(0.0, 0.0))
        assert np.isfinite(gx).all()
        assert np.isfinite(gy).all()


# ---------------------------------------------------------------------------
# eval_gusts — determinism (same inputs → identical outputs)
# ---------------------------------------------------------------------------


class TestEvalGustsDeterminism:
    """eval_gusts is a pure function: same modes + time + positions → bit-equal."""

    def test_same_call_twice_identical(self):
        modes = _modes()
        X, Y = _grid(32, 4.0)
        gx1, gy1 = eval_gusts(modes, X, Y, t_eff=37.0, mean=(4.0, 2.0))
        gx2, gy2 = eval_gusts(modes, X, Y, t_eff=37.0, mean=(4.0, 2.0))
        assert np.array_equal(gx1, gx2), "gx differs on repeated call"
        assert np.array_equal(gy1, gy2), "gy differs on repeated call"

    def test_same_modes_same_result_from_second_seed_call(self):
        """Re-seeding must not matter once modes are built."""
        modes = _modes(SEED_A)
        X, Y = _grid(16, 4.0)
        gx1, _ = eval_gusts(modes, X, Y, t_eff=10.0, mean=(2.0, 0.0))
        # Now set a different world seed — modes object is already frozen.
        set_world_seed(SEED_B)
        gx2, _ = eval_gusts(modes, X, Y, t_eff=10.0, mean=(2.0, 0.0))
        assert np.array_equal(gx1, gx2), (
            "eval_gusts result changed after re-seeding — modes not truly frozen"
        )


# ---------------------------------------------------------------------------
# eval_gusts — time evolution
# ---------------------------------------------------------------------------


class TestEvalGustsTimeEvolution:
    """Field must animate: different times → different values; small dt → small change."""

    def test_different_times_produce_different_fields(self):
        modes = _modes()
        X, Y = _grid(16, 4.0)
        gx0, _gy0 = eval_gusts(modes, X, Y, t_eff=0.0, mean=(3.0, 0.0))
        gx1, _gy1 = eval_gusts(modes, X, Y, t_eff=5.0, mean=(3.0, 0.0))
        assert not np.array_equal(gx0, gx1), "gust field is static across time"

    def test_small_dt_small_change(self):
        """A tiny time step should only produce a small change in the gust field.
        Pin: max abs delta over the grid < 0.2 for dt=0.016 s (one game frame)."""
        modes = _modes()
        X, Y = _grid(32, 4.0)
        t0 = 100.0
        dt = 0.016
        mean = (5.0, 2.0)
        gx0, _gy0 = eval_gusts(modes, X, Y, t_eff=t0, mean=mean)
        gx1, _gy1 = eval_gusts(modes, X, Y, t_eff=t0 + dt, mean=mean)
        max_delta = float(np.abs(gx1 - gx0).max())
        # Pin current behaviour: frame-to-frame delta stays well below 0.2
        # (dimensionless gust shape).  Fails if omega or amp becomes absurd.
        assert max_delta < 0.2, (
            f"frame-to-frame gust delta {max_delta:.4f} too large — continuity suspect"
        )

    def test_large_dt_differs_meaningfully(self):
        """Over a long time (10 s), the field should change appreciably."""
        modes = _modes()
        X, Y = _grid(32, 4.0)
        gx0, _ = eval_gusts(modes, X, Y, t_eff=0.0, mean=(3.0, 0.0))
        gx1, _ = eval_gusts(modes, X, Y, t_eff=10.0, mean=(3.0, 0.0))
        assert not np.allclose(gx0, gx1, atol=1e-3), (
            "gust field barely changed over 10 s — omega or advection suspect"
        )


# ---------------------------------------------------------------------------
# eval_gusts — spatial determinism (same position same time → same value)
# ---------------------------------------------------------------------------


class TestEvalGustsSpatial:
    """Evaluating at the same position twice must give bit-identical results."""

    def test_same_position_same_value(self):
        modes = _modes()
        x0, y0 = np.array([48.0]), np.array([20.0])
        v1x, v1y = eval_gusts(modes, x0, y0, t_eff=7.5, mean=(2.0, 1.0))
        v2x, v2y = eval_gusts(modes, x0, y0, t_eff=7.5, mean=(2.0, 1.0))
        assert np.array_equal(v1x, v2x)
        assert np.array_equal(v1y, v2y)

    def test_field_is_continuous_in_space(self):
        """Adjacent cell centres should have similar gust values (spatial smoothness).
        Pin: max abs gradient between adjacent cells < 0.5 (dimensionless)."""
        modes = _modes()
        X, Y = _grid(32, 4.0)
        gx, _ = eval_gusts(modes, X, Y, t_eff=0.0, mean=(3.0, 1.0))
        diffs = np.abs(np.diff(gx, axis=0))
        max_jump = float(diffs.max())
        assert max_jump < 0.5, (
            f"spatial jump {max_jump:.4f} between adjacent cells — aliasing suspect"
        )


# ---------------------------------------------------------------------------
# eval_gusts — advection
# ---------------------------------------------------------------------------


class TestEvalGustsAdvection:
    """The advection term must visibly shift the pattern downwind over time."""

    def test_advection_shifts_pattern_downwind(self):
        """A single-mode basis with zero intrinsic omega lets us verify the
        k·mean advection shift exactly (mirrors test_wind.py's approach)."""
        wavelength = 50.0
        k = (2.0 * np.pi) / wavelength
        single = GustModes(
            kx=np.array([k], np.float32),
            ky=np.array([0.0], np.float32),
            omega=np.array([0.0], np.float32),  # no intrinsic pulsing
            phase0=np.array([0.0], np.float32),
            pux=np.array([1.0], np.float32),
            puy=np.array([0.0], np.float32),
            amp=np.array([1.0], np.float32),
        )
        mean = (6.0, 0.0)
        dt = 3.0
        t0 = 0.0
        spacing = 0.5
        xs = np.arange(0.0, 200.0, spacing)
        X = xs[:, None]
        Y = np.zeros_like(X)
        g0, _ = eval_gusts(single, X, Y, t_eff=t0, mean=mean)
        g1, _ = eval_gusts(single, X, Y, t_eff=t0 + dt, mean=mean)
        g0 = g0[:, 0]
        g1 = g1[:, 0]
        corr = np.correlate(g1, g0, mode="full")
        lags = np.arange(-len(g0) + 1, len(g0))
        win = np.abs(lags * spacing) <= wavelength * 0.5
        peak_m = lags[win][np.argmax(corr[win])] * spacing
        expected_m = mean[0] * dt  # 18 m
        assert peak_m == pytest.approx(expected_m, abs=1.5), (
            f"advection peak at {peak_m:.1f} m, expected {expected_m:.1f} m"
        )

    def test_zero_mean_no_advection_shift(self):
        """With mean=(0,0) the only time variation comes from omega (intrinsic
        pulsing).  The spatial PATTERN at t and t+pi/omega should differ (not shift)."""
        modes = _modes()
        X, Y = _grid(16, 4.0)
        gx0, _ = eval_gusts(modes, X, Y, t_eff=0.0, mean=(0.0, 0.0))
        gx1, _ = eval_gusts(modes, X, Y, t_eff=5.0, mean=(0.0, 0.0))
        # With zero mean the advection term vanishes; field still evolves via omega.
        assert not np.array_equal(gx0, gx1)

    def test_nonzero_mean_changes_pattern_vs_zero_mean(self):
        """With nonzero mean, the same (modes, t) pair should produce a different
        field than mean=(0,0) because advection is included in the phase."""
        modes = _modes()
        X, Y = _grid(16, 4.0)
        gx_calm, _ = eval_gusts(modes, X, Y, t_eff=20.0, mean=(0.0, 0.0))
        gx_wind, _ = eval_gusts(modes, X, Y, t_eff=20.0, mean=(8.0, 0.0))
        assert not np.array_equal(gx_calm, gx_wind), (
            "mean wind had no effect on gust field — advection term may be broken"
        )
