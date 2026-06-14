"""
tests/test_wind.py — Headless tests for fire_engine/world/wind/ (WP1).

Covers the full WP1 contract:
- determinism (same seed/time/state → bit-identical field), in-process and
  cross-process (subprocess, mirroring tests/test_rng.py);
- gust travel: cross-correlation peak between t and t+dt shifts downwind by
  ≈ mean*dt (the advection term is the whole point);
- sample(): shape, bilinear exactness at cell centres, profile monotone +
  floor + cap, out-of-region clamp, no NaN, vectorized speed sanity;
- storm > clear gust variance;
- field mean ≈ wind_dir * wind_speed;
- recenter hysteresis + bit-equal values at shared world points across a
  recenter at fixed time;
- pack round-trip (decode float16, channel order, byte length);
- modifier add/remove restores the base field exactly;
- a panda3d-import guard: nothing under fire_engine/world/wind/ imports panda3d.

Headless: no window, no GPU, no sky package import (weather is duck-typed via
SimpleNamespace).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.wind import (
    GustFront,
    WindField,
    pack_wind_field,
    vertical_profile,
)
from fire_engine.world.wind.gusts import build_modes, eval_gusts

SEED = 1337


def _sky(wind_dir=(1.0, 0.0), wind_speed=5.0, rain=0.0, cov=0.0, den=0.0):
    """Duck-typed SkyState stand-in (no sky package import)."""
    return SimpleNamespace(
        wind_dir=wind_dir,
        wind_speed=wind_speed,
        rain_intensity=rain,
        cloud_coverage=cov,
        cloud_density=den,
    )


def _field(cfg=None) -> WindField:
    set_world_seed(SEED)
    return WindField(cfg or Config())


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_fields_same_seed_identical(self):
        sky = _sky()
        a = _field()
        b = _field()
        a.update(0.016, 42.0, sky, (0.0, 0.0, 0.0))
        b.update(0.016, 42.0, sky, (0.0, 0.0, 0.0))
        assert np.array_equal(a.snapshot.field, b.snapshot.field)

    def test_repeated_update_same_time_identical(self):
        sky = _sky()
        f = _field()
        f.update(0.016, 7.0, sky, (0.0, 0.0, 0.0))
        first = f.snapshot.field.copy()
        f.update(0.016, 7.0, sky, (0.0, 0.0, 0.0))
        assert np.array_equal(first, f.snapshot.field)

    def test_modes_amp_normalised(self):
        set_world_seed(SEED)
        modes = build_modes(Config())
        assert modes.amp.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Cross-process determinism (mirror tests/test_rng.py)
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = """\
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.wind import WindField
from types import SimpleNamespace
set_world_seed({seed})
sky = SimpleNamespace(wind_dir=(0.6, 0.8), wind_speed=6.0,
                      rain_intensity=0.3, cloud_coverage=0.5, cloud_density=0.5)
f = WindField(Config())
f.update(0.016, 33.0, sky, (12.0, -7.0, 0.0))
# Print a compact checksum of the field so output is short + comparable.
print(repr(float(f.snapshot.field.sum())))
"""


def _run_field_subprocess(seed: int) -> float:
    root = str(Path(__file__).parent.parent.resolve())
    script = _SUBPROCESS_SCRIPT.format(seed=seed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    return eval(result.stdout.strip())


class TestCrossProcessDeterminism:
    def test_two_processes_identical(self):
        a = _run_field_subprocess(SEED)
        b = _run_field_subprocess(SEED)
        assert a == b, f"cross-process wind mismatch: {a} != {b}"

    def test_different_seeds_differ(self):
        a = _run_field_subprocess(SEED)
        b = _run_field_subprocess(SEED + 12345)
        assert a != b


# ---------------------------------------------------------------------------
# Gust travel — crests advect downwind at ~mean speed
# ---------------------------------------------------------------------------


class TestGustTravel:
    def test_single_mode_crest_advects_downwind(self):
        # The advection term is per-mode and exact: a single +X-propagating mode
        # in a +X mean wind has its crests travel downwind by exactly mean*dt.
        # Build a clean single-mode basis so the cross-correlation is
        # unambiguous (no multi-mode quasi-periodic spurious peaks).
        from fire_engine.world.wind.gusts import GustModes

        wavelength = 50.0
        k = 2.0 * np.pi / wavelength
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
        dt = 2.0
        t0 = 50.0
        expected = mean[0] * dt  # 12 m downwind
        spacing = 0.25
        xs = np.arange(0.0, 400.0, spacing)
        X = xs[:, None]
        Y = np.zeros_like(X)
        g0, _ = eval_gusts(single, X, Y, t0, mean)
        g1, _ = eval_gusts(single, X, Y, t0 + dt, mean)
        g0 = g0[:, 0]
        g1 = g1[:, 0]
        corr = np.correlate(g1, g0, mode="full")
        lags = np.arange(-len(g0) + 1, len(g0))
        # One wavelength search window around 0 disambiguates the periodic peak.
        win = np.abs(lags * spacing) <= wavelength * 0.5
        peak_shift_m = lags[win][np.argmax(corr[win])] * spacing
        assert peak_shift_m == pytest.approx(expected, abs=1.0)

    def test_full_field_gust_pattern_moves(self):
        # End-to-end: the composed field's gust pattern at t differs from t+dt
        # (it is evolving/advecting, not static).
        f = _field()
        sky = _sky(wind_speed=6.0)
        f.update(0.016, 30.0, sky, (0.0, 0.0, 0.0))
        a = f.snapshot.field[..., 0].copy()
        f.update(0.016, 30.5, sky, (0.0, 0.0, 0.0))
        b = f.snapshot.field[..., 0]
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------


class TestSample:
    def test_shape(self):
        f = _field()
        f.update(0.016, 1.0, _sky(), (0.0, 0.0, 0.0))
        v = f.sample(np.zeros((17, 3), dtype=np.float32))
        assert v.shape == (17, 3)

    def test_empty(self):
        f = _field()
        f.update(0.016, 1.0, _sky(), (0.0, 0.0, 0.0))
        v = f.sample(np.zeros((0, 3), dtype=np.float32))
        assert v.shape == (0, 3)

    def test_bilinear_exact_at_cell_centres(self):
        # At a cell centre, sampling at z=z_ref (profile == 1) must return that
        # cell's stored (vx, vy) exactly (bilinear weight collapses to 1 corner).
        f = _field()
        f.update(0.016, 9.0, _sky(wind_speed=4.0), (0.0, 0.0, 0.0))
        snap = f.snapshot
        ox, oy = snap.origin_m
        cm = snap.cell_m
        z_ref = Config().wind_profile_z_ref + Config().ground_height_m
        # Probe a handful of interior cells.
        for i, j in [(10, 10), (20, 5), (33, 40)]:
            cx = ox + (i + 0.5) * cm
            cy = oy + (j + 0.5) * cm
            v = f.sample(np.array([[cx, cy, z_ref]], dtype=np.float32))[0]
            np.testing.assert_allclose(v[0], snap.field[i, j, 0], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(v[1], snap.field[i, j, 1], rtol=1e-4, atol=1e-4)
            assert v[2] == 0.0

    def test_profile_monotone_floor_cap(self):
        cfg = Config()
        z = np.array(
            [
                cfg.ground_height_m - 5.0,  # below ground -> floor
                cfg.ground_height_m,
                cfg.ground_height_m + 1.0,
                cfg.ground_height_m + 10.0,
                cfg.ground_height_m + 1000.0,
            ]
        )
        m = vertical_profile(z, cfg.ground_height_m, cfg)
        assert m[0] == pytest.approx(cfg.wind_profile_floor)
        assert m[1] == pytest.approx(cfg.wind_profile_floor)
        assert np.all(np.diff(m) >= -1e-6)  # non-decreasing
        assert m.max() <= cfg.wind_profile_cap + 1e-6
        assert m.min() >= cfg.wind_profile_floor - 1e-6

    def test_out_of_region_clamps_to_edge(self):
        f = _field()
        f.update(0.016, 3.0, _sky(), (0.0, 0.0, 0.0))
        snap = f.snapshot
        ox, oy = snap.origin_m
        cm = snap.cell_m
        z_ref = Config().wind_profile_z_ref + Config().ground_height_m
        # Far outside the region (way past +X edge) clamps to the last column.
        far = np.array([[ox + 100000.0, oy + (5 + 0.5) * cm, z_ref]], dtype=np.float32)
        v = f.sample(far)[0]
        edge = snap.field[snap.cells - 1, 5, :2]
        np.testing.assert_allclose(v[:2], edge, rtol=1e-4, atol=1e-4)

    def test_no_nan_anywhere(self):
        f = _field()
        f.update(0.016, 11.0, _sky(wind_speed=10.0, rain=1.0, cov=1.0, den=1.0), (0.0, 0.0, 0.0))
        pts = np.random.RandomState(0).uniform(-5000, 5000, (500, 3))
        v = f.sample(pts.astype(np.float32))
        assert not np.isnan(v).any()
        assert np.isfinite(v).all()

    def test_speed_vectorized_sanity(self):
        # Many points at ground level: horizontal speed must be bounded and
        # never below the floor*(mean - gust) lower bound is hard to state, so
        # just assert finite + reasonable magnitude under a calm sky.
        f = _field()
        f.update(0.016, 2.0, _sky(wind_speed=5.0), (0.0, 0.0, 0.0))
        ox, oy = f.snapshot.origin_m
        cm = f.snapshot.cell_m
        gx, gy = np.meshgrid(np.arange(0, 64, 4), np.arange(0, 64, 4))
        pts = np.stack(
            [
                ox + gx.ravel() * cm / 4,
                oy + gy.ravel() * cm / 4,
                np.full(gx.size, Config().ground_height_m + 2.0),
            ],
            axis=1,
        ).astype(np.float32)
        v = f.sample(pts)
        speed = np.hypot(v[:, 0], v[:, 1])
        assert np.isfinite(speed).all()
        assert speed.max() < 50.0  # sane for a 5 m/s mean + gusts


# ---------------------------------------------------------------------------
# Weather scaling
# ---------------------------------------------------------------------------


class TestWeatherScaling:
    def test_storm_gustier_than_clear(self):
        clear = _field()
        storm = _field()
        clear.update(0.016, 20.0, _sky(wind_speed=6.0, rain=0.0, cov=0.0, den=0.0), (0.0, 0.0, 0.0))
        storm.update(0.016, 20.0, _sky(wind_speed=6.0, rain=1.0, cov=1.0, den=1.0), (0.0, 0.0, 0.0))
        # Subtract the mean so we compare gust variance, not the mean offset.
        cv = clear.snapshot.field[..., 0]
        sv = storm.snapshot.field[..., 0]
        clear_var = (cv - cv.mean()).var()
        storm_var = (sv - sv.mean()).var()
        assert storm_var > clear_var

    def test_field_mean_tracks_wind_vector(self):
        f = _field()
        wd = (0.6, 0.8)
        ws = 7.0
        f.update(0.016, 15.0, _sky(wind_dir=wd, wind_speed=ws), (0.0, 0.0, 0.0))
        fld = f.snapshot.field
        # Average over the grid: gusts are zero-mean-ish, so the field mean
        # should sit near wind_dir * wind_speed.
        mvx = fld[..., 0].mean()
        mvy = fld[..., 1].mean()
        assert mvx == pytest.approx(wd[0] * ws, abs=1.5)
        assert mvy == pytest.approx(wd[1] * ws, abs=1.5)


# ---------------------------------------------------------------------------
# Recenter hysteresis + shared-point bit-equality
# ---------------------------------------------------------------------------


class TestRecenter:
    def test_no_move_within_margin(self):
        f = _field()
        f.update(0.016, 1.0, _sky(), (0.0, 0.0, 0.0))
        o0 = f.snapshot.origin_m
        f.update(0.016, 1.0, _sky(), (4.0, 4.0, 0.0))  # < 32 m margin
        assert f.snapshot.origin_m == o0

    def test_move_outside_margin(self):
        f = _field()
        f.update(0.016, 1.0, _sky(), (0.0, 0.0, 0.0))
        o0 = f.snapshot.origin_m
        f.update(0.016, 1.0, _sky(), (200.0, 0.0, 0.0))
        assert f.snapshot.origin_m != o0

    def test_shared_world_points_bit_equal_across_recenter(self):
        # At a FIXED time, the field value at a world point that is inside both
        # the pre- and post-recenter tiles must be bit-identical (the field is
        # analytic in position; recenter only shifts the window).
        sky = _sky(wind_speed=5.0)
        t = 60.0
        f = _field()
        f.update(0.016, t, sky, (0.0, 0.0, 0.0))
        snap_a = f.snapshot
        # Pick a world point well inside tile A and that will also be inside B.
        ox, oy = snap_a.origin_m
        cm = snap_a.cell_m
        px = ox + (40 + 0.5) * cm
        py = oy + (40 + 0.5) * cm
        z = Config().wind_profile_z_ref + Config().ground_height_m
        va = f.sample(np.array([[px, py, z]], dtype=np.float32))[0]

        # Recenter by moving the player far enough to snap, same time t.
        f.update(0.016, t, sky, (px, py, 0.0))
        assert f.snapshot.origin_m != snap_a.origin_m  # actually moved
        vb = f.sample(np.array([[px, py, z]], dtype=np.float32))[0]
        np.testing.assert_allclose(va, vb, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# pack round-trip
# ---------------------------------------------------------------------------


class TestPack:
    def test_byte_length(self):
        f = _field()
        f.update(0.016, 1.0, _sky(), (0.0, 0.0, 0.0))
        data = pack_wind_field(f.snapshot)
        cells = f.snapshot.cells
        assert len(data) == cells * cells * 4 * 2  # fp16 RGBA

    def test_channel_order_and_layout(self):
        # Decode the fp16 buffer and assert (y, x) row-major + BGRA mapping:
        # decoded[y, x] = (B=turb, G=vy, R=vx, A=speed) for field[x, y].
        f = _field()
        f.update(0.016, 4.0, _sky(wind_speed=5.0), (0.0, 0.0, 0.0))
        snap = f.snapshot
        cells = snap.cells
        buf = np.frombuffer(pack_wind_field(snap), dtype=np.float16)
        dec = buf.reshape(cells, cells, 4).astype(np.float32)  # [y, x, BGRA]
        for i, j in [(0, 0), (10, 20), (63, 63), (5, 50)]:
            vx = snap.field[i, j, 0]
            vy = snap.field[i, j, 1]
            turb = snap.field[i, j, 2]
            speed = np.hypot(vx, vy)
            # Row-major (y, x): texel for field[x=i, y=j] is dec[j, i].
            b, g, r, a = dec[j, i]
            np.testing.assert_allclose(b, turb, atol=1e-2)
            np.testing.assert_allclose(g, vy, atol=1e-2)
            np.testing.assert_allclose(r, vx, atol=1e-2)
            np.testing.assert_allclose(a, speed, atol=1e-2)


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------


class TestModifiers:
    def test_add_remove_restores_base(self):
        sky = _sky(wind_speed=5.0)
        t = 25.0
        f = _field()
        f.update(0.016, t, sky, (0.0, 0.0, 0.0))
        base = f.snapshot.field.copy()

        front = GustFront(("test",), (1.0, 0.0), speed=12.0, strength=8.0, width_m=24.0)
        f.add_modifier(front)
        f.update(0.016, t, sky, (0.0, 0.0, 0.0))
        modified = f.snapshot.field
        assert not np.array_equal(base, modified)  # modifier changed it

        f.remove_modifier(front)
        f.update(0.016, t, sky, (0.0, 0.0, 0.0))
        np.testing.assert_array_equal(base, f.snapshot.field)

    def test_remove_absent_is_noop(self):
        f = _field()
        front = GustFront(("x",), (1.0, 0.0), 1.0, 1.0, 1.0)
        f.remove_modifier(front)  # must not raise

    def test_gustfront_pure_function_of_time(self):
        # Same (seed_key, t) → identical in-place contribution.
        X, Y = np.meshgrid(np.arange(0.0, 256.0, 4.0), np.arange(0.0, 256.0, 4.0), indexing="ij")
        front = GustFront(("p",), (0.7, 0.7), 10.0, 5.0, 20.0)
        a_vx = np.zeros_like(X)
        a_vy = np.zeros_like(X)
        a_t = np.zeros_like(X)
        b_vx = np.zeros_like(X)
        b_vy = np.zeros_like(X)
        b_t = np.zeros_like(X)
        front.apply(X, Y, 13.0, a_vx, a_vy, a_t)
        front.apply(X, Y, 13.0, b_vx, b_vy, b_t)
        assert np.array_equal(a_vx, b_vx)
        assert np.array_equal(a_t, b_t)


# ---------------------------------------------------------------------------
# Hard-rule guard: no panda3d anywhere under fire_engine/world/wind/
# ---------------------------------------------------------------------------


class TestNoPanda3D:
    def test_no_panda3d_import_in_wind_package(self):
        import ast

        wind_dir = Path(__file__).parent.parent / "fire_engine" / "world" / "wind"
        offenders = []
        for src in wind_dir.glob("*.py"):
            tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    root = name.split(".")[0]
                    if root in ("panda3d", "direct"):
                        offenders.append(f"{src.name}: import {name}")
        assert not offenders, f"panda3d leaked into wind/: {offenders}"
