"""
tests/test_wind_ball.py — Headless tests for the wind-debug-ball physics (WP5).

The rendered ``WindBallDebugComponent`` lives in ``world/`` (panda3d), so it is
excluded from the headless suite by the import rule.  Its *physics* is the pure,
panda3d-free :func:`fire_engine.world.wind.debug_ball_step`, which we step here with a
real (synthetic-weather) :class:`~fire_engine.world.wind.WindField` to prove the seam:
a ball resting on flat ground accelerates and travels **downwind**, settles in
calm air, and stays clamped to the ground plane.

These mirror the in-engine proof (the ball you watch scoot when a gust crosses)
without a window or GPU.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.wind import BallParams, WindField, debug_ball_step
from fire_engine.world.wind.debug import debug_ball_step as _direct_import  # name check


SEED = 1337


def _field() -> WindField:
    set_world_seed(SEED)
    return WindField(Config())


def _params(ground_z: float = 8.0) -> BallParams:
    return BallParams(ground_z=ground_z, radius_m=0.4, drag=2.5,
                      gravity=9.81, friction=1.5, max_speed=25.0)


# ---------------------------------------------------------------------------
# Pure integrator unit behaviour (no field)
# ---------------------------------------------------------------------------

def test_rests_on_ground_in_calm_air():
    """Zero wind → the ball sits at ground+radius and does not drift."""
    params = _params()
    rest_z = params.ground_z + params.radius_m
    pos = np.array([0.0, 0.0, rest_z], dtype=np.float64)
    vel = np.zeros(3)
    wind = np.zeros(3)
    for _ in range(200):
        pos, vel = debug_ball_step(pos, vel, wind, 0.02, params)
    assert pos[2] == rest_z                       # clamped to the plane
    assert abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6   # no horizontal drift
    assert np.hypot(vel[0], vel[1]) < 1e-6        # friction settled it


def test_gust_pushes_ball_downwind():
    """A steady +X wind accelerates the ball in +X (and not in −X / ±Y)."""
    params = _params()
    rest_z = params.ground_z + params.radius_m
    pos = np.array([0.0, 0.0, rest_z], dtype=np.float64)
    vel = np.zeros(3)
    wind = np.array([6.0, 0.0, 0.0])              # +X gust, m/s
    for _ in range(50):                           # ~1 s at 50 Hz
        pos, vel = debug_ball_step(pos, vel, wind, 0.02, params)
    assert pos[0] > 0.5                            # travelled downwind
    assert abs(pos[1]) < 1e-6                      # no lateral drift
    assert vel[0] > 0.0


def test_diagonal_wind_displaces_along_wind():
    """The displacement direction matches the (normalised) wind direction."""
    params = _params()
    rest_z = params.ground_z + params.radius_m
    pos = np.array([0.0, 0.0, rest_z], dtype=np.float64)
    vel = np.zeros(3)
    wind = np.array([4.0, 3.0, 0.0])              # dir (0.8, 0.6)
    start = pos.copy()
    for _ in range(60):
        pos, vel = debug_ball_step(pos, vel, wind, 0.02, params)
    disp = pos[:2] - start[:2]
    assert np.linalg.norm(disp) > 0.5
    wind_dir = wind[:2] / np.linalg.norm(wind[:2])
    disp_dir = disp / np.linalg.norm(disp)
    # Displacement aligns with the wind direction (cosine ≈ 1).
    assert float(np.dot(disp_dir, wind_dir)) > 0.999


def test_speed_clamp_caps_horizontal_velocity():
    """An extreme wind cannot push the ball past max_speed on the ground."""
    params = _params()
    rest_z = params.ground_z + params.radius_m
    pos = np.array([0.0, 0.0, rest_z], dtype=np.float64)
    vel = np.zeros(3)
    wind = np.array([1000.0, 0.0, 0.0])
    for _ in range(100):
        pos, vel = debug_ball_step(pos, vel, wind, 0.02, params)
    assert np.hypot(vel[0], vel[1]) <= params.max_speed + 1e-6


def test_step_does_not_mutate_inputs():
    """debug_ball_step is pure — the input arrays are left untouched."""
    params = _params()
    pos = np.array([1.0, 2.0, 8.4], dtype=np.float64)
    vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    pos_copy, vel_copy = pos.copy(), vel.copy()
    debug_ball_step(pos, vel, np.array([5.0, 0.0, 0.0]), 0.02, params)
    assert np.array_equal(pos, pos_copy)
    assert np.array_equal(vel, vel_copy)


# ---------------------------------------------------------------------------
# Integration with a real WindField (the actual seam)
# ---------------------------------------------------------------------------

def test_ball_scoots_under_real_wind_field():
    """
    Step the ball against a real WindField (steady +X breeze) — it must end up
    displaced roughly along the field's mean wind direction.  This is the
    headless twin of the in-engine "watch it scoot" proof.
    """
    from types import SimpleNamespace

    field = _field()
    params = _params(ground_z=Config().ground_height_m)
    rest_z = params.ground_z + params.radius_m
    pos = np.array([0.0, 0.0, rest_z], dtype=np.float64)
    vel = np.zeros(3)
    sky = SimpleNamespace(wind_dir=(1.0, 0.0), wind_speed=8.0,
                          rain_intensity=0.0, cloud_coverage=0.0,
                          cloud_density=0.0)
    start = pos.copy()
    for i in range(400):                          # ~8 s at 50 Hz
        t = i * 0.02
        field.update(0.02, t, sky, (float(pos[0]), float(pos[1]),
                                    float(pos[2])))
        v_wind = field.sample(pos[None])[0]
        pos, vel = debug_ball_step(pos, vel, v_wind, 0.02, params)
        assert pos[2] == rest_z                   # never leaves the ground
        assert np.all(np.isfinite(pos))

    disp = pos[:2] - start[:2]
    # Net displacement is downwind-dominant (+X mean wind): X component large
    # and positive, well beyond any gust-driven lateral wobble.
    assert disp[0] > 2.0
    assert disp[0] > abs(disp[1])


def test_storm_moves_ball_more_than_calm():
    """A storm field shoves the ball farther than a near-calm field."""
    from types import SimpleNamespace

    def _run(wind_speed, rain, cov, den):
        field = _field()
        params = _params(ground_z=Config().ground_height_m)
        rest_z = params.ground_z + params.radius_m
        pos = np.array([0.0, 0.0, rest_z], dtype=np.float64)
        vel = np.zeros(3)
        sky = SimpleNamespace(wind_dir=(1.0, 0.0), wind_speed=wind_speed,
                              rain_intensity=rain, cloud_coverage=cov,
                              cloud_density=den)
        for i in range(200):
            t = i * 0.02
            field.update(0.02, t, sky, (float(pos[0]), float(pos[1]),
                                        float(pos[2])))
            v = field.sample(pos[None])[0]
            pos, vel = debug_ball_step(pos, vel, v, 0.02, params)
        return float(pos[0])

    calm = _run(wind_speed=2.5, rain=0.0, cov=0.0, den=0.0)
    storm = _run(wind_speed=12.0, rain=1.0, cov=1.0, den=1.0)
    assert storm > calm


def test_exports_present():
    """BallParams + debug_ball_step are exported from the package root."""
    assert debug_ball_step is _direct_import
    assert BallParams(ground_z=8.0).ground_z == 8.0
