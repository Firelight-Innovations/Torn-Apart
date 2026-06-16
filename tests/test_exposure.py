"""
Headless tests for lighting/exposure.py — auto-exposure / eye adaptation.

Covers: determinism, noon open-field neutrality, slow dark adaptation in a
sealed cave, fast bright re-adaptation, point lights suppressing adaptation,
None sky_state behaviour, and numeric robustness at dt=0 / dt=10.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.lighting.exposure import ExposureMeter

CHUNK = 32  # voxels per chunk edge


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def make_config() -> Config:
    """Default engine config (exposure keys fall back to module defaults)."""
    return Config()


def noon_sky() -> SimpleNamespace:
    """Clear noon sky: bright sun nearly overhead, no moon."""
    sun_dir = np.array([0.0, -0.34, 0.94])
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    return SimpleNamespace(
        sun_radiance=(3.2, 3.0, 2.6),
        sky_ambient=(0.21, 0.40, 0.71),
        moon_radiance=(0.0, 0.0, 0.0),
        sun_dir=SimpleNamespace(x=float(sun_dir[0]), y=float(sun_dir[1]), z=float(sun_dir[2])),
    )


def cave_chunks() -> dict[tuple[int, int, int], SimpleNamespace]:
    """Chunk (0,0,0) is solid rock with a small air pocket around (8,8,8) m.

    Probe rays only travel upward/outward, so a single sealed chunk around
    the camera blocks every ray before it can escape into (missing == air)
    neighbour chunks.
    """
    materials = np.ones((CHUNK, CHUNK, CHUNK), dtype=np.uint8)
    materials[13:20, 13:20, 13:20] = 0  # ~3.5 m air pocket, camera inside
    return {(0, 0, 0): SimpleNamespace(materials=materials)}


CAVE_CAM = (8.0, 8.0, 8.0)  # center of the air pocket, meters
OPEN_CAM = (8.0, 8.0, 30.0)  # floating in open air (no chunks)

NO_LIGHTS = (np.zeros((4, 12), dtype=np.float32), 0)


def step(meter: ExposureMeter, cam, sky, chunks, lights, seconds: float, dt: float = 0.1) -> float:
    """Run update() repeatedly for `seconds` of simulated time."""
    val = meter.exposure
    n = round(seconds / dt)
    for _ in range(n):
        val = meter.update(cam, sky, chunks, lights, dt)
    return val


# ----------------------------------------------------------------------
# (a) determinism
# ----------------------------------------------------------------------


def test_determinism_same_inputs_identical_result():
    cfg = make_config()
    results = []
    for _ in range(2):
        meter = ExposureMeter(cfg)
        sky = noon_sky()
        chunks = cave_chunks()
        v1 = meter.update(CAVE_CAM, sky, chunks, NO_LIGHTS, 0.1)
        v2 = meter.update(OPEN_CAM, sky, {}, NO_LIGHTS, 0.1)
        results.append((v1, v2))
    assert results[0] == results[1]  # bit-identical floats


# ----------------------------------------------------------------------
# (b) noon open field ~ 1.0
# ----------------------------------------------------------------------


def test_noon_open_field_stays_near_one():
    cfg = make_config()
    meter = ExposureMeter(cfg)
    val = step(meter, OPEN_CAM, noon_sky(), {}, NO_LIGHTS, seconds=10.0)
    assert 0.75 <= val <= 1.25
    assert meter.exposure == val


# ----------------------------------------------------------------------
# (c) sealed cave: climbs slowly toward exposure_max
# ----------------------------------------------------------------------


def test_cave_adapts_slowly_toward_max():
    cfg = make_config()
    meter = ExposureMeter(cfg)
    sky = noon_sky()
    chunks = cave_chunks()
    # After 0.5 s the aperture has barely opened (tau_dark = 4 s).
    val_half = step(meter, CAVE_CAM, sky, chunks, NO_LIGHTS, seconds=0.5)
    assert val_half > 1.0
    assert val_half < 1.5
    # After ~20 s total it has converged close to exposure_max (5.0).
    val_long = step(meter, CAVE_CAM, sky, chunks, NO_LIGHTS, seconds=19.5)
    assert val_long > 4.0
    assert val_long <= 5.0 + 1e-9


# ----------------------------------------------------------------------
# (d) back outside: drops near 1.0 quickly
# ----------------------------------------------------------------------


def test_back_outside_drops_fast():
    cfg = make_config()
    meter = ExposureMeter(cfg)
    sky = noon_sky()
    chunks = cave_chunks()
    step(meter, CAVE_CAM, sky, chunks, NO_LIGHTS, seconds=20.0)
    assert meter.exposure > 4.0  # fully dark-adapted
    val = step(meter, OPEN_CAM, sky, {}, NO_LIGHTS, seconds=2.0)
    assert val < 1.3  # tau_bright = 0.7 s -> ~95% of the way back in 2 s


# ----------------------------------------------------------------------
# (e) bright light inside the cave caps adaptation
# ----------------------------------------------------------------------


def test_light_in_cave_keeps_exposure_low():
    cfg = make_config()
    meter = ExposureMeter(cfg)
    sky = noon_sky()
    chunks = cave_chunks()
    lights = np.zeros((4, 12), dtype=np.float32)
    lights[0, 0:3] = (9.0, 8.0, 8.0)  # 1 m from the camera
    lights[0, 3] = 10.0  # falloff radius, meters
    lights[0, 4:7] = (8.0, 8.0, 8.0)  # color * intensity, HDR
    packed = (lights, 1)
    val = step(meter, CAVE_CAM, sky, chunks, packed, seconds=20.0)
    assert val < 2.0  # well below exposure_max despite total sky occlusion


# ----------------------------------------------------------------------
# (f) None sky_state -> 1.0
# ----------------------------------------------------------------------


def test_none_sky_state_holds_one():
    cfg = make_config()
    meter = ExposureMeter(cfg)
    val = meter.update(OPEN_CAM, None, {}, NO_LIGHTS, 0.1)
    assert val == pytest.approx(1.0)
    # And a dark-adapted meter decays back toward 1.0 when sky goes None.
    meter2 = ExposureMeter(cfg)
    step(meter2, CAVE_CAM, noon_sky(), cave_chunks(), NO_LIGHTS, seconds=20.0)
    assert meter2.exposure > 4.0
    val2 = step(meter2, CAVE_CAM, None, cave_chunks(), NO_LIGHTS, seconds=5.0)
    assert val2 == pytest.approx(1.0, abs=0.05)


# ----------------------------------------------------------------------
# (g) numeric robustness: dt=0 and dt=10
# ----------------------------------------------------------------------


def test_dt_zero_and_dt_huge_no_nans():
    cfg = make_config()
    sky = noon_sky()
    chunks = cave_chunks()

    meter = ExposureMeter(cfg)
    before = meter.exposure
    val0 = meter.update(CAVE_CAM, sky, chunks, NO_LIGHTS, 0.0)
    assert np.isfinite(val0)
    assert val0 == before  # dt=0 -> no change

    val10 = meter.update(CAVE_CAM, sky, chunks, NO_LIGHTS, 10.0)
    assert np.isfinite(val10)
    assert 0.55 - 1e-9 <= val10 <= 5.0 + 1e-9

    # Big dt in the bright direction too.
    val_out = meter.update(OPEN_CAM, sky, {}, NO_LIGHTS, 10.0)
    assert np.isfinite(val_out)
    assert 0.5 <= val_out <= 1.5


# ----------------------------------------------------------------------
# extra: exposure_adapt_enabled=False pins the multiplier at 1.0
# ----------------------------------------------------------------------


def test_disabled_decays_to_one():
    cfg = SimpleNamespace(exposure_adapt_enabled=False, chunk_size=32, voxel_size=0.5)
    meter = ExposureMeter(cfg)
    val = step(meter, CAVE_CAM, noon_sky(), cave_chunks(), NO_LIGHTS, seconds=5.0)
    assert val == pytest.approx(1.0)
