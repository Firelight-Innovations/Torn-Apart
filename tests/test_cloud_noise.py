"""Volumetric cloud noise bake: determinism, tileability, shape, coverage.

The bake (``fire_engine.world.sky.cloud_noise``) feeds the volumetric cloud raymarch.
It must be deterministic (same seed → byte-identical, so worlds reproduce) and
tileable (no seam as the field scrolls with the wind).  Pure numpy / headless.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fire_engine.core.rng import set_world_seed
from fire_engine.world.sky.cloud_noise import bake_shape_noise, bake_detail_noise

_ROOT = Path(__file__).resolve().parents[1]


def test_shape_determinism_same_seed():
    set_world_seed(1337)
    a = bake_shape_noise(48)
    set_world_seed(1337)
    b = bake_shape_noise(48)
    assert np.array_equal(a, b)


def test_detail_determinism_same_seed():
    set_world_seed(7)
    a = bake_detail_noise(32)
    set_world_seed(7)
    b = bake_detail_noise(32)
    assert np.array_equal(a, b)


def test_different_seed_differs():
    set_world_seed(1)
    a = bake_shape_noise(48)
    set_world_seed(2)
    b = bake_shape_noise(48)
    assert not np.array_equal(a, b)


def test_shape_dtype_and_range():
    set_world_seed(1337)
    vol = bake_shape_noise(64)
    assert vol.shape == (64, 64, 64, 4)
    assert vol.dtype == np.uint8
    # Every channel carries signal (non-zero variance) at this size.
    for c in range(4):
        assert vol[..., c].std() > 0.0, f"channel {c} is flat"


def test_tileable_no_seam():
    """Wrap-around faces must be ~as continuous as interior neighbours."""
    set_world_seed(99)
    r = bake_shape_noise(64)[..., 0].astype(np.float32)
    for axis in range(3):
        interior = np.abs(np.diff(r, axis=axis)).mean()
        seam = np.abs(np.take(r, 0, axis) - np.take(r, -1, axis)).mean()
        # A real seam would be many times the interior step; tileable ≈ interior.
        assert seam < 4.0 * interior + 1.0, (
            f"axis {axis}: seam {seam:.2f} vs interior {interior:.2f}")


def test_coverage_monotonic():
    """The shader's coverage remap (R > 1-coverage) must be monotonic."""
    set_world_seed(1337)
    r = bake_shape_noise(64)[..., 0].astype(np.float32) / 255.0
    fracs = [float((r > (1.0 - cov)).mean())
             for cov in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert fracs == sorted(fracs), fracs
    assert fracs[-1] > fracs[0]


def test_cross_process_determinism():
    probe = (
        "import numpy as np;"
        "from fire_engine.core.rng import set_world_seed;"
        "from fire_engine.world.sky.cloud_noise import bake_shape_noise;"
        "set_world_seed(2024);"
        "v=bake_shape_noise(32);"
        "print(int(v.astype(np.int64).sum()))"
    )
    outs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, "-c", probe], cwd=str(_ROOT),
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        outs.append(p.stdout.strip())
    assert outs[0] == outs[1] and outs[0] != ""
