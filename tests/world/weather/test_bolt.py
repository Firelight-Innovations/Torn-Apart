"""
tests/world/weather/test_bolt.py — Mirror for fire_engine/world/weather/bolt.py.

Authored tests covering BoltGeometry dataclass and generate_bolt() function.
Headless — no panda3d imports.

Coverage
--------
CORRECTNESS:
  - BoltGeometry fields have the expected shapes and dtypes.
  - generate_bolt returns a BoltGeometry with at least one segment.
  - Main channel (is_main) exists and reaches ground_z.
  - Branches (if any) stay above ground_z.
  - width and brightness are all positive.
  - The main channel is at least as wide/bright as any branch.

DETERMINISM:
  - Same seed → byte-identical geometry (a, b, width, brightness, is_main).
  - Different seeds → different geometry.

CORRECTNESS (step budget):
  - Total segments never exceed bolt_max_steps.

CORRECTNESS (geometry direction):
  - Main channel descends from start to ground_z.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather.bolt import BoltGeometry, generate_bolt

_START = (0.0, 0.0, 220.0)
_GROUND_Z = 8.0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def bolt(cfg):
    """A single bolt with a fixed seed, shared across tests in this module."""
    set_world_seed(1337)
    return generate_bolt(42, _START, _GROUND_Z, cfg)


# ---------------------------------------------------------------------------
# BoltGeometry structure
# ---------------------------------------------------------------------------


class TestBoltGeometryShape:
    def test_a_is_float32_nx3(self, bolt: BoltGeometry):
        assert bolt.a.ndim == 2 and bolt.a.shape[1] == 3
        assert bolt.a.dtype == np.float32

    def test_b_is_float32_nx3(self, bolt: BoltGeometry):
        assert bolt.b.ndim == 2 and bolt.b.shape[1] == 3
        assert bolt.b.dtype == np.float32

    def test_width_is_float32_n(self, bolt: BoltGeometry):
        assert bolt.width.ndim == 1
        assert bolt.width.dtype == np.float32

    def test_brightness_is_float32_n(self, bolt: BoltGeometry):
        assert bolt.brightness.ndim == 1
        assert bolt.brightness.dtype == np.float32

    def test_is_main_is_bool_n(self, bolt: BoltGeometry):
        assert bolt.is_main.ndim == 1
        assert bolt.is_main.dtype == bool

    def test_all_arrays_same_length(self, bolt: BoltGeometry):
        n = len(bolt.a)
        assert len(bolt.b) == n
        assert len(bolt.width) == n
        assert len(bolt.brightness) == n
        assert len(bolt.is_main) == n

    def test_at_least_one_segment(self, bolt: BoltGeometry):
        assert len(bolt.a) > 0

    def test_len_matches_array_length(self, bolt: BoltGeometry):
        assert len(bolt) == len(bolt.a)


# ---------------------------------------------------------------------------
# Main channel correctness
# ---------------------------------------------------------------------------


class TestMainChannel:
    def test_main_channel_exists(self, bolt: BoltGeometry):
        assert bolt.is_main.any(), "no main return-stroke channel"

    def test_main_channel_reaches_ground_z(self, bolt: BoltGeometry):
        main_b = bolt.b[bolt.is_main]
        assert main_b[:, 2].min() == pytest.approx(_GROUND_Z, abs=1e-3)

    def test_main_channel_descends_overall(self, bolt: BoltGeometry):
        """The main channel as a whole must trend downward."""
        main_a = bolt.a[bolt.is_main]
        main_b = bolt.b[bolt.is_main]
        # The first segment starts near _START z; last segment ends at ground_z.
        assert main_a[0, 2] > main_b[-1, 2]

    def test_branches_stay_above_ground(self, bolt: BoltGeometry):
        if (~bolt.is_main).any():
            branch_b = bolt.b[~bolt.is_main]
            assert branch_b[:, 2].min() > _GROUND_Z + 1.0

    def test_width_all_positive(self, bolt: BoltGeometry):
        assert np.all(bolt.width > 0.0)

    def test_brightness_all_positive(self, bolt: BoltGeometry):
        assert np.all(bolt.brightness > 0.0)

    def test_main_channel_widest(self, bolt: BoltGeometry):
        """Main channel segments are at least as wide as any branch."""
        max_main = bolt.width[bolt.is_main].max()
        assert max_main >= bolt.width.max() - 1e-6

    def test_all_values_finite(self, bolt: BoltGeometry):
        for arr in (bolt.a, bolt.b, bolt.width, bolt.brightness):
            assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_byte_identical(self, cfg):
        set_world_seed(1337)
        a = generate_bolt(99, _START, _GROUND_Z, cfg)
        b = generate_bolt(99, _START, _GROUND_Z, cfg)
        assert np.array_equal(a.a, b.a)
        assert np.array_equal(a.b, b.b)
        assert np.array_equal(a.width, b.width)
        assert np.array_equal(a.brightness, b.brightness)
        assert np.array_equal(a.is_main, b.is_main)

    def test_different_seeds_differ(self, cfg):
        set_world_seed(1337)
        bolt_a = generate_bolt(1, _START, _GROUND_Z, cfg)
        bolt_b = generate_bolt(2, _START, _GROUND_Z, cfg)
        # Different bolt seeds must produce different geometry.
        differs = bolt_a.a.shape != bolt_b.a.shape or not np.array_equal(bolt_a.b, bolt_b.b)
        assert differs, "two different seeds produced identical bolt geometry"

    def test_same_world_seed_same_bolt_for_same_bolt_seed(self, cfg):
        """Same world seed + same bolt seed must produce identical geometry."""
        set_world_seed(1337)
        bolt1 = generate_bolt(42, _START, _GROUND_Z, cfg)
        set_world_seed(1337)
        bolt2 = generate_bolt(42, _START, _GROUND_Z, cfg)
        assert np.array_equal(bolt1.a, bolt2.a)
        assert np.array_equal(bolt1.b, bolt2.b)


# ---------------------------------------------------------------------------
# Step budget
# ---------------------------------------------------------------------------


class TestStepBudget:
    def test_segments_within_max_steps(self, cfg):
        set_world_seed(77)
        b = generate_bolt(7, _START, _GROUND_Z, cfg)
        assert len(b) <= int(cfg.bolt_max_steps)

    def test_multiple_seeds_within_budget(self, cfg):
        set_world_seed(1337)
        for seed in range(5):
            b = generate_bolt(seed, _START, _GROUND_Z, cfg)
            assert len(b) <= int(cfg.bolt_max_steps), (
                f"seed {seed}: {len(b)} segments > bolt_max_steps {cfg.bolt_max_steps}"
            )
