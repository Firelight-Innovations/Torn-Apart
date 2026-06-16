"""
tests/zones/test_flora_placement.py — unit tests for fire_engine/zones/flora_placement.py.

Categories (CLAUDE.md):
- DETERMINISM: flora_hash_seed and flora_instance_attribs produce byte-identical
  results for identical (world-seed, volume-id, kind) inputs.
- CORRECTNESS: instance attributes stay within their documented bounds;
  variant values are in [0, n_variants); flora_instance_count matches density
  formula; FLORA_KINDS constant is non-empty.
- Seed sensitivity: different seeds produce different placements.

No panda3d imports (Hard Rule 1).
All randomness through core.rng.for_domain (Hard Rule 2 — flora_hash_seed does this).
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.core import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.zones.flora_placement import (
    FLORA_KINDS,
    flora_hash_seed,
    flora_instance_attribs,
    flora_instance_count,
)
from fire_engine.zones.volume import ZoneVolume

# Reference volume used across tests
_VOL = ZoneVolume(1, "flowers", (-10.0, -10.0, 0.0), (10.0, 10.0, 8.0))


# ---------------------------------------------------------------------------
# FLORA_KINDS constant
# ---------------------------------------------------------------------------


class TestFloraKinds:
    def test_flora_kinds_is_nonempty_tuple(self):
        assert isinstance(FLORA_KINDS, tuple)
        assert len(FLORA_KINDS) > 0

    def test_flowers_in_flora_kinds(self):
        assert "flowers" in FLORA_KINDS


# ---------------------------------------------------------------------------
# flora_hash_seed — DETERMINISM
# ---------------------------------------------------------------------------


class TestFloraHashSeed:
    def test_same_seed_same_result(self):
        set_world_seed(1337)
        s1 = flora_hash_seed(_VOL, "flowers")
        set_world_seed(1337)
        s2 = flora_hash_seed(_VOL, "flowers")
        assert s1 == s2

    def test_result_in_signed_int_range(self):
        set_world_seed(42)
        s = flora_hash_seed(_VOL, "flowers")
        assert 0 <= s < 2**31

    def test_different_volume_id_gives_different_seed(self):
        set_world_seed(1337)
        s1 = flora_hash_seed(_VOL, "flowers")
        other = ZoneVolume(2, "flowers", _VOL.min_corner, _VOL.max_corner)
        set_world_seed(1337)
        s2 = flora_hash_seed(other, "flowers")
        assert s1 != s2

    def test_different_world_seed_gives_different_seed(self):
        set_world_seed(1)
        s1 = flora_hash_seed(_VOL, "flowers")
        set_world_seed(2)
        s2 = flora_hash_seed(_VOL, "flowers")
        assert s1 != s2


# ---------------------------------------------------------------------------
# flora_instance_count — CORRECTNESS
# ---------------------------------------------------------------------------


class TestFloraInstanceCount:
    def setup_method(self):
        self.cfg = Config()

    def test_density_times_area(self):
        # 20×20 = 400 m², default density 1.5 plants/m² → 600
        v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (20.0, 20.0, 8.0))
        count = flora_instance_count(v, self.cfg, "flowers")
        assert count == 600

    def test_custom_density_via_params(self):
        v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 8.0), params={"density": 4.0})
        count = flora_instance_count(v, self.cfg, "flowers")
        assert count == 400  # 100 m² × 4.0

    def test_clamped_by_max_instances(self):
        # Very large volume + high density must not exceed cap
        v = ZoneVolume(
            1, "flowers", (-1000.0, -1000.0, 0.0), (1000.0, 1000.0, 8.0), params={"density": 1000.0}
        )
        count = flora_instance_count(v, self.cfg, "flowers")
        assert count == self.cfg.flora_flower_max_instances

    def test_zero_density_gives_zero(self):
        v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 8.0), params={"density": 0.0})
        assert flora_instance_count(v, self.cfg, "flowers") == 0

    def test_negative_density_clamps_to_zero(self):
        v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 8.0), params={"density": -5.0})
        assert flora_instance_count(v, self.cfg, "flowers") == 0


# ---------------------------------------------------------------------------
# flora_instance_attribs — DETERMINISM + CORRECTNESS
# ---------------------------------------------------------------------------


class TestFloraInstanceAttribs:
    def setup_method(self):
        set_world_seed(1337)
        self.seed = flora_hash_seed(_VOL, "flowers")
        self.idx = np.arange(2048)

    def _attribs(self, idx=None, seed=None, n_variants=4):
        return flora_instance_attribs(
            self.idx if idx is None else idx,
            self.seed if seed is None else seed,
            _VOL.min_corner,
            _VOL.max_corner,
            n_variants=n_variants,
        )

    def test_deterministic(self):
        a = self._attribs()
        b = self._attribs()
        for key in a:
            assert np.array_equal(a[key], b[key]), f"Non-deterministic key: {key}"

    def test_different_seed_different_placement(self):
        a = self._attribs()
        b = self._attribs(seed=self.seed + 1)
        assert not np.array_equal(a["x"], b["x"])

    def test_x_in_volume_bounds(self):
        a = self._attribs()
        assert (a["x"] >= _VOL.min_corner[0]).all()
        assert (a["x"] <= _VOL.max_corner[0]).all()

    def test_y_in_volume_bounds(self):
        a = self._attribs()
        assert (a["y"] >= _VOL.min_corner[1]).all()
        assert (a["y"] <= _VOL.max_corner[1]).all()

    def test_rot_in_0_2pi(self):
        a = self._attribs()
        assert (a["rot"] >= 0.0).all()
        assert (a["rot"] < 2.0 * math.pi + 1e-4).all()

    def test_scale_in_default_range(self):
        # Default scale_min=0.7, scale_span=0.6 → [0.7, 1.3)
        a = self._attribs()
        assert (a["scale"] >= 0.7 - 1e-5).all()
        assert (a["scale"] <= 1.3 + 1e-5).all()

    def test_phase_in_0_2pi(self):
        a = self._attribs()
        assert (a["phase"] >= 0.0).all()
        assert (a["phase"] < 2.0 * math.pi + 1e-4).all()

    def test_variant_in_0_n_variants(self):
        n = 4
        a = self._attribs(n_variants=n)
        assert (a["variant"] >= 0).all()
        assert (a["variant"] < n).all()

    def test_variant_distributes_across_all_cells(self):
        # With 2048 instances and 4 variants each cell should get some hits.
        n = 4
        a = self._attribs(idx=np.arange(2048), n_variants=n)
        for v in range(n):
            assert (a["variant"] == v).sum() > 0, f"Variant {v} never selected"

    def test_single_variant_all_zero(self):
        a = self._attribs(n_variants=1)
        assert (a["variant"] == 0).all()

    def test_output_keys_present(self):
        a = self._attribs()
        assert set(a.keys()) == {"x", "y", "rot", "scale", "phase", "variant"}

    def test_custom_scale_range(self):
        a = flora_instance_attribs(
            self.idx,
            self.seed,
            _VOL.min_corner,
            _VOL.max_corner,
            n_variants=2,
            scale_min=1.0,
            scale_span=2.0,
        )
        assert (a["scale"] >= 1.0 - 1e-5).all()
        assert (a["scale"] <= 3.0 + 1e-5).all()

    def test_well_distributed_in_xy(self):
        # 2048 plants over a 4×4 grid on the XY plane — every cell should be hit.
        idx = np.arange(2048)
        a = self._attribs(idx=idx)
        size_x = _VOL.max_corner[0] - _VOL.min_corner[0]
        size_y = _VOL.max_corner[1] - _VOL.min_corner[1]
        gx = np.clip(((a["x"] - _VOL.min_corner[0]) / (size_x / 4)).astype(int), 0, 3)
        gy = np.clip(((a["y"] - _VOL.min_corner[1]) / (size_y / 4)).astype(int), 0, 3)
        counts = np.bincount(gx * 4 + gy, minlength=16)
        assert (counts > 2048 / 16 * 0.4).all(), f"Poor distribution: {counts}"
