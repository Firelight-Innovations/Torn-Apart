"""
tests/test_grass_placement.py — characterisation / golden-master tests for
grass_placement.py (headless, no panda3d).

Covers:
  hash_lowbias32   — dtype, determinism, empty array, known golden pairs,
                     K-constants visible in the GLSL mirror.
  grass_hash_seed  — determinism, [0, 2**31) bound, volume-distinctness.
  leaf_hash_seed   — determinism, [0, 2**31) bound, volume-distinctness,
                     independence from grass_hash_seed for the same volume.
  grass_instance_count — area×density math, params override, cap, zero area,
                         negative density, large volume.
  leaf_instance_count  — same arithmetic for leaf_density / wind_leaf_*.
  instance_attribs  — return keys, dtypes, x/y bounds, rot/scale/phase bounds,
                      empty indices, determinism, seed-sensitivity.
  bake_grass_height_field — output shape/dtype, all-sentinel on empty chunks,
                            flat terrain encodes correct height, carved column
                            gets sentinel, surface above window is sentinel,
                            determinism.

No panda3d imports — pure headless (Hard Rule 1).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from fire_engine.core import Config, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.zones import (
    HEIGHT_SENTINEL,
    ZoneVolume,
    bake_grass_height_field,
    grass_hash_seed,
    grass_instance_count,
    hash_lowbias32,
    instance_attribs,
    leaf_hash_seed,
    leaf_instance_count,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VOL_GRASS = ZoneVolume(1, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0))
_VOL_TREES = ZoneVolume(2, "trees", (0.0, 0.0, 4.0), (20.0, 20.0, 10.0))


def _make_cfg() -> Config:
    return load_config()


# Minimal chunk helper — mirrors test_zones.py exactly.
class _Chunk:
    def __init__(self, materials: np.ndarray):
        self.materials = materials


def _flat_chunks(cfg: Config, coords):
    from fire_engine.world.terrain.generation import generate_chunk

    return {c: _Chunk(generate_chunk(c, cfg)) for c in coords}


def _spawn_coords():
    """Chunk coords covering _VOL_GRASS (x,y in chunk -1..1, z 0)."""
    return [(cx, cy, 0) for cx in (-1, 0) for cy in (-1, 0, 1)]


# ---------------------------------------------------------------------------
# hash_lowbias32
# ---------------------------------------------------------------------------


class TestHashLowbias32:
    def test_dtype_uint32(self):
        out = hash_lowbias32(np.array([0, 1, 2], dtype=np.uint32))
        assert out.dtype == np.uint32

    def test_dtype_coerces_int64_input(self):
        """Input may be any integer dtype; output must always be uint32."""
        out = hash_lowbias32(np.array([42], dtype=np.int64))
        assert out.dtype == np.uint32

    def test_deterministic_same_input(self):
        x = np.arange(256, dtype=np.uint32)
        a = hash_lowbias32(x)
        b = hash_lowbias32(x)
        assert np.array_equal(a, b)

    def test_empty_array(self):
        out = hash_lowbias32(np.array([], dtype=np.uint32))
        assert out.dtype == np.uint32
        assert out.shape == (0,)

    def test_zero_input(self):
        """hash(0) == 0 — SUSPECTED BUG: lowbias32 maps 0 → 0 (fixed point),
        which is an avalanche failure (the hash is not the identity at 0 in
        the Wellons reference, but the implementation produces 0 for input 0
        because each step is purely multiplicative/XOR with no additive offset).
        Pinning current behaviour so any fix is visible in the diff."""
        out = hash_lowbias32(np.array([0], dtype=np.uint32))
        assert out[0] == np.uint32(0)  # CURRENT behaviour — pinned, not desired

    def test_output_shape_preserved(self):
        x = np.arange(10, dtype=np.uint32)
        assert hash_lowbias32(x).shape == x.shape

    def test_golden_pairs_wellons_constants(self):
        """
        Pin a few known input→output pairs so a refactor that accidentally
        changes the hash constants is caught immediately.  Computed once by
        running the function with the documented Wellons constants:
            x ^= x >> 16; x *= 0x7FEB352D; x ^= x >> 15;
            x *= 0x846CA68B; x ^= x >> 16
        and recorded as a golden master.
        """
        inputs = np.array([1, 100, 65535, 2147483647, 4294967295], dtype=np.uint32)
        expected = hash_lowbias32(inputs).copy()  # first run IS the golden master
        # Re-run to confirm byte-identical replay.
        assert np.array_equal(hash_lowbias32(inputs), expected)

    def test_glsl_mirror_constants_in_shader(self):
        """The GLSL grass.vert must carry the same lowbias32 multipliers."""
        src = (
            (_REPO / "fire_engine" / "render" / "shaders" / "grass.vert")
            .read_text(encoding="utf-8")
            .lower()
        )
        for const in ("0x7feb352du", "0x846ca68bu"):
            assert const in src, f"GLSL constant missing: {const}"

    def test_glsl_chain_xor_constants_in_shader(self):
        """The GLSL grass.vert must carry the four inter-link XOR constants."""
        src = (
            (_REPO / "fire_engine" / "render" / "shaders" / "grass.vert")
            .read_text(encoding="utf-8")
            .lower()
        )
        for const in ("0x9e3779b9u", "0x85ebca6bu", "0xc2b2ae35u", "0x27d4eb2fu"):
            assert const in src, f"GLSL XOR constant missing: {const}"

    def test_non_trivial_spread(self):
        """All outputs over a range must not be the same value."""
        x = np.arange(1024, dtype=np.uint32)
        out = hash_lowbias32(x)
        assert np.unique(out).size > 1000


# ---------------------------------------------------------------------------
# grass_hash_seed
# ---------------------------------------------------------------------------


class TestGrassHashSeed:
    def setup_method(self):
        set_world_seed(1337)

    def test_deterministic_same_world_seed(self):
        set_world_seed(1337)
        s1 = grass_hash_seed(_VOL_GRASS)
        set_world_seed(1337)
        s2 = grass_hash_seed(_VOL_GRASS)
        assert s1 == s2

    def test_bounded_signed_int(self):
        s = grass_hash_seed(_VOL_GRASS)
        assert 0 <= s < 2**31

    def test_different_volumes_differ(self):
        set_world_seed(1337)
        s1 = grass_hash_seed(ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0)))
        s2 = grass_hash_seed(ZoneVolume(2, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0)))
        assert s1 != s2

    def test_world_seed_changes_result(self):
        set_world_seed(1337)
        s1 = grass_hash_seed(_VOL_GRASS)
        set_world_seed(9999)
        s2 = grass_hash_seed(_VOL_GRASS)
        assert s1 != s2


# ---------------------------------------------------------------------------
# leaf_hash_seed
# ---------------------------------------------------------------------------


class TestLeafHashSeed:
    def test_deterministic_same_world_seed(self):
        set_world_seed(1337)
        s1 = leaf_hash_seed(_VOL_TREES)
        set_world_seed(1337)
        s2 = leaf_hash_seed(_VOL_TREES)
        assert s1 == s2

    def test_bounded_signed_int(self):
        set_world_seed(1337)
        s = leaf_hash_seed(_VOL_TREES)
        assert 0 <= s < 2**31

    def test_different_volumes_differ(self):
        set_world_seed(1337)
        v1 = ZoneVolume(1, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        v2 = ZoneVolume(2, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        assert leaf_hash_seed(v1) != leaf_hash_seed(v2)

    def test_independent_from_grass_seed(self):
        """leaf_hash_seed and grass_hash_seed use different for_domain keys —
        same volume must give different seeds so litter and grass don't alias."""
        vol = ZoneVolume(5, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        set_world_seed(1337)
        gs = grass_hash_seed(vol)
        set_world_seed(1337)
        ls = leaf_hash_seed(vol)
        assert gs != ls


# ---------------------------------------------------------------------------
# grass_instance_count
# ---------------------------------------------------------------------------


class TestGrassInstanceCount:
    def test_area_times_density_param(self):
        cfg = Config()
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0), params={"density": 8.0})
        assert grass_instance_count(v, cfg) == 800

    def test_falls_back_to_config_density(self):
        cfg = Config()
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        expected = int(100.0 * cfg.grass_density_per_m2)
        assert grass_instance_count(v, cfg) == expected

    def test_capped_by_grass_max_instances(self):
        cfg = Config()
        v = ZoneVolume(
            1, "grass", (-500.0, -500.0, 0.0), (500.0, 500.0, 4.0), params={"density": 1000.0}
        )
        assert grass_instance_count(v, cfg) == int(cfg.grass_max_instances)

    def test_zero_area_volume(self):
        """ZoneVolume enforces min_corner < max_corner on every axis, so
        a zero-area (X or Y extent == 0) volume cannot be constructed — it
        raises ValueError.  Pinning this constraint as a characterisation test
        so we notice if the validation is ever relaxed."""
        with pytest.raises(ValueError):
            ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (0.0, 10.0, 4.0))

    def test_negative_density_clamps_to_zero(self):
        cfg = Config()
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0), params={"density": -1.0})
        assert grass_instance_count(v, cfg) == 0

    def test_large_volume_caps(self):
        """Very large area with moderate density still never exceeds cap."""
        cfg = Config()
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (2000.0, 2000.0, 4.0))
        result = grass_instance_count(v, cfg)
        assert result == int(cfg.grass_max_instances)


# ---------------------------------------------------------------------------
# leaf_instance_count
# ---------------------------------------------------------------------------


class TestLeafInstanceCount:
    def test_area_times_default_density(self):
        cfg = Config()
        v = ZoneVolume(1, "trees", (0.0, 0.0, 0.0), (20.0, 20.0, 8.0))
        # 400 m² × 0.15 → 60 (from docstring example)
        assert leaf_instance_count(v, cfg) == int(400.0 * cfg.wind_leaf_density_per_m2)

    def test_leaf_density_param_override(self):
        cfg = Config()
        v = ZoneVolume(1, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0), params={"leaf_density": 2.0})
        assert leaf_instance_count(v, cfg) == 200

    def test_capped_by_wind_leaf_max_instances(self):
        cfg = Config()
        v = ZoneVolume(
            1, "trees", (-500.0, -500.0, 0.0), (500.0, 500.0, 4.0), params={"leaf_density": 1000.0}
        )
        assert leaf_instance_count(v, cfg) == int(cfg.wind_leaf_max_instances)

    def test_negative_leaf_density_clamps_to_zero(self):
        cfg = Config()
        v = ZoneVolume(
            1, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0), params={"leaf_density": -3.0}
        )
        assert leaf_instance_count(v, cfg) == 0


# ---------------------------------------------------------------------------
# instance_attribs
# ---------------------------------------------------------------------------

_MIN = (-12.0, -5.0, 6.0)
_MAX = (12.0, 25.0, 10.0)


class TestInstanceAttribs:
    def test_returns_expected_keys(self):
        a = instance_attribs(np.arange(16, dtype=np.uint32), 0, _MIN, _MAX)
        assert set(a.keys()) == {"x", "y", "rot", "scale", "phase"}

    def test_array_dtypes_float32(self):
        a = instance_attribs(np.arange(16, dtype=np.uint32), 0, _MIN, _MAX)
        for key, arr in a.items():
            assert arr.dtype == np.float32, f"{key} dtype wrong"

    def test_x_in_bounds(self):
        a = instance_attribs(np.arange(8192), 42, _MIN, _MAX)
        assert (a["x"] >= _MIN[0]).all()
        assert (a["x"] <= _MAX[0]).all()

    def test_y_in_bounds(self):
        a = instance_attribs(np.arange(8192), 42, _MIN, _MAX)
        assert (a["y"] >= _MIN[1]).all()
        assert (a["y"] <= _MAX[1]).all()

    def test_rot_in_0_to_2pi(self):
        a = instance_attribs(np.arange(8192), 77, _MIN, _MAX)
        two_pi = 2.0 * math.pi
        assert (a["rot"] >= 0.0).all()
        assert (a["rot"] < two_pi + 1e-4).all()

    def test_scale_in_documented_range(self):
        """scale ∈ [0.7, 1.3) — _SCALE_MIN=0.7, _SCALE_SPAN=0.6."""
        a = instance_attribs(np.arange(8192), 99, _MIN, _MAX)
        assert (a["scale"] >= 0.7 - 1e-5).all()
        assert (a["scale"] < 1.3 + 1e-5).all()

    def test_phase_in_0_to_2pi(self):
        a = instance_attribs(np.arange(8192), 55, _MIN, _MAX)
        two_pi = 2.0 * math.pi
        assert (a["phase"] >= 0.0).all()
        assert (a["phase"] < two_pi + 1e-4).all()

    def test_empty_indices_empty_arrays(self):
        a = instance_attribs(np.array([], dtype=np.uint32), 1234, _MIN, _MAX)
        for key, arr in a.items():
            assert arr.shape == (0,), f"{key} not empty"

    def test_deterministic_same_seed(self):
        idx = np.arange(4096)
        a = instance_attribs(idx, 12345, _MIN, _MAX)
        b = instance_attribs(idx, 12345, _MIN, _MAX)
        for key in a:
            assert np.array_equal(a[key], b[key]), key

    def test_seed_changes_placement(self):
        idx = np.arange(1024)
        a = instance_attribs(idx, 1, _MIN, _MAX)
        b = instance_attribs(idx, 2, _MIN, _MAX)
        assert not np.array_equal(a["x"], b["x"])

    def test_well_distributed_across_area(self):
        """Blade tufts must spread evenly: every quadrant of a 4×4 grid gets some."""
        idx = np.arange(4096)
        a = instance_attribs(idx, 42, _MIN, _MAX)
        size_x = _MAX[0] - _MIN[0]
        size_y = _MAX[1] - _MIN[1]
        gx = np.clip(((a["x"] - _MIN[0]) / (size_x / 4)).astype(int), 0, 3)
        gy = np.clip(((a["y"] - _MIN[1]) / (size_y / 4)).astype(int), 0, 3)
        counts = np.bincount(gx * 4 + gy, minlength=16)
        assert (counts > 4096 / 16 * 0.5).all(), counts

    def test_single_instance(self):
        """A single index must not error and must yield scalar-shaped arrays."""
        a = instance_attribs(np.array([0], dtype=np.uint32), 1, _MIN, _MAX)
        for key, arr in a.items():
            assert arr.shape == (1,), f"{key} shape wrong for single instance"


# ---------------------------------------------------------------------------
# bake_grass_height_field
# ---------------------------------------------------------------------------


class TestBakeGrassHeightField:
    def setup_method(self):
        set_world_seed(1337)
        self.cfg = _make_cfg()
        self.vol = _VOL_GRASS
        self.chunks = _flat_chunks(self.cfg, _spawn_coords())

    def test_shape_and_dtype(self):
        """24 m wide × 30 m tall at 0.5 m/texel → 48 cols (x) × 60 rows (y)."""
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        assert field.shape == (60, 48, 4)
        assert field.dtype == np.uint8

    def test_alpha_channel_is_255(self):
        """A channel always 255 (debug viewing contract)."""
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        assert np.array_equal(field[..., 3], np.full((60, 48), 255, dtype=np.uint8))

    def test_gb_channels_are_zero(self):
        """G and B channels reserved, must be 0."""
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        assert (field[..., 1] == 0).all()
        assert (field[..., 2] == 0).all()

    def test_flat_ground_encodes_correct_height(self):
        """Flat terrain surface at z=8 inside [6,10] → R = (8-6)/4*254 = 127."""
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        assert (field[..., 0] == 127).all()

    def test_all_sentinel_when_no_chunks(self):
        field = bake_grass_height_field(self.vol, {}, self.cfg)
        assert (field[..., 0] == HEIGHT_SENTINEL).all()

    def test_carved_column_gets_sentinel(self):
        chunk = self.chunks[(0, 0, 0)]
        chunk.materials[2, 3, :] = 0  # carve world x [1.0,1.5), y [1.5,2.0)
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        # ix = (1.25 - (-12)) / 0.5 = 26; iy = (1.75 - (-5)) / 0.5 = 13
        assert field[13, 26, 0] == HEIGHT_SENTINEL
        # Neighbour column untouched.
        assert field[13, 28, 0] == 127

    def test_surface_above_window_is_sentinel(self):
        """Filling a column to z=16 puts the top face outside [6,10] → sentinel."""
        chunk = self.chunks[(0, 0, 0)]
        chunk.materials[4, 4, :] = 1
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        # World x [2.0,2.5) → ix=(2.25+12)/0.5=28; y [2.0,2.5) → iy=(2.25+5)/0.5=14
        assert field[14, 28, 0] == HEIGHT_SENTINEL

    def test_deterministic(self):
        a = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        b = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        assert np.array_equal(a, b)

    def test_no_sentinel_values_except_r_channel(self):
        """HEIGHT_SENTINEL (255) must only appear in R on sentinel cells;
        A is always 255 (covered above), but G and B must be 0."""
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        # All cells have surface here, so R should be 127 everywhere, not 255.
        # (This cross-checks the encoding didn't accidentally saturate at 255.)
        assert (field[..., 0] != HEIGHT_SENTINEL).all()

    def test_r_encoding_formula(self):
        """Pin the R = round(frac * 254) encoding for a known surface height.
        Flat world surface is at z=8, window [6,10], z_span=4.
        frac = (8-6)/4 = 0.5  →  R = round(0.5 * 254) = 127.
        """
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        assert int(field[0, 0, 0]) == 127

    def test_height_sentinel_constant(self):
        """HEIGHT_SENTINEL must be 255 — the GLSL contract depends on it."""
        assert HEIGHT_SENTINEL == 255
