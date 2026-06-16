"""
tests/test_zones.py — Zone volumes, store saves, GPU-grass placement math.

Test categories (CLAUDE.md):
1. Round-trips — ZoneVolume dict round-trip; ZoneStore through a real
   SaveManager file; baseline delta is empty; old saves without a "zones"
   key still load.
2. Determinism — instance placement and height-field bakes are byte-identical
   for identical inputs (the python hash mirror IS the GPU placement).
3. Correctness fixtures — placement stays in bounds and spreads out;
   height-field encodes the flat-world surface and the carved-crater sentinel.

No panda3d imports — pure headless (Hard Rule 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import Clock, Config, EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.zones import (
    HEIGHT_SENTINEL,
    ZoneStore,
    ZoneVolume,
    bake_grass_height_field,
    grass_hash_seed,
    grass_instance_count,
    instance_attribs,
)

# ---------------------------------------------------------------------------
# ZoneVolume
# ---------------------------------------------------------------------------


class TestZoneVolume:
    def test_dict_round_trip(self):
        v = ZoneVolume(3, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0), params={"density": 9.5})
        assert ZoneVolume.from_dict(v.to_dict()) == v

    def test_biome_round_trip(self):
        v = ZoneVolume(1, "biome", (0.0, 0.0, 0.0), (50.0, 50.0, 16.0), biome="snow")
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2.biome == "snow" and v2 == v

    def test_invalid_corners_raise(self):
        with pytest.raises(ValueError):
            ZoneVolume(1, "grass", (10.0, 0.0, 0.0), (0.0, 5.0, 5.0))

    def test_empty_tag_raises(self):
        with pytest.raises(ValueError):
            ZoneVolume(1, "", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def test_area_and_contains(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 20.0, 4.0))
        assert v.area_xy_m2 == pytest.approx(200.0)
        inside = v.contains_xy(np.array([5.0, -1.0]), np.array([5.0, 5.0]))
        assert inside.tolist() == [True, False]

    def test_intersects_chunk(self):
        v = ZoneVolume(1, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0))
        assert v.intersects_chunk((0, 0, 0), 16.0)  # overlaps origin chunk
        assert v.intersects_chunk((-1, 0, 0), 16.0)  # crosses to -x chunk
        assert not v.intersects_chunk((5, 5, 0), 16.0)  # far away
        assert not v.intersects_chunk((0, 0, 2), 16.0)  # above the z-window


# ---------------------------------------------------------------------------
# ZoneStore + saves
# ---------------------------------------------------------------------------


def _make_save_env(seed: int = 1337):
    from fire_engine.save import SaveManager

    cfg = load_config()
    set_world_seed(seed)
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    sm = SaveManager(cfg, clock)
    return cfg, clock, bus, sm


class TestZoneStore:
    def test_add_remove_query(self):
        store = ZoneStore()
        a = store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0))
        b = store.add("biome", (0.0, 0.0, 0.0), (50.0, 50.0, 16.0), biome="snow")
        assert store.volumes() == (a, b)
        assert store.volumes("grass") == (a,)
        assert store.get(b.id) is b
        assert store.remove(a.id)
        assert not store.remove(a.id)
        assert store.volumes() == (b,)

    def test_version_bumps_on_change(self):
        store = ZoneStore()
        v0 = store.version
        vol = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert store.version > v0
        v1 = store.version
        store.remove(vol.id)
        assert store.version > v1

    def test_baseline_delta_empty(self):
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0))
        store.mark_baseline()
        assert store.get_delta() == {}
        store.add("grass", (20.0, 0.0, 0.0), (28.0, 8.0, 4.0))
        assert store.get_delta() != {}

    def test_save_manager_round_trip(self, tmp_path):
        cfg, clock, bus, sm = _make_save_env()
        store = ZoneStore()
        store.add("grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0), params={"density": 12.0})
        store.mark_baseline()
        store.add("grass", (30.0, 30.0, 6.0), (40.0, 40.0, 10.0))  # deviation
        sm.register(store)
        path = tmp_path / "zones.ta"
        sm.save(str(path))

        cfg2, clock2, bus2, sm2 = _make_save_env()
        store2 = ZoneStore()
        store2.add(
            "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0), params={"density": 12.0}
        )  # boot defaults, as main.py does
        store2.mark_baseline()
        sm2.register(store2)
        sm2.load(str(path))
        assert store2.volumes() == store.volumes()
        assert store2.get_delta() == store.get_delta()

    def test_old_save_without_zones_key_loads(self, tmp_path):
        # A save written before zones existed has no "zones" envelope key:
        # loading must leave the store's fresh boot defaults untouched.
        cfg, clock, bus, sm = _make_save_env()
        path = tmp_path / "old.ta"
        sm.save(str(path))  # nothing registered → no key

        cfg2, clock2, bus2, sm2 = _make_save_env()
        store = ZoneStore()
        default = store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0))
        store.mark_baseline()
        sm2.register(store)
        sm2.load(str(path))
        assert store.volumes() == (default,)

    def test_next_id_survives_round_trip(self):
        store = ZoneStore()
        v1 = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        store.remove(v1.id)
        store.add("grass", (2.0, 0.0, 0.0), (3.0, 1.0, 1.0))  # id 2
        delta = store.get_delta()

        store2 = ZoneStore()
        store2.apply_delta(delta)
        v3 = store2.add("grass", (4.0, 0.0, 0.0), (5.0, 1.0, 1.0))
        assert v3.id == 3  # never reuses ids


# ---------------------------------------------------------------------------
# Instance placement (python mirror of the GLSL hash chain)
# ---------------------------------------------------------------------------

_VOL = ZoneVolume(1, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0))


class TestInstancePlacement:
    def setup_method(self):
        set_world_seed(1337)

    def test_deterministic(self):
        idx = np.arange(4096)
        a = instance_attribs(idx, 12345, _VOL.min_corner, _VOL.max_corner)
        b = instance_attribs(idx, 12345, _VOL.min_corner, _VOL.max_corner)
        for key in a:
            assert np.array_equal(a[key], b[key]), key

    def test_seed_changes_placement(self):
        idx = np.arange(1024)
        a = instance_attribs(idx, 1, _VOL.min_corner, _VOL.max_corner)
        b = instance_attribs(idx, 2, _VOL.min_corner, _VOL.max_corner)
        assert not np.array_equal(a["x"], b["x"])

    def test_in_bounds(self):
        idx = np.arange(8192)
        a = instance_attribs(idx, 777, _VOL.min_corner, _VOL.max_corner)
        assert (a["x"] >= _VOL.min_corner[0]).all()
        assert (a["x"] <= _VOL.max_corner[0]).all()
        assert (a["y"] >= _VOL.min_corner[1]).all()
        assert (a["y"] <= _VOL.max_corner[1]).all()
        assert (a["scale"] >= 0.7 - 1e-5).all()
        assert (a["scale"] <= 1.3 + 1e-5).all()
        assert (a["rot"] >= 0.0).all() and (a["rot"] < 2.0 * np.pi + 1e-4).all()

    def test_well_distributed(self):
        # 4096 blades over a 4×4 occupancy grid: every cell gets some.
        idx = np.arange(4096)
        a = instance_attribs(idx, 42, _VOL.min_corner, _VOL.max_corner)
        gx = np.clip(((a["x"] - _VOL.min_corner[0]) / (_VOL.size_m[0] / 4)).astype(int), 0, 3)
        gy = np.clip(((a["y"] - _VOL.min_corner[1]) / (_VOL.size_m[1] / 4)).astype(int), 0, 3)
        counts = np.bincount(gx * 4 + gy, minlength=16)
        assert (counts > 4096 / 16 * 0.5).all(), counts

    def test_hash_seed_deterministic_per_volume(self):
        set_world_seed(1337)
        s1 = grass_hash_seed(_VOL)
        set_world_seed(1337)
        s2 = grass_hash_seed(_VOL)
        assert s1 == s2
        assert 0 <= s1 < 2**31
        other = ZoneVolume(2, "grass", _VOL.min_corner, _VOL.max_corner)
        assert grass_hash_seed(other) != s1

    def test_instance_count(self):
        cfg = Config()
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0), params={"density": 8.0})
        assert grass_instance_count(v, cfg) == 800
        # Falls back to config density without a param.
        v2 = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        assert grass_instance_count(v2, cfg) == int(100 * cfg.grass_density_per_m2)
        # Clamped by the hard cap.
        v3 = ZoneVolume(
            1, "grass", (-500.0, -500.0, 0.0), (500.0, 500.0, 4.0), params={"density": 1000.0}
        )
        assert grass_instance_count(v3, cfg) == cfg.grass_max_instances


# ---------------------------------------------------------------------------
# Height-field bake
# ---------------------------------------------------------------------------


def _flat_chunks(cfg, coords):
    """Dict chunk provider with baseline flat terrain (like test_brush.py)."""
    from fire_engine.world.terrain.generation import generate_chunk

    class _Chunk:
        def __init__(self, materials):
            self.materials = materials

    return {c: _Chunk(generate_chunk(c, cfg)) for c in coords}


def _spawn_chunk_coords():
    """Chunk coords covering the demo grass volume (x,y in chunk -1..1, z 0)."""
    return [(cx, cy, 0) for cx in (-1, 0) for cy in (-1, 0, 1)]


class TestHeightFieldBake:
    def setup_method(self):
        set_world_seed(1337)
        self.cfg = load_config()
        self.vol = ZoneVolume(1, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0))
        self.chunks = _flat_chunks(self.cfg, _spawn_chunk_coords())

    def test_shape_and_dtype(self):
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        # 24 m × 30 m at 0.5 m texels → 48 columns (x), 60 rows (y).
        assert field.shape == (60, 48, 4)
        assert field.dtype == np.uint8

    def test_flat_ground_encodes_surface(self):
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        # Flat ground top at z=8 inside window [6, 10] → (8-6)/4*254 = 127.
        assert (field[..., 0] == 127).all()

    def test_carved_column_gets_sentinel(self):
        # Empty a full voxel column under one corner of the volume.
        chunk = self.chunks[(0, 0, 0)]
        chunk.materials[2, 3, :] = 0  # world x [1.0,1.5), y [1.5,2.0)
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        # Texel for world (1.25, 1.75): ix = (1.25-(-12))/0.5 = 26 (x → col),
        # iy = (1.75-(-5))/0.5 = 13 (y → row).
        assert field[13, 26, 0] == HEIGHT_SENTINEL
        # A neighbour column is untouched.
        assert field[13, 28, 0] == 127

    def test_unloaded_chunks_are_sentinel(self):
        field = bake_grass_height_field(self.vol, {}, self.cfg)
        assert (field[..., 0] == HEIGHT_SENTINEL).all()

    def test_deterministic(self):
        a = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        b = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        assert np.array_equal(a, b)

    def test_surface_above_window_is_sentinel(self):
        # Raise the terrain above the volume's z-window in one column:
        # filling the column to the chunk top (z=16) puts the surface at 16,
        # outside [6, 10] → sentinel (no grass floating mid-cliff).
        chunk = self.chunks[(0, 0, 0)]
        chunk.materials[4, 4, :] = 1  # solid to z=16
        field = bake_grass_height_field(self.vol, self.chunks, self.cfg)
        # World x [2.0,2.5), y [2.0,2.5) → ix=(2.25+12)/0.5=28, iy=(2.25+5)/0.5=14
        assert field[14, 28, 0] == HEIGHT_SENTINEL
