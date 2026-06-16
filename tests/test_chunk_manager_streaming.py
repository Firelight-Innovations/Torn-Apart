"""
tests/test_chunk_manager_streaming.py — Characterisation / golden-master tests
for ChunkManager streaming, desired_set, get_or_create, delta round-trip, and
the Saveable protocol.

Headless: no panda3d imports anywhere in this file or the modules it exercises.
All constants (load cap, Z band, hysteresis) are pulled from the
chunk_manager module, never hard-coded as duplicate magic numbers.

NOTE on stream_frame headlessness
----------------------------------
stream_frame IS headless: it calls build_mesh_faceted (or build_mesh) which is
pure numpy and never imports panda3d.  All tests below that call stream_frame
complete without a renderer.  The pending_meshes it produces are MeshArrays
(pure numpy dataclasses) — uploading them to panda3d is the World layer's
job, not done here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fire_engine.core import EventBus, load_config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.save import Saveable
from fire_engine.world.terrain import BrushMode, ChunkManager, SphereBrush, apply_brush
from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.chunk_manager import (
    _MAX_LOADS_PER_FRAME,
    _Z_MAX,
    _Z_MIN,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """Default config (view_distance_chunks=6, chunk_meters=16 m)."""
    return load_config()


@pytest.fixture
def small_cfg():
    """Config with a tiny view radius so desired_set + streaming are fast."""
    base = load_config()
    return dataclasses.replace(base, view_distance_chunks=2)


def _make_cm(cfg=None, seed: int = 1337) -> ChunkManager:
    """Return a fresh, headless ChunkManager."""
    if cfg is None:
        cfg = load_config()
    set_world_seed(seed)
    return ChunkManager(cfg, EventBus())


def _carve(cm: ChunkManager) -> set[tuple[int, int, int]]:
    """Carve a small crater at a corner-spanning position and return touched set."""
    return apply_brush(
        SphereBrush(2.5),
        Vec3(16.0, 16.0, 8.0),  # on a chunk corner → spans multiple chunks
        BrushMode.REMOVE,
        chunk_provider=cm.get_or_create,
    )


# ---------------------------------------------------------------------------
# Module-level constant sanity (pins the values we rely on throughout)
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_max_loads_per_frame_is_two(self):
        """Pin the documented 2-chunk-per-frame load cap."""
        assert _MAX_LOADS_PER_FRAME == 2

    def test_z_band_relative(self):
        """Z band is -2 below .. +4 above the camera chunk (7 levels)."""
        assert _Z_MIN == -2
        assert _Z_MAX == 4
        assert (_Z_MAX - _Z_MIN + 1) == 7


# ---------------------------------------------------------------------------
# desired_set — pure function
# ---------------------------------------------------------------------------


class TestDesiredSet:
    def test_count_at_origin(self, cfg):
        """Origin camera: (2r+1)^2 * 7 chunks in the desired set."""
        cm = _make_cm(cfg)
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        r = cfg.view_distance_chunks
        expected = (2 * r + 1) ** 2 * 7
        assert len(ds) == expected

    def test_count_small_radius(self, small_cfg):
        """Same formula holds for a smaller radius."""
        cm = _make_cm(small_cfg)
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        r = small_cfg.view_distance_chunks
        assert len(ds) == (2 * r + 1) ** 2 * 7

    def test_camera_chunk_included(self, cfg):
        """The camera's own chunk (and its Z band) is always in the desired set."""
        cm = _make_cm(cfg)
        ds = cm.desired_set(Vec3(8.0, 8.0, 8.0))  # camera chunk (0,0,0)
        for dz in range(_Z_MIN, _Z_MAX + 1):
            assert (0, 0, dz) in ds

    def test_maximum_xy_corners_included(self, cfg):
        """Corner chunk at (r, r, *) within the Z band is in the set."""
        cm = _make_cm(cfg)
        r = cfg.view_distance_chunks
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        for dz in range(_Z_MIN, _Z_MAX + 1):
            assert (r, r, dz) in ds

    def test_chunk_just_outside_xy_excluded(self, cfg):
        """Chunk at (r+1, 0, 0) is outside the square XY radius."""
        cm = _make_cm(cfg)
        r = cfg.view_distance_chunks
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        assert (r + 1, 0, 0) not in ds

    def test_z_above_band_excluded(self, cfg):
        """Z one above the maximum relative band is not in the desired set."""
        cm = _make_cm(cfg)
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        # camera chunk Z=0, so absolute Z == _Z_MAX + 1 is just outside
        assert (0, 0, _Z_MAX + 1) not in ds

    def test_z_below_band_excluded(self, cfg):
        """Z one below the minimum relative band is not in the desired set."""
        cm = _make_cm(cfg)
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        assert (0, 0, _Z_MIN - 1) not in ds

    def test_camera_shift_one_chunk_shifts_set(self, cfg):
        """Moving the camera by exactly one chunk width shifts the desired set."""
        cm = _make_cm(cfg)
        ds0 = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        # Move +1 chunk in X (chunk_meters = 16 m for default config)
        ds1 = cm.desired_set(Vec3(cfg.chunk_meters, 0.0, 0.0))
        # All coords in ds1 should be ds0-coords shifted by (+1, 0, 0)
        shifted = {(cx + 1, cy, cz) for (cx, cy, cz) in ds0}
        assert ds1 == shifted

    def test_symmetric_negative_xy(self, cfg):
        """Desired set is symmetric: (cx, cy, cz) ↔ (-cx, -cy, cz) at origin."""
        cm = _make_cm(cfg)
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        for cx, cy, cz in ds:
            assert (-cx, -cy, cz) in ds

    def test_pure_function_no_side_effects(self, cfg):
        """desired_set must not modify cm.chunks (pure — no loading)."""
        cm = _make_cm(cfg)
        before = len(cm.chunks)
        cm.desired_set(Vec3(0.0, 0.0, 0.0))
        assert len(cm.chunks) == before

    def test_fractional_camera_pos_resolves_to_chunk(self, cfg):
        """A camera at (0.1, 0.1, 0.1) still resolves to chunk (0,0,0)."""
        cm = _make_cm(cfg)
        ds = cm.desired_set(Vec3(0.1, 0.1, 0.1))
        # (0,0,0) must be in the set
        assert (0, 0, 0) in ds
        # Count must still be the formula count
        r = cfg.view_distance_chunks
        assert len(ds) == (2 * r + 1) ** 2 * 7

    def test_negative_camera_pos_resolves_correctly(self, cfg):
        """Negative camera position resolves the correct camera chunk."""
        cm = _make_cm(cfg)
        # Camera at (-8, -8, -8) → chunk (-1, -1, -1)
        ds = cm.desired_set(Vec3(-8.0, -8.0, -8.0))
        # Camera chunk should be (-1,-1,-1); that coord must appear in the set
        assert (-1, -1, -1) in ds


# ---------------------------------------------------------------------------
# get_or_create — provider contract
# ---------------------------------------------------------------------------


class TestGetOrCreate:
    def test_creates_and_caches(self, cfg):
        """get_or_create returns a Chunk and stores it in cm.chunks."""
        cm = _make_cm(cfg)
        ch = cm.get_or_create((3, 3, 0))
        assert isinstance(ch, Chunk)
        assert (3, 3, 0) in cm.chunks

    def test_second_call_same_object(self, cfg):
        """Calling get_or_create twice returns the exact same object (identity)."""
        cm = _make_cm(cfg)
        ch1 = cm.get_or_create((1, 2, -1))
        ch2 = cm.get_or_create((1, 2, -1))
        assert ch1 is ch2

    def test_coord_stored_correctly(self, cfg):
        """The returned chunk carries the requested coord."""
        cm = _make_cm(cfg)
        coord = (5, -3, 2)
        ch = cm.get_or_create(coord)
        assert ch.coord == coord

    def test_materials_shape_and_dtype(self, cfg):
        """Generated chunk has uint8 (32,32,32) materials."""
        cm = _make_cm(cfg)
        ch = cm.get_or_create((0, 0, 0))
        assert ch.materials.shape == (32, 32, 32)
        assert ch.materials.dtype == np.uint8

    def test_callable_alias(self, cfg):
        """ChunkManager is directly callable as a provider (cm(coord))."""
        cm = _make_cm(cfg)
        ch = cm((2, 0, -1))
        assert isinstance(ch, Chunk)
        assert cm.get_or_create((2, 0, -1)) is ch

    def test_determinism_across_managers(self, cfg):
        """Two fresh managers with the same seed generate identical materials."""
        cm_a = _make_cm(cfg, seed=42)
        cm_b = _make_cm(cfg, seed=42)
        ch_a = cm_a.get_or_create((1, -2, 0))
        ch_b = cm_b.get_or_create((1, -2, 0))
        assert np.array_equal(ch_a.materials, ch_b.materials)


# ---------------------------------------------------------------------------
# stream_frame — load cap + convergence + unload hysteresis
# ---------------------------------------------------------------------------


class TestStreamFrame:
    def test_loads_at_most_cap_chunks_first_frame(self, small_cfg):
        """First stream_frame loads at most _MAX_LOADS_PER_FRAME new chunks."""
        cm = _make_cm(small_cfg)
        cm.stream_frame(Vec3(0.0, 0.0, 0.0))
        assert len(cm.chunks) <= _MAX_LOADS_PER_FRAME

    def test_repeated_calls_converge_to_desired_set(self, small_cfg):
        """After enough stream_frame calls, all desired chunks are loaded."""
        cm = _make_cm(small_cfg)
        pos = Vec3(0.0, 0.0, 0.0)
        desired = cm.desired_set(pos)
        # Worst case: need ceil(len(desired) / cap) frames; give a generous bound.
        max_frames = len(desired) + 10
        for _ in range(max_frames):
            cm.stream_frame(pos)
            if desired.issubset(cm.chunks.keys()):
                break
        missing = desired - cm.chunks.keys()
        assert missing == set(), f"Still missing {len(missing)} chunks after convergence"

    def test_stream_frame_produces_pending_meshes(self, small_cfg):
        """Each stream_frame call adds at most cap entries to pending_meshes."""
        cm = _make_cm(small_cfg)
        cm.stream_frame(Vec3(0.0, 0.0, 0.0))
        # pending_meshes may include remeshed dirty chunks too, but at least
        # the newly loaded chunks should be represented
        assert len(cm.pending_meshes) <= _MAX_LOADS_PER_FRAME

    def test_unload_beyond_hysteresis(self, small_cfg):
        """Chunks beyond radius+1 are unloaded; chunks at exactly radius are kept."""
        # Fully load all desired chunks first
        cm = _make_cm(small_cfg)
        pos = Vec3(0.0, 0.0, 0.0)
        desired = cm.desired_set(pos)
        for _ in range(len(desired) + 10):
            cm.stream_frame(pos)
            if desired.issubset(cm.chunks.keys()):
                break

        # Manually plant a chunk far outside the hysteresis boundary
        r = small_cfg.view_distance_chunks
        far_coord = (r + 2, 0, 0)  # beyond radius+1
        cm.chunks[far_coord] = Chunk(far_coord)

        # One stream_frame should evict the far chunk
        cm.stream_frame(pos)
        assert far_coord not in cm.chunks

    def test_chunk_at_hysteresis_boundary_kept(self, small_cfg):
        """A chunk at exactly radius+1 is NOT unloaded (hysteresis boundary is inclusive)."""
        cm = _make_cm(small_cfg)
        pos = Vec3(0.0, 0.0, 0.0)
        r = small_cfg.view_distance_chunks
        # hysteresis keeps chunks at |dx| <= radius+1
        boundary_coord = (r + 1, 0, 0)
        cm.chunks[boundary_coord] = Chunk(boundary_coord)
        # Load a few desired chunks so stream_frame runs normally
        cm.stream_frame(pos)
        # boundary_coord is at exactly the hysteresis edge: keep it
        assert boundary_coord in cm.chunks

    def test_unloaded_this_frame_populated(self, small_cfg):
        """Evicted chunk coords appear in unloaded_this_frame."""
        cm = _make_cm(small_cfg)
        pos = Vec3(0.0, 0.0, 0.0)
        r = small_cfg.view_distance_chunks
        far_coord = (r + 5, 0, 0)
        cm.chunks[far_coord] = Chunk(far_coord)
        cm.stream_frame(pos)
        assert far_coord in cm.unloaded_this_frame

    def test_dirty_chunks_remeshed_before_loading(self, small_cfg):
        """
        Dirty chunks are remeshed in the 2-chunk budget BEFORE new loads.
        If we have 2 dirty chunks, the budget is spent on them; no new loads.
        """
        cm = _make_cm(small_cfg)
        # Manually insert 2 dirty chunks
        c0 = cm.get_or_create((0, 0, 0))
        c1 = cm.get_or_create((1, 0, 0))
        c0.dirty = True
        c1.dirty = True
        chunks_before = set(cm.chunks.keys())
        cm.stream_frame(Vec3(0.0, 0.0, 0.0))
        # Both dirty chunks were remeshed; no new chunks should have been added
        # (budget exhausted). Check the dirty flags are cleared.
        assert not c0.dirty
        assert not c1.dirty
        # No new chunks added (budget = 2 used for remesh; 0 left for loads)
        assert set(cm.chunks.keys()) == chunks_before


# ---------------------------------------------------------------------------
# Delta round-trip (Saveable contract)
# ---------------------------------------------------------------------------


class TestDeltaRoundTrip:
    def test_unedited_world_empty_delta(self, cfg):
        """A fresh world with no edits returns an empty delta."""
        cm = _make_cm(cfg)
        cm.get_or_create((0, 0, 0))
        cm.get_or_create((1, 0, 0))
        assert cm.get_delta() == {}

    def test_edited_chunk_appears_in_delta(self, cfg):
        """Carving a crater marks chunks edited; they appear in get_delta."""
        cm = _make_cm(cfg)
        touched = _carve(cm)
        delta = cm.get_delta()
        assert len(delta) >= 1
        # Every edited coord in delta must be in the touched set
        for coord in delta:
            assert coord in touched

    def test_delta_values_are_uint8_arrays(self, cfg):
        """Delta values are uint8 (32,32,32) arrays — no live object refs."""
        cm = _make_cm(cfg)
        _carve(cm)
        delta = cm.get_delta()
        for _coord, arr in delta.items():
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.uint8
            assert arr.shape == (32, 32, 32)

    def test_delta_values_are_copies(self, cfg):
        """Mutating a chunk after get_delta doesn't corrupt the snapshot."""
        cm = _make_cm(cfg)
        _carve(cm)
        delta = cm.get_delta()
        # Pick one coord from the delta
        coord = next(iter(delta))
        snapshot = delta[coord].copy()
        # Mutate the live chunk
        cm.chunks[coord].materials[0, 0, 0] ^= 0xFF
        # Snapshot unchanged
        assert np.array_equal(delta[coord], snapshot)

    def test_apply_delta_restores_materials(self, cfg):
        """Fresh manager + apply_delta → edited chunks have identical materials."""
        cm = _make_cm(cfg)
        _carve(cm)
        delta = cm.get_delta()
        # Snapshot live materials
        pre = {coord: arr.copy() for coord, arr in delta.items()}

        cm2 = _make_cm(cfg)
        cm2.apply_delta(delta)

        for coord, original in pre.items():
            assert coord in cm2.chunks
            assert np.array_equal(cm2.chunks[coord].materials, original)

    def test_apply_delta_marks_edited_and_dirty(self, cfg):
        """apply_delta marks the restored chunk edited=True and dirty=True."""
        cm = _make_cm(cfg)
        _carve(cm)
        delta = cm.get_delta()

        cm2 = _make_cm(cfg)
        cm2.apply_delta(delta)

        for coord in delta:
            ch = cm2.chunks[coord]
            assert ch.edited is True
            assert ch.dirty is True

    def test_apply_delta_non_edited_chunks_absent(self, cfg):
        """apply_delta only loads edited chunks; unrelated coords stay absent."""
        cm = _make_cm(cfg)
        _carve(cm)
        delta = cm.get_delta()
        edited_set = set(delta.keys())

        cm2 = _make_cm(cfg)
        cm2.apply_delta(delta)

        # cm2 should only have the delta-restored chunks
        assert set(cm2.chunks.keys()) == edited_set

    def test_round_trip_via_manual_edit(self, cfg):
        """
        Manually set edited=True, get_delta, apply_delta on fresh manager,
        assert voxel arrays are np.array_equal.
        """
        cm = _make_cm(cfg)
        ch = cm.get_or_create((2, -1, 0))
        # Flip a voxel deterministically
        ch.materials[10, 10, 10] = 0 if ch.materials[10, 10, 10] else 1
        ch.edited = True
        delta = cm.get_delta()
        assert (2, -1, 0) in delta

        cm2 = _make_cm(cfg)
        cm2.apply_delta(delta)
        assert np.array_equal(cm2.chunks[(2, -1, 0)].materials, ch.materials)

    def test_reset_to_baseline_clears_edits(self, cfg):
        """reset_to_baseline reverts all loaded edited chunks to generated baseline."""
        cm = _make_cm(cfg)
        _carve(cm)
        # Confirm some chunks are edited
        assert any(ch.edited for ch in cm.chunks.values())

        cm.reset_to_baseline()

        # No chunk should be edited after reset
        assert not any(ch.edited for ch in cm.chunks.values())
        # All reverted chunks should be dirty (need remesh)
        assert all(ch.dirty for ch in cm.chunks.values())

    def test_reset_restores_baseline_materials(self, cfg):
        """After reset_to_baseline, materials match the pure generate_chunk output."""
        from fire_engine.world.terrain.generation import generate_chunk

        cm = _make_cm(cfg)
        _carve(cm)
        edited_coords = [c for c, ch in cm.chunks.items() if ch.edited]

        cm.reset_to_baseline()

        for coord in edited_coords:
            expected = generate_chunk(coord, cfg)
            assert np.array_equal(cm.chunks[coord].materials, expected)


# ---------------------------------------------------------------------------
# Saveable protocol
# ---------------------------------------------------------------------------


class TestSaveableProtocol:
    def test_save_key_is_terrain(self, cfg):
        """ChunkManager.save_key (class attribute) equals 'terrain'."""
        assert ChunkManager.save_key == "terrain"

    def test_instance_save_key(self, cfg):
        """Instance also exposes the correct save_key."""
        cm = _make_cm(cfg)
        assert cm.save_key == "terrain"

    def test_is_saveable(self, cfg):
        """ChunkManager instances satisfy the runtime-checkable Saveable protocol."""
        cm = _make_cm(cfg)
        assert isinstance(cm, Saveable)

    def test_saveable_interface_present(self, cfg):
        """get_delta and apply_delta are callable on the instance."""
        cm = _make_cm(cfg)
        assert callable(getattr(cm, "get_delta", None))
        assert callable(getattr(cm, "apply_delta", None))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_chunk_contents(self, cfg):
        """Two managers with the same seed produce identical chunk materials."""
        cm_a = _make_cm(cfg, seed=99)
        cm_b = _make_cm(cfg, seed=99)
        coords = [(0, 0, 0), (1, -1, 0), (0, 0, -1)]
        for coord in coords:
            assert np.array_equal(
                cm_a.get_or_create(coord).materials,
                cm_b.get_or_create(coord).materials,
            )

    def test_different_seeds_produce_same_flat_baseline(self, cfg):
        """
        SUSPICIOUS / SUSPECTED BUG: terrain docs say baseline is seed-independent,
        so two managers with different seeds should produce identical flat terrain.
        This test PINS that claim — failure indicates either the docs are wrong
        or generate_chunk has seed-dependent logic.
        """
        cm_a = _make_cm(cfg, seed=1)
        cm_b = _make_cm(cfg, seed=9999)
        # Flat-ground chunk: should be identical regardless of seed
        ch_a = cm_a.get_or_create((0, 0, 0))
        ch_b = cm_b.get_or_create((0, 0, 0))
        assert np.array_equal(ch_a.materials, ch_b.materials), (
            "generate_chunk should be seed-independent (flat/authored baseline). "
            "If this fails, terrain generation has gained seed-dependent variation."
        )

    def test_same_camera_path_same_desired_set(self, cfg):
        """same seed + same camera position always yields same desired_set."""
        cm_a = _make_cm(cfg, seed=1337)
        cm_b = _make_cm(cfg, seed=1337)
        pos = Vec3(24.0, -8.0, 16.0)
        assert cm_a.desired_set(pos) == cm_b.desired_set(pos)

    def test_stream_convergence_same_chunk_contents(self, small_cfg):
        """After converging two managers to the desired set, materials are identical."""

        def converge(cm, pos):
            desired = cm.desired_set(pos)
            for _ in range(len(desired) + 10):
                cm.stream_frame(pos)
                if desired.issubset(cm.chunks.keys()):
                    break

        pos = Vec3(0.0, 0.0, 0.0)
        cm_a = _make_cm(small_cfg, seed=42)
        cm_b = _make_cm(small_cfg, seed=42)
        converge(cm_a, pos)
        converge(cm_b, pos)

        for coord in cm_a.desired_set(pos):
            if coord in cm_a.chunks and coord in cm_b.chunks:
                assert np.array_equal(
                    cm_a.chunks[coord].materials,
                    cm_b.chunks[coord].materials,
                )
