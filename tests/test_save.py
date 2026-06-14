"""
tests/test_save.py — SaveManager round-trip, compatibility checks, size bounds,
no-pickle enforcement, and numpy/tuple-key encoding.  Headless: no panda3d.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.save import Saveable, SaveIncompatibleError, SaveManager
from fire_engine.save.save_manager import (
    _decode_delta,
    _decode_value,
    _encode_delta,
    _encode_value,
)
from fire_engine.world.terrain import ChunkManager
from fire_engine.world.terrain.brush import BrushMode, SphereBrush, apply_brush

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_world(seed: int = 1337):
    """Return (cfg, clock, bus, cm, sm) for a headless world at ``seed``."""
    cfg = load_config()
    # Build a config with the given seed via a simple replacement
    from dataclasses import replace

    cfg = replace(cfg, world_seed=seed)
    set_world_seed(seed)
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    cm = ChunkManager(cfg, bus)
    sm = SaveManager(cfg, clock)
    sm.register(cm)
    return cfg, clock, bus, cm, sm


def _blast_craters(cm: ChunkManager) -> list[tuple[int, int, int]]:
    """
    Carve 3 craters at fixed positions so chunks become ``edited``.

    Returns
    -------
    list of chunk coords that became edited.
    """
    centers = [
        Vec3(8.0, 8.0, 12.0),
        Vec3(24.0, 8.0, 12.0),
        Vec3(8.0, 24.0, 10.0),
    ]
    edited_coords: set = set()
    for center in centers:
        changed = apply_brush(
            SphereBrush(3.0),
            center,
            BrushMode.REMOVE,
            chunk_provider=cm,
        )
        edited_coords.update(changed)
    return list(edited_coords)


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Save → fresh world same seed → load → voxel arrays identical."""

    def test_craters_survive_round_trip(self, tmp_path):
        """
        Blast 3 craters into headless terrain, save, create a fresh world with
        the same seed, load, and assert edited-chunk voxel arrays are identical.
        """
        save_file = tmp_path / "test.ta"

        # --- Build world and blast craters ---
        cfg, clock, bus, cm, sm = _make_world(seed=1337)
        edited_coords = _blast_craters(cm)
        assert edited_coords, "Expected at least one edited chunk"

        # Snapshot voxel arrays before saving
        pre_save = {coord: cm.chunks[coord].materials.copy() for coord in edited_coords}

        # Advance the clock a little so we verify clock restore too
        clock.update(5.0)
        saved_game_day = clock.game_day
        saved_game_time = clock.game_time_of_day

        sm.save(save_file)
        assert save_file.exists()

        # --- Fresh world, same seed ---
        cfg2, clock2, bus2, cm2, sm2 = _make_world(seed=1337)
        sm2.load(save_file)

        # Clock must be restored
        assert clock2.game_day == saved_game_day
        assert abs(clock2.game_time_of_day - saved_game_time) < 1e-6

        # Edited chunks must have identical voxel arrays
        for coord in edited_coords:
            assert coord in cm2.chunks, f"Chunk {coord} should be in cm2.chunks after apply_delta"
            post_load = cm2.chunks[coord].materials
            assert np.array_equal(pre_save[coord], post_load), (
                f"Voxel array mismatch for chunk {coord} after round-trip"
            )
            assert cm2.chunks[coord].edited, (
                f"Chunk {coord} should be marked edited after apply_delta"
            )

    def test_unedited_chunks_not_in_delta(self, tmp_path):
        """An unedited world has an empty terrain delta."""
        save_file = tmp_path / "empty.ta"
        _, _, _, cm, sm = _make_world(seed=42)
        sm.save(save_file)
        delta = cm.get_delta()
        assert delta == {}, "Unedited world should have empty terrain delta"


# ---------------------------------------------------------------------------
# Wrong-seed incompatibility
# ---------------------------------------------------------------------------


class TestWrongSeedRaisesIncompatible:
    def test_wrong_seed_raises(self, tmp_path):
        """
        Save with seed 1337, attempt load against a Config with seed 9999.
        Must raise SaveIncompatibleError without modifying state.
        """
        save_file = tmp_path / "seed_mismatch.ta"

        # Save with seed 1337
        _, _, _, _, sm1 = _make_world(seed=1337)
        sm1.save(save_file)

        # Prepare a fresh world with seed 9999
        _, clock2, _, cm2, sm2 = _make_world(seed=9999)
        original_clock_state = clock2.get_state()

        with pytest.raises(SaveIncompatibleError, match="world_seed"):
            sm2.load(save_file)

        # State must be unchanged (no partial load)
        assert clock2.get_state() == original_clock_state


# ---------------------------------------------------------------------------
# Unedited world is tiny
# ---------------------------------------------------------------------------


class TestUneditedWorldIsTiny:
    def test_file_under_1kb(self, tmp_path):
        """
        A world with zero edited chunks should produce a save file whose terrain
        blob (zlib-compressed delta) is under 1 KB.

        We check the total file size as well — it should be well under 1 KB
        plus a small header overhead.
        """
        import msgpack as _msgpack

        save_file = tmp_path / "tiny.ta"
        _, _, _, cm, sm = _make_world(seed=1)
        sm.save(save_file)

        # Verify the terrain delta blob is < 1 KB
        raw = save_file.read_bytes()
        envelope = _msgpack.unpackb(raw, raw=False)
        systems = envelope.get("systems") or envelope.get(b"systems") or {}
        if isinstance(systems, dict):
            terrain_blob = systems.get("terrain") or systems.get(b"terrain")
        else:
            terrain_blob = None

        assert terrain_blob is not None, "Expected a 'terrain' system blob"
        assert len(terrain_blob) < 1024, (
            f"Unedited terrain blob is {len(terrain_blob)} bytes, expected < 1 KB"
        )


# ---------------------------------------------------------------------------
# No-pickle test
# ---------------------------------------------------------------------------


class TestNoPickle:
    """Walk fire_engine/ and tools/ .py files; fail if any contain pickle imports."""

    # Match any form of pickle import:
    #   import pickle
    #   import cPickle
    #   from pickle import ...
    #   from cPickle import ...
    #   pickle.loads(...)  (usage without explicit import — also banned)
    _PICKLE_PATTERN = re.compile(
        r"(import\s+(c?pickle)"
        r"|from\s+(c?pickle)\s+import"
        r"|pickle\.\w+)",
        re.MULTILINE,
    )

    def _collect_py_files(self) -> list[Path]:
        repo_root = Path(__file__).resolve().parent.parent
        dirs = [repo_root / "fire_engine", repo_root / "tools"]
        files = []
        for d in dirs:
            if d.exists():
                files.extend(d.rglob("*.py"))
        return files

    def test_no_pickle_imports(self):
        """None of the source files in fire_engine/ or tools/ import pickle."""
        violations: list[str] = []
        for py_file in self._collect_py_files():
            source = py_file.read_text(encoding="utf-8", errors="replace")
            for match in self._PICKLE_PATTERN.finditer(source):
                lineno = source[: match.start()].count("\n") + 1
                violations.append(f"{py_file}:{lineno}: {match.group()!r}")

        if violations:
            joined = "\n  ".join(violations)
            pytest.fail(
                f"Found forbidden pickle usage in {len(violations)} location(s):\n"
                f"  {joined}\n"
                "Hard Rule 3: no pickle anywhere — saves are seed + msgpack deltas."
            )


# ---------------------------------------------------------------------------
# Numpy + tuple-key encoding
# ---------------------------------------------------------------------------


class TestNumpyEncoding:
    """Encoder/decoder round-trips for numpy arrays and tuple keys."""

    def test_uint8_array_round_trip(self):
        """uint8 [32,32,32] terrain array survives encode→decode intact."""
        arr = np.arange(32 * 32 * 32, dtype=np.uint8).reshape(32, 32, 32)
        encoded = _encode_value(arr)
        decoded = _decode_value(encoded)
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.uint8
        assert decoded.shape == (32, 32, 32)
        assert np.array_equal(arr, decoded)

    def test_float32_array_round_trip(self):
        """float32 array round-trips with correct dtype and shape."""
        arr = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        encoded = _encode_value(arr)
        decoded = _decode_value(encoded)
        assert decoded.dtype == np.float32
        assert decoded.shape == (10, 10)
        assert np.array_equal(arr, decoded)

    def test_tuple_key_dict_round_trip(self):
        """Dict with tuple-of-int keys round-trips through encode→decode."""
        arr = np.zeros((32, 32, 32), dtype=np.uint8)
        original = {
            (0, 0, 0): arr.copy(),
            (1, -2, 3): arr + 1,
        }
        encoded = _encode_value(original)
        decoded = _decode_value(encoded)

        assert isinstance(decoded, dict)
        assert set(decoded.keys()) == {(0, 0, 0), (1, -2, 3)}
        for k in original:
            assert np.array_equal(original[k], decoded[k])

    def test_full_encode_decode_round_trip(self):
        """Full msgpack encode→decode round-trip for terrain-like delta."""
        arr1 = np.ones((32, 32, 32), dtype=np.uint8) * 5
        arr2 = np.zeros((32, 32, 32), dtype=np.uint8)
        delta = {
            (0, 0, 1): arr1,
            (2, 3, -1): arr2,
        }
        raw = _encode_delta(delta)
        recovered = _decode_delta(raw)

        assert set(recovered.keys()) == {(0, 0, 1), (2, 3, -1)}
        assert np.array_equal(recovered[(0, 0, 1)], arr1)
        assert np.array_equal(recovered[(2, 3, -1)], arr2)

    def test_empty_delta_round_trip(self):
        """Empty dict encodes and decodes to empty dict."""
        raw = _encode_delta({})
        recovered = _decode_delta(raw)
        assert recovered == {}

    def test_string_key_dict_not_wrapped(self):
        """Dict with only string keys encodes directly (no kv_pairs wrapper)."""
        delta = {"health": 42, "name": "test"}
        encoded = _encode_value(delta)
        # Should be a plain dict (no __delta_type__ wrapper)
        assert isinstance(encoded, dict)
        assert "__delta_type__" not in encoded
        decoded = _decode_value(encoded)
        assert decoded == delta


# ---------------------------------------------------------------------------
# Saveable protocol
# ---------------------------------------------------------------------------


class TestSaveableProtocol:
    def test_chunk_manager_is_saveable(self):
        """ChunkManager satisfies the Saveable protocol."""
        cfg = load_config()
        set_world_seed(1337)
        bus = EventBus()
        cm = ChunkManager(cfg, bus)
        assert isinstance(cm, Saveable)

    def test_non_saveable_raises(self):
        """Registering a non-Saveable raises TypeError."""
        cfg = load_config()
        set_world_seed(1337)
        bus = EventBus()
        clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
        sm = SaveManager(cfg, clock)

        class NotSaveable:
            pass

        with pytest.raises(TypeError):
            sm.register(NotSaveable())

    def test_save_incompatible_error_is_exception(self):
        """SaveIncompatibleError is a subclass of Exception."""
        exc = SaveIncompatibleError("test message")
        assert isinstance(exc, Exception)
        assert "test message" in str(exc)
