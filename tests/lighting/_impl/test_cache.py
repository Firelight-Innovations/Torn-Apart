"""
tests/lighting/_impl/test_cache.py — Headless tests for
fire_engine.lighting._impl.cache.ChunkBlockCache.

Covers:
- get/put round-trip.
- LRU eviction at max_entries.
- invalidate removes all cell sizes for that coord.
- clear empties the cache.
- Thread-safety: concurrent puts do not corrupt length.
- Stored arrays are read-only (WRITEABLE flag cleared).

No panda3d imports.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from fire_engine.lighting._impl.cache import ChunkBlockCache


def _block(val: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Minimal 2x2x2 mini-block for cache tests."""
    mat = np.full((2, 2, 2), val, dtype=np.uint8)
    cnt = np.full((2, 2, 2), val, dtype=np.uint16)
    return mat, cnt


class TestGetPutRoundTrip:
    def test_miss_returns_none(self):
        cache = ChunkBlockCache()
        assert cache.get((0, 0, 0), 0.5) is None

    def test_put_then_get_returns_block(self):
        cache = ChunkBlockCache()
        blk = _block(5)
        cache.put((0, 0, 0), 0.5, blk)
        result = cache.get((0, 0, 0), 0.5)
        assert result is not None
        np.testing.assert_array_equal(result[0], blk[0])
        np.testing.assert_array_equal(result[1], blk[1])

    def test_different_cell_m_is_separate_key(self):
        cache = ChunkBlockCache()
        blk1 = _block(1)
        blk2 = _block(2)
        cache.put((0, 0, 0), 0.5, blk1)
        cache.put((0, 0, 0), 2.0, blk2)
        r1 = cache.get((0, 0, 0), 0.5)
        r2 = cache.get((0, 0, 0), 2.0)
        assert r1 is not None and r2 is not None
        assert int(r1[0][0, 0, 0]) == 1
        assert int(r2[0][0, 0, 0]) == 2

    def test_stored_arrays_are_readonly(self):
        cache = ChunkBlockCache()
        cache.put((0, 0, 0), 0.5, _block())
        mat, _cnt = cache.get((0, 0, 0), 0.5)
        with pytest.raises((ValueError, TypeError)):
            mat[0, 0, 0] = 99


class TestLength:
    def test_len_zero_initially(self):
        assert len(ChunkBlockCache()) == 0

    def test_len_increments_on_put(self):
        cache = ChunkBlockCache()
        cache.put((0, 0, 0), 0.5, _block())
        assert len(cache) == 1
        cache.put((1, 0, 0), 0.5, _block())
        assert len(cache) == 2

    def test_duplicate_put_does_not_grow(self):
        cache = ChunkBlockCache()
        cache.put((0, 0, 0), 0.5, _block(1))
        cache.put((0, 0, 0), 0.5, _block(2))
        assert len(cache) == 1


class TestLruEviction:
    def test_max_entries_not_exceeded(self):
        cache = ChunkBlockCache(max_entries=3)
        for i in range(6):
            cache.put((i, 0, 0), 0.5, _block(i))
        assert len(cache) <= 3

    def test_lru_entry_evicted(self):
        cache = ChunkBlockCache(max_entries=2)
        cache.put((0, 0, 0), 0.5, _block(0))
        cache.put((1, 0, 0), 0.5, _block(1))
        # Access (0,0,0) to make it MRU.
        cache.get((0, 0, 0), 0.5)
        # Adding a third entry should evict LRU = (1,0,0).
        cache.put((2, 0, 0), 0.5, _block(2))
        assert cache.get((1, 0, 0), 0.5) is None
        assert cache.get((0, 0, 0), 0.5) is not None


class TestInvalidate:
    def test_invalidate_removes_all_cell_sizes(self):
        cache = ChunkBlockCache()
        cache.put((0, 0, 0), 0.5, _block())
        cache.put((0, 0, 0), 2.0, _block())
        assert len(cache) == 2
        cache.invalidate((0, 0, 0))
        assert len(cache) == 0
        assert cache.get((0, 0, 0), 0.5) is None
        assert cache.get((0, 0, 0), 2.0) is None

    def test_invalidate_nonexistent_no_error(self):
        cache = ChunkBlockCache()
        cache.invalidate((99, 99, 99))  # must not raise

    def test_invalidate_leaves_other_coords(self):
        cache = ChunkBlockCache()
        cache.put((0, 0, 0), 0.5, _block(1))
        cache.put((1, 0, 0), 0.5, _block(2))
        cache.invalidate((0, 0, 0))
        assert cache.get((1, 0, 0), 0.5) is not None


class TestClear:
    def test_clear_empties_cache(self):
        cache = ChunkBlockCache()
        for i in range(5):
            cache.put((i, 0, 0), 0.5, _block(i))
        cache.clear()
        assert len(cache) == 0

    def test_get_after_clear_returns_none(self):
        cache = ChunkBlockCache()
        cache.put((0, 0, 0), 0.5, _block())
        cache.clear()
        assert cache.get((0, 0, 0), 0.5) is None


class TestThreadSafety:
    def test_concurrent_puts_do_not_corrupt_length(self):
        """Multiple threads putting entries concurrently must not exceed max_entries."""
        cache = ChunkBlockCache(max_entries=50)
        errors: list[Exception] = []

        def _worker(start: int) -> None:
            try:
                for i in range(start, start + 20):
                    cache.put((i, 0, 0), 0.5, _block(i % 255))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i * 20,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(cache) <= 50
