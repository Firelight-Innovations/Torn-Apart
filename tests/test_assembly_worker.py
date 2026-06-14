"""
tests/test_assembly_worker.py — Headless tests for off-thread cascade assembly.

Covers the panda3d-free worker that moves the GPU-lighting volume gather off the
main thread (``lighting/assembly_worker.py``) plus its supporting helpers in
``lighting/volume.py`` (``window_chunk_span``, ``pack_volume``).  The GPU
consumer (``lighting/gpu.py``) imports panda3d and is intentionally not imported
here (headless suite rule).

The key guarantee: a worker-produced result is byte-identical to a synchronous
``assemble_geometry`` + ``pack_volume`` for the same inputs — so threading
changes performance, never output.
"""

from __future__ import annotations

import time

import numpy as np

from fire_engine.lighting.assembly_worker import (
    AssemblyJob,
    CascadeAssemblyWorker,
    assemble_packed,
)
from fire_engine.lighting.palette import MaterialPalette
from fire_engine.lighting.volume import (
    VolumeWindow,
    assemble_geometry,
    pack_volume,
    window_chunk_span,
)

VOXEL = 0.5
CHUNK = 32


class _Chunk:
    """Minimal chunk stand-in: just a materials array."""

    def __init__(self, fill: int = 0) -> None:
        self.materials = np.full((CHUNK, CHUNK, CHUNK), fill, dtype=np.uint8)


def _palette() -> MaterialPalette:
    albedo = np.zeros((256, 3), dtype=np.float32)
    albedo[1] = (0.4, 0.3, 0.2)
    albedo[2] = (0.2, 0.5, 0.1)
    emission = np.zeros((256, 3), dtype=np.float32)
    emission[2] = (1.0, 0.5, 0.25)
    return MaterialPalette(albedo=albedo, emission=emission)


def _world() -> dict:
    """A few loaded chunks with some solid voxels around the origin."""
    chunks: dict = {}
    for cc in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, -1)]:
        ch = _Chunk()
        ch.materials[:, :, :16] = 1  # bottom half solid dirt
        ch.materials[:, :, 15] = 2  # grass skin layer
        chunks[cc] = ch
    return chunks


def _drain_until(worker, want: int, timeout_s: float = 5.0):
    """
    Poll the worker until ``want`` results arrive (or timeout).

    A tight ``drain_results`` spin would starve the worker of the GIL, so we
    yield with a short sleep each iteration.  (Production drains once per frame,
    where panda3d/GPU calls release the GIL constantly — no starvation.)
    """
    out = []
    deadline = time.monotonic() + timeout_s
    while len(out) < want and time.monotonic() < deadline:
        out += worker.drain_results()
        if len(out) < want:
            time.sleep(0.001)
    return out


def _job(origin_cell, cells, cell_m, materials, palette, seq=1) -> AssemblyJob:
    return AssemblyJob(
        cascade_index=0,
        origin_cell=tuple(origin_cell),
        cells=cells,
        cell_m=cell_m,
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        materials=materials,
        palette=palette,
        seq=seq,
    )


# ---------------------------------------------------------------------------
# window_chunk_span
# ---------------------------------------------------------------------------


class TestWindowChunkSpan:
    def test_covers_every_intersecting_chunk(self):
        # A 96-cell @ 0.5 m window (48 m) placed at origin spans chunks the
        # assembler will actually read; assert it covers a known solid chunk.
        win = VolumeWindow(cells=96, cell_m=0.5)
        win.recenter((8.0, 8.0, 8.0))
        coords = window_chunk_span(win.origin_cell, 96, 0.5, CHUNK, VOXEL)
        assert (0, 0, 0) in coords
        # No duplicates; it's a dense box range.
        assert len(coords) == len(set(coords))

    def test_span_includes_solid_chunks_in_box(self):
        # The known solid chunks near origin are within the 48 m box.
        win = VolumeWindow(cells=96, cell_m=0.5)
        win.recenter((8.0, 8.0, 8.0))
        span = set(window_chunk_span(win.origin_cell, 96, 0.5, CHUNK, VOXEL))
        assert {(0, 0, 0), (1, 0, 0), (0, 1, 0)} <= span


# ---------------------------------------------------------------------------
# pack_volume
# ---------------------------------------------------------------------------


class TestPackVolume:
    def test_pack_is_deterministic(self):
        arr = np.random.default_rng(0).integers(0, 256, size=(8, 8, 8, 4), dtype=np.uint8)
        assert pack_volume(arr) == pack_volume(arr)

    def test_pack_length_matches_block(self):
        arr = np.zeros((8, 8, 8, 4), dtype=np.uint8)
        assert len(pack_volume(arr)) == 8 * 8 * 8 * 4


# ---------------------------------------------------------------------------
# assemble_packed equivalence (the core guarantee)
# ---------------------------------------------------------------------------


class TestAssemblePacked:
    def test_matches_synchronous_assemble(self):
        chunks = _world()
        palette = _palette()
        win = VolumeWindow(cells=96, cell_m=0.5)
        win.recenter((8.0, 8.0, 8.0))
        materials = {c: ch.materials for c, ch in chunks.items()}

        # Synchronous reference: assemble_geometry on chunk objects + pack.
        vol = assemble_geometry(win, chunks, palette, chunk_size=CHUNK, voxel_size=VOXEL)
        ref_albedo = pack_volume(vol.albedo_occ)
        ref_emis = pack_volume(vol.emission)

        # Worker path: assemble_packed on the bare-array snapshot.
        res = assemble_packed(_job(win.origin_cell, 96, 0.5, materials, palette))
        assert res.albedo_bytes == ref_albedo
        assert res.emis_bytes == ref_emis
        assert res.origin_cell == win.origin_cell

    def test_downsampled_cascade_matches(self):
        # Cascade 1 (2.0 m cells → k=4 max-downsample) must also match.
        chunks = _world()
        palette = _palette()
        win = VolumeWindow(cells=96, cell_m=2.0)
        win.recenter((8.0, 8.0, 8.0))
        materials = {c: ch.materials for c, ch in chunks.items()}
        vol = assemble_geometry(win, chunks, palette, chunk_size=CHUNK, voxel_size=VOXEL)
        res = assemble_packed(_job(win.origin_cell, 96, 2.0, materials, palette))
        assert res.albedo_bytes == pack_volume(vol.albedo_occ)
        assert res.emis_bytes == pack_volume(vol.emission)


# ---------------------------------------------------------------------------
# CascadeAssemblyWorker thread
# ---------------------------------------------------------------------------


class TestCascadeAssemblyWorker:
    def test_worker_result_matches_inline(self):
        chunks = _world()
        palette = _palette()
        win = VolumeWindow(cells=96, cell_m=0.5)
        win.recenter((8.0, 8.0, 8.0))
        materials = {c: ch.materials for c, ch in chunks.items()}
        job = _job(win.origin_cell, 96, 0.5, materials, palette, seq=7)

        inline = assemble_packed(job)

        worker = CascadeAssemblyWorker()
        worker.start()
        try:
            worker.submit(job)
            results = _drain_until(worker, 1)
            assert len(results) == 1
            res = results[0]
            assert res.seq == 7
            assert res.albedo_bytes == inline.albedo_bytes
            assert res.emis_bytes == inline.emis_bytes
            assert worker.pending() == 0
        finally:
            worker.stop()

    def test_stop_is_clean_and_idempotent(self):
        worker = CascadeAssemblyWorker()
        worker.start()
        worker.stop()
        worker.stop()  # second stop is a no-op, must not raise

    def test_pending_count_tracks_inflight(self):
        chunks = _world()
        palette = _palette()
        win = VolumeWindow(cells=32, cell_m=0.5)
        win.recenter((8.0, 8.0, 8.0))
        materials = {c: ch.materials for c, ch in chunks.items()}
        worker = CascadeAssemblyWorker()
        worker.start()
        try:
            for i in range(3):
                worker.submit(_job(win.origin_cell, 32, 0.5, materials, palette, seq=i))
            drained = _drain_until(worker, 3)
            assert len(drained) == 3
            assert worker.pending() == 0
        finally:
            worker.stop()
