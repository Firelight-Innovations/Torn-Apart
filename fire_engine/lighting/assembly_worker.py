"""
lighting/assembly_worker.py — Background cascade-volume assembly.

The GPU lighting pipeline re-gathers a cascade's 96³ geometry volume from the
loaded chunk material arrays every time its window recenters (the camera flies
~8 cells) or terrain inside it changes.  That gather — slice + max-downsample +
palette index + transpose/channel-swap pack — is pure numpy and, measured on a
fly-around, costs ~90 ms p99 on the **main thread**, which is the dominant cause
of the flight stutter (``tools/profile_stream.py``).

This module moves that work onto a single background thread.  numpy releases the
GIL during the heavy array ops, so the worker genuinely overlaps the render
thread.  The main thread (``lighting/gpu.py``) keeps only the parts that must
stay there: snapshotting which chunk arrays to read, the cheap
``Texture.set_ram_image(bytes)`` GPU upload, and compute dispatch.

No panda3d import — fully headless-testable (this module is NOT excluded from
the headless suite, unlike its ``gpu.py`` consumer).

Threading contract
------------------
- The main thread builds an :class:`AssemblyJob` with a *snapshot* of the chunk
  material arrays the gather will read (references, not copies — see
  ``gpu.py``), submits it, and continues rendering with the previously
  committed volume.
- The worker assembles + packs and posts an :class:`AssemblyResult`.
- The main thread drains results, uploads the packed bytes, and only THEN
  commits the new window origin — so the GPU geometry texture and the shader's
  origin uniform are never out of sync (at most a 1–2 frame positional lag in
  the lighting volume, well within the window's existing hysteresis margin).

Determinism: a job's result depends only on its snapshot, origin, and palette,
so worker output is byte-identical to a synchronous ``assemble_geometry`` +
``pack_volume`` for the same inputs (asserted in ``tests/``).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from fire_engine.core import get_logger
from fire_engine.lighting.occluders import TreeOccluderSet
from fire_engine.lighting.palette import MaterialPalette
from fire_engine.lighting.volume import (
    ChunkBlockCache,
    VolumeWindow,
    assemble_geometry,
    pack_volume,
)

_log = get_logger("lighting.assembly_worker")

__all__ = [
    "AssemblyJob",
    "AssemblyResult",
    "CascadeAssemblyWorker",
    "assemble_packed",
]


@dataclass(frozen=True)
class AssemblyJob:
    """
    One cascade-volume reassembly request.

    Attributes
    ----------
    cascade_index : int
        Which cascade (0 or 1) this volume is for.
    origin_cell : tuple[int, int, int]
        Window origin (in cells) the volume is assembled for — committed to the
        cascade window only when the result is uploaded.
    cells, cell_m : int, float
        Window dimensions (texels per axis, meters per cell).
    chunk_size, voxel_size : int, float
        Terrain constants for the slice/downsample math.
    materials : dict[tuple[int, int, int], numpy.ndarray]
        Snapshot of ``uint8 (S, S, S)`` material arrays for the chunks the
        gather will read (coord → array).  Built on the main thread.
    palette : MaterialPalette
        Immutable material → albedo/emission lookup (safe to share read-only).
    seq : int
        Monotonic id; lets the consumer drop a superseded result.
    occluders : TreeOccluderSet | None
        Static tree/bush occluder snapshot splatted into the volume (see
        ``lighting/occluders.py``).  Immutable struct-of-arrays — safe to
        share read-only across the thread boundary.  ``None`` → chunks only.
    trunk_occ, canopy_occ : float
        Occluder splat opacities (``config.light_tree_*_occ``).
    """

    cascade_index: int
    origin_cell: tuple[int, int, int]
    cells: int
    cell_m: float
    chunk_size: int
    voxel_size: float
    materials: dict
    palette: MaterialPalette
    seq: int
    occluders: "TreeOccluderSet | None" = None
    trunk_occ: float = 0.0
    canopy_occ: float = 0.0


@dataclass(frozen=True)
class AssemblyResult:
    """
    A finished cascade volume, packed and ready for ``Texture.set_ram_image``.

    Attributes
    ----------
    cascade_index : int
    origin_cell : tuple[int, int, int]
        The origin the volume was assembled for (commit this to the window).
    albedo_bytes, emis_bytes : bytes
        Page-major BGRA 3-D-texture RAM images (see ``volume.pack_volume``).
    seq : int
        Echoes the job's ``seq``.
    """

    cascade_index: int
    origin_cell: tuple[int, int, int]
    albedo_bytes: bytes
    emis_bytes: bytes
    seq: int


def assemble_packed(
    job: AssemblyJob, cache: "ChunkBlockCache | None" = None,
) -> AssemblyResult:
    """
    Run one job: ``assemble_geometry`` on the snapshot, then ``pack_volume``.

    Pure function of the job (no shared state) — used both by the worker thread
    and, synchronously, by the boot/first-frame path and the tests.

    Parameters
    ----------
    job : AssemblyJob
    cache : ChunkBlockCache, optional
        Per-chunk downsampled-block cache passed through to
        :func:`assemble_geometry`.  Output is byte-identical with or without it.
    """
    window = VolumeWindow(cells=job.cells, cell_m=job.cell_m)
    window.origin_cell = job.origin_cell  # placed directly; no recenter needed
    vol = assemble_geometry(
        window, job.materials, job.palette,
        chunk_size=job.chunk_size, voxel_size=job.voxel_size, cache=cache,
        occluders=job.occluders,
        trunk_occ=job.trunk_occ, canopy_occ=job.canopy_occ)
    return AssemblyResult(
        cascade_index=job.cascade_index,
        origin_cell=job.origin_cell,
        albedo_bytes=pack_volume(vol.albedo_occ),
        emis_bytes=pack_volume(vol.emission),
        seq=job.seq,
    )


class CascadeAssemblyWorker:
    """
    Single background thread that assembles + packs cascade volumes.

    Lifecycle: :meth:`start` once after construction, :meth:`stop` at shutdown.
    The thread is a daemon, so a missed ``stop`` never blocks process exit.

    Producer/consumer
    -----------------
    - :meth:`submit` — main thread enqueues an :class:`AssemblyJob` (non-blocking).
    - :meth:`drain_results` — main thread pops all finished
      :class:`AssemblyResult`\\ s (non-blocking).
    - :meth:`pending` — number of jobs submitted but not yet returned.

    The two queues cross the thread boundary lock-free.  The one piece of
    shared mutable state is :attr:`block_cache` — the worker thread reads +
    populates it during assembly while the main thread calls
    :meth:`invalidate_chunk` / :meth:`clear_cache` on terrain edits; the cache
    guards itself with an internal lock, so that sharing is safe.

    Attributes
    ----------
    block_cache : ChunkBlockCache
        Per-chunk coarse-block cache reused across reassemblies (see
        :class:`fire_engine.lighting.volume.ChunkBlockCache`).
    """

    def __init__(self, *, cache_max_entries: int = 4096) -> None:
        self._in: "queue.Queue[AssemblyJob | None]" = queue.Queue()
        self._out: "queue.Queue[AssemblyResult]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending = 0
        self.block_cache = ChunkBlockCache(max_entries=cache_max_entries)

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="CascadeAssemblyWorker", daemon=True)
        self._thread.start()

    def submit(self, job: AssemblyJob) -> None:
        """Enqueue a reassembly job (main thread)."""
        self._pending += 1
        self._in.put(job)

    def drain_results(self) -> list[AssemblyResult]:
        """Pop and return all finished results (main thread, non-blocking)."""
        out: list[AssemblyResult] = []
        while True:
            try:
                res = self._out.get_nowait()
            except queue.Empty:
                break
            self._pending -= 1
            out.append(res)
        return out

    def pending(self) -> int:
        """Jobs submitted but not yet drained."""
        return self._pending

    def invalidate_chunk(self, coord: tuple[int, int, int]) -> None:
        """
        Drop the block cache's mini-blocks for ``coord`` (main thread, on a
        terrain edit) so the next reassembly recomputes them.  Thread-safe.
        """
        self.block_cache.invalidate(coord)

    def clear_cache(self) -> None:
        """Drop the entire block cache (e.g. world reload).  Thread-safe."""
        self.block_cache.clear()

    def stop(self, *, join: bool = True, timeout: float = 2.0) -> None:
        """Signal the worker to exit and (optionally) join it."""
        if self._thread is None:
            return
        self._in.put(None)  # sentinel
        if join:
            self._thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._in.get()
            if job is None:  # sentinel → shutdown
                break
            try:
                self._out.put(assemble_packed(job, cache=self.block_cache))
            except Exception:  # noqa: BLE001 — never let the worker die silently
                _log.exception(
                    "Cascade assembly failed (cascade %d, seq %d)",
                    job.cascade_index, job.seq)
                # Post a failure sentinel (empty bytes) so the consumer can
                # clear its in-flight flag and retry — a raised job must not
                # leave the cascade stuck forever.
                self._out.put(AssemblyResult(
                    cascade_index=job.cascade_index,
                    origin_cell=job.origin_cell,
                    albedo_bytes=b"", emis_bytes=b"", seq=job.seq))
