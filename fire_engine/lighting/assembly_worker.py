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

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

from fire_engine.core import get_logger
from fire_engine.core._impl.worker import QueueWorker
from fire_engine.lighting._impl.types import AssemblyJob, AssemblyResult
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


def assemble_packed(
    job: AssemblyJob,
    cache: ChunkBlockCache | None = None,
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

    Docs: docs/systems/lighting.md
    """
    window = VolumeWindow(cells=job.cells, cell_m=job.cell_m)
    window.origin_cell = job.origin_cell  # placed directly; no recenter needed
    vol = assemble_geometry(
        window,
        job.materials,
        job.palette,
        chunk_size=job.chunk_size,
        voxel_size=job.voxel_size,
        cache=cache,
        occluders=job.occluders,
        trunk_occ=job.trunk_occ,
        canopy_gain=job.canopy_gain,
    )
    return AssemblyResult(
        cascade_index=job.cascade_index,
        origin_cell=job.origin_cell,
        albedo_bytes=pack_volume(vol.albedo_occ),
        emis_bytes=pack_volume(vol.emission),
        seq=job.seq,
    )


class CascadeAssemblyWorker(QueueWorker[AssemblyJob, AssemblyResult]):
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

    Docs: docs/systems/lighting.md
    """

    def __init__(self, *, cache_max_entries: int = 4096) -> None:
        super().__init__("CascadeAssemblyWorker")
        self.block_cache = ChunkBlockCache(max_entries=cache_max_entries)

    def _process(self, job: AssemblyJob) -> AssemblyResult:
        return assemble_packed(job, cache=self.block_cache)

    def _on_error(self, job: AssemblyJob) -> None:
        _log.exception("Cascade assembly failed (cascade %d, seq %d)", job.cascade_index, job.seq)
        # Post a failure sentinel (empty bytes) so the consumer can
        # clear its in-flight flag and retry — a raised job must not
        # leave the cascade stuck forever.
        self._out.put(
            AssemblyResult(
                cascade_index=job.cascade_index,
                origin_cell=job.origin_cell,
                albedo_bytes=b"",
                emis_bytes=b"",
                seq=job.seq,
            )
        )

    def invalidate_chunk(self, coord: tuple[int, int, int]) -> None:
        """
        Drop the block cache's mini-blocks for ``coord`` (main thread, on a
        terrain edit) so the next reassembly recomputes them.  Thread-safe.

        Docs: docs/systems/lighting.md
        """
        self.block_cache.invalidate(coord)

    def clear_cache(self) -> None:
        """Drop the entire block cache (e.g. world reload).  Thread-safe.

        Docs: docs/systems/lighting.md
        """
        self.block_cache.clear()
