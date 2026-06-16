"""
Cascade volume assembly helpers for GpuLightingPipeline.

Extracted from ``fire_engine.lighting.gpu`` to keep that module under the
500-line limit.  All functions receive the pipeline instance as their first
argument and operate on its private attributes.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import LVecBase3i, ShaderAttrib

from fire_engine.lighting.assembly_worker import AssemblyJob, assemble_packed
from fire_engine.lighting.volume import (
    assemble_geometry,
    pack_volume,
    window_chunk_span,
)

if TYPE_CHECKING:
    from fire_engine.lighting._impl.gpu_cascade import _Cascade
    from fire_engine.lighting.gpu import GpuLightingPipeline

__all__ = [
    "apply_edits_sync",
    "assemble_and_upload_sync",
    "commit_assembly_result",
    "drain_assembly_results",
    "schedule_assembly",
    "shift_radiance",
    "submit_assembly",
    "upload_volume",
]


def upload_volume(tex: Any, arr: Any) -> None:
    """
    Upload a ``uint8 (N, N, N, 4)`` block to a 3-D texture.

    Panda3D RAM layout for 3-D textures is page-major ``(z, y, x)`` with BGRA
    channel order; ``pack_volume`` handles the transpose + channel swap before
    calling ``set_ram_image``.

    Docs: docs/systems/lighting.md
    """
    tex.set_ram_image(pack_volume(arr))


def assemble_and_upload_sync(pipeline: GpuLightingPipeline, casc: _Cascade) -> None:
    """
    Gather + upload one cascade volume inline on the main thread.

    Used only for the boot/first frame so the world is lit immediately;
    steady-state reassembly goes through the worker.

    Passes the worker's thread-safe ``block_cache`` (when threaded) so the
    synchronous boot/edit downsample warms the same cache the worker reuses —
    the coarse cascades skip re-downsampling already-seen chunks.  The cache is
    lock-guarded, so a concurrent worker read is safe.

    Docs: docs/systems/lighting.md
    """
    cache = pipeline._assembly_worker.block_cache if pipeline._assembly_worker is not None else None
    vol = assemble_geometry(
        casc.window,
        pipeline._provider.chunks,
        pipeline._palette,
        chunk_size=pipeline._config.chunk_size,
        voxel_size=pipeline._config.voxel_size,
        cache=cache,
        occluders=pipeline._tree_occluders,
        trunk_occ=float(pipeline._config.light_tree_trunk_occ),
        canopy_gain=float(pipeline._config.light_tree_canopy_extinction_gain),
    )
    upload_volume(casc.geom, vol.albedo_occ)
    upload_volume(casc.emis, vol.emission)
    casc.needs_inject = True
    pipeline._tree_occ_stale.discard(casc.index)


def apply_edits_sync(pipeline: GpuLightingPipeline) -> None:
    """
    Refresh brush-edited cascades' geometry SAME-FRAME (kills the black flash).

    Terrain edits (explosions, digging) must light immediately.  The
    steady-state async reassembly lags 1-2 frames; until it lands the stale
    occupancy still marks the new crater as solid (so its sun visibility is
    shadowed and its GI cell unlit) and it renders black, then pops to lit
    once the volume catches up — the "black then lit" artefact.

    Edits are discrete events (not per-frame like flying), so a synchronous
    gather of the few hit cascades is affordable.  A cascade window does not
    move on an edit, so this re-slices the live (already-edited) materials at
    the committed origin and forces re-injection this frame.  Cascades with
    an async job already in flight are skipped here and left to the normal
    ``_pending_coords`` path (rare for the camera-local cascade 0 during a
    stationary edit); the eager uploads here are otherwise redundant with —
    not conflicting with — that path, which keeps the committed-origin
    bookkeeping untouched.

    Docs: docs/systems/lighting.md
    """
    if not pipeline._edited_coords:
        return
    for casc in pipeline.cascades:
        # Only the near/mid cascades refresh synchronously: their crater is
        # what the player is looking at, and their full-window gather is
        # cheap (cascade 0 ~27, cascade 1 ~1.7k chunk coords).  The coarse
        # FAR cascade (index 2, ~33k coords over 512 m) would add a one-frame
        # hitch on every edit for a relight that lags invisibly at 96 m+, so
        # it stays on the async ``_pending_coords`` path.
        if casc.index >= 2:
            continue
        if casc._assembly_inflight:
            continue
        if casc.window.origin_cell is None:
            continue
        if not pipeline._any_coord_hits(pipeline._edited_coords, casc.window):
            continue
        assemble_and_upload_sync(pipeline, casc)  # sets needs_inject = True
    pipeline._edited_coords.clear()


def schedule_assembly(pipeline: GpuLightingPipeline, camera_pos: Any) -> None:
    """
    Submit reassembly jobs for cascades that moved or whose terrain changed.

    Non-mutating w.r.t. the committed window origin: ``needs_recenter`` and
    ``_coords_hit_window`` test against the *committed* origin, and a job is
    submitted for the new (snapped) origin without advancing it — the origin
    commits only when the matching volume lands (``drain_assembly_results``).
    At most one job per cascade is in flight at a time.

    Docs: docs/systems/lighting.md
    """
    from fire_engine.lighting.gpu import _LOAD_REASSEMBLE_INTERVAL_S  # local to avoid circular

    have_pending = bool(pipeline._pending_coords)
    batch_ready = have_pending and pipeline._load_dirty_timer >= _LOAD_REASSEMBLE_INTERVAL_S
    deferred = False  # a cascade still owes pending edits but is busy
    for casc in pipeline.cascades:
        moved = casc.window.needs_recenter(camera_pos)
        # Cascade 0 (the small 48 m near box) reassembles the instant a
        # newly-streamed chunk intersects it — its ~27-chunk gather is cheap
        # and the 0.25 s batch interval otherwise leaves freshly-loaded
        # near terrain rendering unshadowed until it fires, then it pops.
        # The mid/far cascades keep the batch interval (their frontier loads
        # are frequent and the relight lags invisibly far away).
        casc_ready = batch_ready if casc.index > 0 else have_pending
        hit = casc_ready and pipeline._coords_hit_window(casc.window)
        # The static tree-occluder set changed since this cascade's volume
        # was assembled (set_static_occluders) → re-splat at the committed
        # origin.
        stale = casc.index in pipeline._tree_occ_stale
        if casc._assembly_inflight:
            if hit:
                deferred = True
            continue
        if not (moved or hit or stale):
            continue
        # 'moved' → assemble for the new snapped origin; 'hit' (terrain
        # changed inside the window, origin unchanged) → re-assemble the
        # committed origin.  Either way the snapshot reads live materials
        # and the current occluder set, so a submit also satisfies any
        # pending 'hit' and clears occluder staleness.
        origin = casc.window._desired_origin(camera_pos) if moved else casc.window.origin_cell
        submit_assembly(pipeline, casc, origin)
        pipeline._tree_occ_stale.discard(casc.index)
    # Clear pending edits only once every cascade they touch has an
    # up-to-date (just-submitted or current) volume — otherwise a busy
    # cascade would silently miss the edit.  Gated on the batch interval so
    # the mid/far cascades (which only act on a batch) don't lose a coord a
    # c0-immediate pass already consumed.
    if batch_ready and not deferred:
        pipeline._pending_coords.clear()
        pipeline._load_dirty_timer = 0.0


def submit_assembly(pipeline: GpuLightingPipeline, casc: _Cascade, origin_cell: Any) -> None:
    """Snapshot the chunks a reassembly will read and enqueue the job.

    Docs: docs/systems/lighting.md
    """
    coords = window_chunk_span(
        origin_cell,
        casc.cells,
        casc.cell_m,
        int(pipeline._config.chunk_size),
        float(pipeline._config.voxel_size),
    )
    live = pipeline._provider.chunks
    # Snapshot *references* to the material arrays (not copies): cheap, and
    # safe against streaming (dict membership changes don't affect captured
    # arrays).  A concurrent in-place brush edit of a captured array is the
    # only race; it self-corrects on the next reassembly.
    materials = {c: live[c].materials for c in coords if c in live}
    pipeline._assembly_seq += 1
    job = AssemblyJob(
        cascade_index=casc.index,
        origin_cell=tuple(origin_cell),
        cells=casc.cells,
        cell_m=casc.cell_m,
        chunk_size=int(pipeline._config.chunk_size),
        voxel_size=float(pipeline._config.voxel_size),
        materials=materials,
        palette=pipeline._palette,
        seq=pipeline._assembly_seq,
        occluders=pipeline._tree_occluders,
        trunk_occ=float(pipeline._config.light_tree_trunk_occ),
        canopy_gain=float(pipeline._config.light_tree_canopy_extinction_gain),
    )
    if pipeline._threaded:
        casc._assembly_inflight = True
        casc._pending_seq = pipeline._assembly_seq
        assert pipeline._assembly_worker is not None
        pipeline._assembly_worker.submit(job)
    else:
        # Inline (tooling/tests): assemble + commit immediately.
        commit_assembly_result(pipeline, assemble_packed(job))


def drain_assembly_results(pipeline: GpuLightingPipeline) -> None:
    """Upload finished volumes and commit their window origins (main thread).

    Docs: docs/systems/lighting.md
    """
    if pipeline._assembly_worker is None:
        return
    for res in pipeline._assembly_worker.drain_results():
        commit_assembly_result(pipeline, res)


def commit_assembly_result(pipeline: GpuLightingPipeline, res: Any) -> None:
    """Upload one finished volume and advance its committed window origin.

    Docs: docs/systems/lighting.md
    """
    casc = pipeline.cascades[res.cascade_index]
    if pipeline._threaded and res.seq != casc._pending_seq:
        casc._assembly_inflight = False
        return  # superseded (single-inflight makes this rare)
    casc._assembly_inflight = False
    if not res.albedo_bytes:
        return  # assembly failed → flag cleared, retry next frame
    casc.geom.set_ram_image(res.albedo_bytes)
    casc.emis.set_ram_image(res.emis_bytes)
    # Radiance continuity: the two ping-pong textures still hold the OLD
    # window's field at the OLD origin.  If this commit moves the origin,
    # shift the current radiance by the integer cell delta so the same-
    # frame re-gather's feedback term (and anything sampling radiance
    # before it lands) reads a spatially-aligned field.
    old_origin = casc.window.origin_cell
    new_origin = tuple(res.origin_cell)
    if old_origin is not None and old_origin != new_origin:
        shift_radiance(pipeline, casc, old_origin, new_origin)
    # Commit: the GPU geom texture now matches this origin, so the shader
    # origin uniforms (read from window.origin_cell) line up exactly.
    casc.window.origin_cell = new_origin
    casc.needs_inject = True


def shift_radiance(
    pipeline: GpuLightingPipeline,
    casc: _Cascade,
    old_origin: tuple[int, int, int],
    new_origin: tuple[int, int, int],
) -> None:
    """
    Copy ``casc``'s current radiance into its other ping-pong texture,
    offset by the recenter cell delta, then swap so the next gather reads
    the spatially-aligned field (kills the recenter GI pop).

    ``u_shift = new_origin - old_origin``: a cell at new-window index ``c``
    holds the same world cell the previous window held at index
    ``c + u_shift``; source cells outside the previous window become
    ``vec4(0)`` (the newly-exposed border band).

    Docs: docs/systems/lighting.md
    """
    from fire_engine.lighting.gpu import _groups  # local to avoid circular

    shift = tuple(int(new_origin[i] - old_origin[i]) for i in range(3))
    node = casc.shift_np[casc.ping]  # reads radiance[ping] → other
    node.set_shader_input("u_shift", LVecBase3i(*shift))
    gsg = pipeline._base.win.get_gsg()
    engine = pipeline._base.graphicsEngine
    groups = (_groups(casc.cells, 4),) * 3
    engine.dispatch_compute(groups, node.get_attrib(ShaderAttrib), gsg)
    casc.ping ^= 1  # the shifted texture is now current
