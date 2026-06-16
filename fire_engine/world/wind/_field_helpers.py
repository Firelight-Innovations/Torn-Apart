"""
wind/_field_helpers.py — Private helpers extracted from wind/field.py.

Contains the standalone utility functions (:func:`vertical_profile`,
:func:`pack_wind_field`) and the venturi-orchestration method-cluster helpers
(:func:`_venturi_step`, :func:`_commit_venturi`, :func:`_snapshot_materials`)
that were extracted from :class:`~fire_engine.world.wind.field.WindField` to keep
``field.py`` under 500 lines.

These are **internal implementation details** of the wind package.  The public
API is still ``from fire_engine.world.wind.field import vertical_profile,
pack_wind_field`` (re-exported from ``field.py``).

Docs: docs/systems/world.wind.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fire_engine.core.config import Config
from fire_engine.world.wind.types import VenturiJob, VenturiResult, WindSnapshot

if TYPE_CHECKING:
    from fire_engine.world.wind.field import WindField

__all__ = ["pack_wind_field", "vertical_profile"]


# ---------------------------------------------------------------------------
# Standalone public utilities
# ---------------------------------------------------------------------------


def vertical_profile(z: np.ndarray, z_ground: float, cfg: Config) -> np.ndarray:
    """
    Analytic boundary-layer wind-speed multiplier vs. height above ground.

    A power-law wind-shear profile clamped to a floor and a cap::

        m = clamp( ( max(z - z_ground, 0) / z_ref ) ** shear, floor, cap )

    So wind never fully dies at ground level (``floor``, default 0.35 — grass
    still sways), grows with height to 1.0 at ``z_ref`` (default 10 m), and
    saturates at ``cap`` (default 1.6) high up.  Monotonically non-decreasing
    in ``z`` between the floor and cap.

    Parameters
    ----------
    z : numpy.ndarray
        World heights in meters (any shape).
    z_ground : float
        Ground height in meters at the sample (the profile is 0-anchored here).
    cfg : Config
        Reads ``wind_shear``, ``wind_profile_z_ref``, ``wind_profile_floor``,
        ``wind_profile_cap``.

    Returns
    -------
    numpy.ndarray
        Same shape as ``z``: the per-height speed multiplier, in
        ``[floor, cap]``.

    Example
    -------
    >>> import numpy as np
    >>> from fire_engine.core.config import Config
    >>> m = vertical_profile(np.array([0.0, 10.0, 100.0]), 0.0, Config())
    >>> bool(m[0] == Config().wind_profile_floor)   # floor at ground
    True
    >>> bool(m[1] >= m[0] and m[2] >= m[1])          # monotone
    True

    Docs: docs/systems/world.wind.md
    """
    shear = float(cfg.wind_shear)
    z_ref = float(cfg.wind_profile_z_ref)
    floor = float(cfg.wind_profile_floor)
    cap = float(cfg.wind_profile_cap)
    above = np.maximum(np.asarray(z, dtype=np.float32) - float(z_ground), 0.0)
    prof = (above / z_ref) ** shear
    return np.asarray(np.clip(prof, floor, cap), dtype=np.float32)


def pack_wind_field(snap: WindSnapshot) -> bytes:
    """
    Pack a :class:`WindSnapshot` into Panda3D 2-D-texture RAM bytes.

    Produces a **float16** buffer in Panda3D's 2-D RAM layout: **row-major
    ``(y, x)``** (the field is stored ``[x, y]``, so it is transposed) with
    **BGRA** channel order — i.e. ``B = turb, G = vy, R = vx, A = horizontal
    speed`` (``hypot(vx, vy)``).  This mirrors
    ``lighting/volume.pack_volume``'s transpose + channel-swap convention so an
    upload is just ``Texture.set_ram_image(bytes)`` on the render thread.  Pure
    and thread-safe (no shared state) — safe to call off the main thread.

    LAYOUT IS PINNED (a test asserts it): if you change the transpose order or
    channel mapping you must update the GPU uniform contract
    (``u_wind_tex`` R=vx G=vy B=turb A=speed) and the shader decode together.

    Parameters
    ----------
    snap : WindSnapshot
        The field snapshot to pack.

    Returns
    -------
    bytes
        ``cells * cells * 4 * 2`` bytes of little-endian float16, ready for
        ``Texture(F_rgba16).set_ram_image``.

    Example
    -------
    >>> data = pack_wind_field(field.snapshot)
    >>> len(data) == field.snapshot.cells ** 2 * 4 * 2
    True

    Docs: docs/systems/world.wind.md
    """
    f = snap.field  # (cells, cells, 4) [x, y]: vx, vy, turb, reserved
    vx = f[..., 0]
    vy = f[..., 1]
    turb = f[..., 2]
    speed = np.hypot(vx, vy)

    # Build the RGBA-in-shader buffer in the texel's channel order, then
    # transpose [x, y] -> [y, x] (Panda3D 2-D RAM is row-major y outer) and
    # swap RGBA -> BGRA.  Mirrors pack_volume's transpose+swap discipline.
    rgba = np.stack([vx, vy, turb, speed], axis=-1)  # R, G, B, A
    bgra = rgba[..., [2, 1, 0, 3]]  # B, G, R, A
    data = np.ascontiguousarray(np.transpose(bgra, (1, 0, 2)).astype(np.float16))  # (y,x,4) fp16
    return data.tobytes()


# ---------------------------------------------------------------------------
# Venturi orchestration method-cluster helpers (extracted from WindField)
# ---------------------------------------------------------------------------


def _snapshot_materials(
    chunks: dict[tuple[int, int, int], Any],
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Build the ``coord -> uint8 materials`` snapshot the worker reads.

    Accepts either chunk objects (reads ``.materials``) or bare ndarrays
    (mirrors ``lighting`` assembly-worker's dual acceptance).  References,
    not copies — the arrays are treated as immutable for the solve's life.
    """
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for coord, ch in chunks.items():
        out[coord] = getattr(ch, "materials", ch)
    return out


def _venturi_step(
    wf: WindField,
    recentered: bool,
    chunks: dict[tuple[int, int, int], Any] | None,
    mean: tuple[float, float],
) -> None:
    """
    Submit / drain the venturi worker and update the applied correction.

    Pure orchestration (no field math): decide whether a fresh
    :class:`~fire_engine.world.wind.types.VenturiJob` is warranted, submit it,
    then drain finished results and commit the newest one whose
    ``origin_cell`` still matches the region's current origin.

    Submit when (and only when) there is a worker AND any of:

    - the region **recentered** this update (the old grid is for a stale
      origin — the renderer signals dirt by re-passing ``chunks`` too, but
      recenter alone is enough to re-solve), OR
    - ``chunks`` is available and **no job has ever been submitted** (first
      terrain solve), OR
    - ``chunks`` is not ``None`` — the renderer passes ``chunks`` *only* on
      a recenter or terrain-edit (dirty) event, so a non-``None`` ``chunks``
      is itself the recompute request (keeps ``wind/`` bus-free).

    Origin-match discipline (a Gotcha): a result solved for a previous
    origin is **discarded**, never shift-applied — the field re-submits on
    recenter and applies identity in the meantime.  This keeps the applied
    grid and the cells it scales perfectly aligned with zero index math.
    """
    worker = wf._worker
    if worker is None:
        return

    want_submit = (
        recentered
        or (chunks is not None and not wf._venturi_ever_submitted)
        or (chunks is not None)
    )
    if want_submit and chunks is not None:
        wf._venturi_seq += 1
        wf._venturi_ever_submitted = True
        assert wf._region.origin_cell is not None
        ground = wf._z_ground
        job = VenturiJob(
            origin_cell=wf._region.origin_cell,
            cells=int(wf._region.cells),
            cell_m=float(wf._region.cell_m),
            chunk_size=int(wf._cfg.chunk_size),
            voxel_size=float(wf._cfg.voxel_size),
            ground_band=(ground, ground + float(wf._cfg.wind_layer_m)),
            materials=_snapshot_materials(chunks),
            venturi_iters=int(wf._cfg.wind_venturi_iters),
            venturi_max=float(wf._cfg.wind_venturi_max),
            deflect_gain=float(wf._cfg.wind_deflect_gain),
            seq=wf._venturi_seq,
        )
        worker.submit(job)

    # Drain all finished results; keep only the newest (highest seq).
    newest: VenturiResult | None = None
    for res in worker.drain_results():
        if newest is None or res.seq >= newest.seq:
            newest = res
    if newest is not None:
        _commit_venturi(wf, newest)

    # A correction solved for an origin we have since moved away from must
    # not be applied — drop back to identity until a matching result lands.
    if wf._venturi_origin != wf._region.origin_cell:
        wf._venturi_speedup.fill(1.0)
        wf._venturi_deflect.fill(0.0)
        wf._updraft_gain_grid.fill(0.0)
        wf._venturi_origin = None


def _commit_venturi(wf: WindField, res: VenturiResult) -> None:
    """
    Apply a drained :class:`~fire_engine.world.wind.types.VenturiResult`.

    Only commits if the result's ``origin_cell`` matches the region's
    current origin (else it is a stale result for a window we have left —
    discard it, identity holds).  Derives the vz updraft-gain grid from the
    committed speed-up: ``wind_updraft_gain * clip(speedup - 1, 0, None)``.
    """
    if res.origin_cell != wf._region.origin_cell:
        return  # stale — discard (origin-match discipline)
    wf._venturi_speedup = res.speedup
    wf._venturi_deflect = res.deflect
    wf._venturi_origin = res.origin_cell
    wf._updraft_gain_grid = (
        float(wf._cfg.wind_updraft_gain) * np.clip(res.speedup - 1.0, 0.0, None)
    ).astype(np.float32)
