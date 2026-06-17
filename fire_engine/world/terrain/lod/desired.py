"""
terrain/lod/desired.py — Vectorised desired-node planner (`desired_node_set`).

:func:`desired_node_set` replaces the legacy ``O(r³)`` triple loop in
``ChunkManager.desired_set`` with a single :func:`numpy.meshgrid` sweep over the
XY chunk window × the Z band, a bulk Chebyshev horizontal-distance array, a
per-cell rank via :func:`numpy.searchsorted` against the LOD band radii, and a
node snap (``cell >> rank``) + :func:`numpy.unique` dedup.  It returns a
:class:`NodePlan` partitioning the window into the near (``L0``) chunk set and
the coarse-node sets per rank.

Hard band cuts (P2): every cell belongs to **exactly one** rank, so the near
set and every coarse-rank set are mutually disjoint — a column is either an
editable ``L0`` chunk OR in exactly one coarse node, never both (no double-draw;
crossfade is P3).

Regression invariant (pinned by tests)
---------------------------------------
With ``max_rank=0`` and ``near_radius_chunks = view_distance_chunks = 6`` and
``z_band = (-2, 4)``, ``near_chunks`` equals the legacy
``ChunkManager.desired_set`` output **exactly** — the square (Chebyshev) XY
radius-6 × Z[-2..4] set of ``13·13·7 = 1183`` chunks.  This keeps the off-thread
P1 near-streaming path byte-identical when coarse ranks are disabled.

No RNG (Hard Rule 2 — this layer is deterministic and seed-independent).
No panda3d import (Hard Rule 1).  No per-voxel loops (Hard Rule 4 — meshgrid +
whole-array ops; the only iteration is over the ≤3 coarse ranks).

Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fire_engine.world.terrain.lod.node import LodNode

if TYPE_CHECKING:
    from fire_engine.core.config import Config

__all__ = ["NodePlan", "desired_node_set"]

# Default LOD band radii (meters) — used when Config lacks the P2 keys (the
# keys are added in step 2; read here via getattr so step 1 stays config-free).
_DEFAULT_BAND_L1_M: float = 32.0
_DEFAULT_BAND_L2_M: float = 96.0
_DEFAULT_BAND_L3_M: float = 192.0
_DEFAULT_NEAR_RADIUS_CHUNKS: int = 6
_DEFAULT_FAR_RADIUS_CHUNKS: int = 32


@dataclass(frozen=True)
class NodePlan:
    """
    The partitioned desired set for one camera frame (trivial frozen dataclass).

    Attributes
    ----------
    near_chunks : set[tuple[int, int, int]]
        Native ``L0`` chunk coords ``(cx, cy, cz)`` that stay editable/lit/saved
        (rank 0).  At ``max_rank=0`` this equals the legacy desired set exactly.
        At ``max_rank≥1`` it is **still** the full Chebyshev radius-``near_r``
        square × Z band — near is authoritative inside ``near_r`` (the editable
        footprint never shrinks when coarse ranks turn on; see
        :func:`desired_node_set`).
    coarse_nodes : dict[int, set[tuple[int, int, int, int]]]
        Maps rank ``L`` (``1..max_rank``) to the set of coarse-node keys
        ``(L, nx, ny, nz)`` at that rank.  Each node key is unique (deduped via
        :func:`numpy.unique`); each covers ``(2**L)³`` chunk columns.

    Invariant
    ---------
    ``near_chunks`` and every value of ``coarse_nodes`` are pairwise disjoint in
    the columns they cover (hard band cuts), AND ``near_chunks`` contains every
    chunk whose XY column is within Chebyshev ``near_r`` of the camera column
    (the editable radius is exactly ``near_r``, coarse fills strictly beyond).
    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """

    near_chunks: set[tuple[int, int, int]] = field(default_factory=set)
    coarse_nodes: dict[int, set[tuple[int, int, int, int]]] = field(default_factory=dict)


def _cfg_float(config: Config | None, name: str, default: float) -> float:
    """Read float config key ``name`` (or ``default`` when absent / no config).

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    if config is None:
        return float(default)
    return float(getattr(config, name, default))


def _band_radii_m(config: Config | None) -> np.ndarray:
    """Ascending LOD band radii ``[l1, l2, l3]`` (meters) from config (or defaults).

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    return np.array(
        [
            _cfg_float(config, "lod_band_l1_m", _DEFAULT_BAND_L1_M),
            _cfg_float(config, "lod_band_l2_m", _DEFAULT_BAND_L2_M),
            _cfg_float(config, "lod_band_l3_m", _DEFAULT_BAND_L3_M),
        ],
        dtype=np.float64,
    )


def _entry_radii_chunks(
    config: Config | None, near_r: int, chunk_meters: float, max_rank: int
) -> dict[int, float]:
    """
    Per-rank entry radii ``enter[L]`` (nearest-corner Chebyshev distance, chunks).

    A block is allowed at rank ``L`` once its nearest chunk-corner is at least
    ``enter[L]`` chunks from the camera column.  ``enter[1]`` is the near radius;
    ``enter[L]`` for ``L≥2`` is the band radius for the ``(L-1)→L`` transition,
    clamped to be ≥ ``enter[1]`` (with the default config near = 96 m subsumes the
    32/96 m L1/L2 bands, which simply means L1/L2 may be empty; the partition
    stays valid).  NOTE: these radii only *gate* coarse promotion by distance —
    the near footprint itself is made authoritative by the radius-``near_r``
    square pre-claim in :func:`desired_node_set`, so a boundary block at
    nearest-corner distance exactly ``near_r`` is rejected (it overlaps the
    pre-claimed near square) rather than swallowing the edge ring.
    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    band_chunks = _band_radii_m(config) / float(chunk_meters)  # [l1, l2, l3]
    enter: dict[int, float] = {1: float(near_r)}
    for L in range(2, max_rank + 1):
        # bands index: L=2 -> band_chunks[1] (l2, the 1->2 boundary), etc.
        enter[L] = max(float(near_r), float(band_chunks[L - 1]))
    return enter


def _nearest_axis_dist(block_lo: np.ndarray, k: int, camera: int) -> np.ndarray:
    """
    Per-axis distance (chunks) from ``camera`` to the nearest chunk of a block.

    The block spans chunk coords ``[block_lo, block_lo + k)``.  Distance is 0
    when the camera lies inside the span, else the gap to the nearer edge.
    Vectorised over ``block_lo``.
    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    below = block_lo - camera  # >0 when block is entirely above camera
    above = camera - (block_lo + k - 1)  # >0 when block is entirely below camera
    out: np.ndarray = np.maximum(np.maximum(below, above), 0)
    return out


def _near_radius(config: Config | None, override: int | None) -> int:
    """Resolve the near (rank-0) Chebyshev radius in chunks.

    Precedence: explicit ``override`` → ``Config.lod_near_radius_chunks`` →
    ``Config.view_distance_chunks`` → the module default.
    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    if override is not None:
        return int(override)
    if config is None:
        return _DEFAULT_NEAR_RADIUS_CHUNKS
    val = getattr(config, "lod_near_radius_chunks", None)
    if val is None:
        val = getattr(config, "view_distance_chunks", _DEFAULT_NEAR_RADIUS_CHUNKS)
    return int(val if val is not None else _DEFAULT_NEAR_RADIUS_CHUNKS)


def desired_node_set(
    camera_chunk: tuple[int, int, int],
    config: Config | None,
    z_band: tuple[int, int],
    max_rank: int,
    *,
    near_radius_chunks: int | None = None,
    far_radius_chunks: int | None = None,
    chunk_meters: float | None = None,
) -> NodePlan:
    """
    Plan the desired near chunks + coarse nodes for ``camera_chunk``.

    Builds the XY chunk window ``[-R, R]²`` (``R`` = far radius when
    ``max_rank > 0`` else near radius) × the Z band once via
    :func:`numpy.meshgrid`, classifies every cell by Chebyshev horizontal
    distance, and partitions into the near (``L0``) set and per-rank coarse-node
    sets.  Vectorised end-to-end (Hard Rule 4); deterministic (Hard Rule 2).

    Parameters
    ----------
    camera_chunk : tuple[int, int, int]
        Integer chunk coord ``(ccx, ccy, ccz)`` the camera is in.
    config : Config | None
        Engine config.  Read for the band radii (``lod_band_l{1,2,3}_m``), the
        near radius (``lod_near_radius_chunks`` then ``view_distance_chunks``),
        the far radius (``lod_far_radius_chunks``) and ``chunk_meters`` — each
        via ``getattr`` with documented defaults so this works before the P2
        config keys land (step 2).  May be ``None`` in unit tests, in which case
        the explicit keyword args / defaults are used.
    z_band : tuple[int, int]
        Inclusive Z chunk offsets relative to ``ccz`` (legacy ``(-2, 4)``).
    max_rank : int
        Highest coarse rank to emit (``0`` = near only — the regression case).
    near_radius_chunks : int | None, keyword
        Chebyshev XY radius (chunks) of the rank-0 near region; overrides config.
    far_radius_chunks : int | None, keyword
        Outer Chebyshev XY radius (chunks) of the coarse window; overrides config.
    chunk_meters : float | None, keyword
        Native chunk edge in meters (16 m); overrides config.

    Returns
    -------
    NodePlan
        ``near_chunks`` + ``coarse_nodes`` (disjoint).

    Example
    -------
    >>> plan = desired_node_set((0, 0, 0), None, (-2, 4), max_rank=0,
    ...                         near_radius_chunks=6)
    >>> len(plan.near_chunks)              # 13*13*7
    1183
    >>> plan.coarse_nodes
    {}

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    ccx, ccy, ccz = camera_chunk
    near_r = _near_radius(config, near_radius_chunks)
    if far_radius_chunks is not None:
        far_r = int(far_radius_chunks)
    else:
        far_r = int(_cfg_float(config, "lod_far_radius_chunks", _DEFAULT_FAR_RADIUS_CHUNKS))
    if chunk_meters is not None:
        cm_m = float(chunk_meters)
    else:
        cm_m = _cfg_float(config, "chunk_meters", 16.0)
    max_rank = int(max_rank)
    z_lo, z_hi = int(z_band[0]), int(z_band[1])

    plan = NodePlan(near_chunks=set(), coarse_nodes={})

    if max_rank <= 0:
        # --- Near-only (P1 / regression): the exact Chebyshev radius square. ---
        # One meshgrid over the XY window × Z band; the set equals the legacy
        # ChunkManager.desired_set output exactly (the 1183-chunk lock).
        dxs = np.arange(-near_r, near_r + 1)
        dzs = np.arange(z_lo, z_hi + 1)
        gx, gy, gz = np.meshgrid(dxs, dxs, dzs, indexing="ij")
        cx, cy, cz = gx + ccx, gy + ccy, gz + ccz
        plan.near_chunks.update(
            zip(cx.ravel().tolist(), cy.ravel().tolist(), cz.ravel().tolist(), strict=True)
        )
        return plan

    # --- Coarse ranks active: a NESTED QUADTREE (provably a partition). -------
    #
    # Every coarse node is aligned to global ``2**L`` chunk blocks (``col >> L``),
    # so a rank-L node tiles exactly 8 rank-(L-1) children and rank levels nest.
    # We assign each block by its NEAREST-corner Chebyshev distance to the camera
    # chunk and process COARSEST → FINEST, marking covered chunks: a block is
    # emitted at rank L iff (a) it is far enough for rank L (nearest-corner ≥ the
    # rank-L entry radius) and (b) none of its chunks were already claimed by a
    # coarser block OR the authoritative near square.  Because nearest-corner
    # distance is monotone across nesting and the entry radii are strictly
    # increasing, this covers every windowed chunk exactly once — no overlap, no
    # gap (hard band cuts).  The only loops are over the ≤3 ranks and the
    # ≤(2·far+1)² blocks per rank — never voxels.
    #
    # NEAR IS AUTHORITATIVE inside ``near_r`` (P2 boundary fix): the full
    # Chebyshev radius-``near_r`` XY square × Z band is pre-claimed before the
    # coarse sweep, so any coarse block that overlaps even one near column is
    # rejected by the ``claimed`` guard below.  This guarantees the editable
    # (rank-0) footprint stays the exact radius-``near_r`` square the NodePlan
    # docstring promises — without the pre-claim, a boundary rank-1 block has
    # nearest-corner Chebyshev distance EXACTLY ``near_r`` (``enter[1] == near_r``
    # with ``keep = nearest >= enter[L]``) and silently swallows the ±near_r edge
    # ring (~16 m/side), shrinking the brush/save/light radius once coarse is on.
    enter = _entry_radii_chunks(config, near_r, cm_m, max_rank)  # enter[L] for L≥1

    z_blocks = np.arange(z_lo, z_hi + 1) + ccz  # absolute chunk-z layers (band)

    # Pre-claim the authoritative near square (radius-near_r Chebyshev × Z band)
    # so coarse blocks can only fill STRICTLY beyond it (vectorised, Hard Rule 4).
    # These columns ARE the editable footprint, so they go straight into
    # ``near_chunks`` AND seed ``claimed`` (the overlap guard the coarse sweep
    # rejects against) — a coarse block touching any of them is dropped.
    near_dxs = np.arange(-near_r, near_r + 1)
    ngx, ngy, ngz = np.meshgrid(near_dxs, near_dxs, z_blocks - ccz, indexing="ij")
    near_square = set(
        zip(
            (ngx + ccx).ravel().tolist(),
            (ngy + ccy).ravel().tolist(),
            (ngz + ccz).ravel().tolist(),
            strict=True,
        )
    )
    plan.near_chunks.update(near_square)
    claimed: set[tuple[int, int, int]] = set(near_square)

    for L in range(max_rank, 0, -1):
        k = 1 << L
        # Candidate rank-L node coords whose block intersects the XY window.
        nlo_x, nhi_x = (ccx - far_r) >> L, (ccx + far_r) >> L
        nlo_y, nhi_y = (ccy - far_r) >> L, (ccy + far_r) >> L
        nxs = np.arange(nlo_x, nhi_x + 1)
        nys = np.arange(nlo_y, nhi_y + 1)
        bx, by = np.meshgrid(nxs, nys, indexing="ij")
        # Block chunk-coord span [n·k, n·k+k); nearest-corner Chebyshev distance
        # (chunks) from the camera column to the block, 0 if the camera is inside.
        near_dx = _nearest_axis_dist(bx * k, k, ccx)
        near_dy = _nearest_axis_dist(by * k, k, ccy)
        nearest = np.maximum(near_dx, near_dy)
        keep = nearest >= enter[L]
        if not keep.any():
            continue
        rank_set: set[tuple[int, int, int, int]] = set()
        for nx, ny in zip(bx[keep].tolist(), by[keep].tolist(), strict=True):
            for nz in (z_blocks >> L).tolist():
                node = (L, nx, ny, nz)
                cols = LodNode(*node).covered_chunks()
                if any(c in claimed for c in cols):
                    continue  # a finer/coarser sibling already owns part of it
                rank_set.add(node)
                claimed.update(cols)
        if rank_set:
            plan.coarse_nodes[L] = rank_set

    # Rank 0 (near): the pre-claimed authoritative square (already added above)
    # PLUS any windowed chunk beyond it that no coarse block claimed (keeps the
    # partition complete — no gaps in the far window).
    dxs = np.arange(-far_r, far_r + 1)
    wx, wy, wz = np.meshgrid(dxs, dxs, z_blocks - ccz, indexing="ij")
    near_x = (wx + ccx).ravel().tolist()
    near_y = (wy + ccy).ravel().tolist()
    near_z = (wz + ccz).ravel().tolist()
    for x, y, z in zip(near_x, near_y, near_z, strict=True):
        if (x, y, z) not in claimed:
            plan.near_chunks.add((x, y, z))
    return plan
