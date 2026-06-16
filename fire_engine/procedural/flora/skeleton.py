"""
procedural/flora/skeleton.py — branch-skeleton growth helpers for 3-D flora.

This is the heart of the **species-script API**: a tree or bush species is a
plain Python file (``procedural/flora/species/*.py``) that grows a branch
skeleton by chaining calls on a :class:`SkeletonBuilder` — the engine's
"node-graph editor in code".  The builder produces a :class:`TreeSkeleton`,
a struct-of-arrays description of every branch segment (start/end points,
tapered radii, parent links, sway weights) that the shared mesher
(``procedural/flora/mesher.py``) turns into renderable geometry and the
impostor rasterizer (``procedural/flora/impostor.py``) turns into far-LOD
sprites.

The design follows the Minecraft *Dynamic Trees* mod: a tree is a connected
graph of tapering segments rooted at the origin; branches sprout from points
ON their parent at script-controlled angle sets (≈90° entries give the
blocky look); leaves attach to branch tips.  Because every ``start`` point is
constructed on the parent segment, "floating canopy" bugs are structurally
impossible — and :func:`validate_skeleton` machine-checks it anyway.

Determinism: every builder method consumes the ``numpy.random.Generator``
passed to ``__init__`` (seeded upstream via ``core.rng.for_domain`` — Hard
Rule 2).  Same rng state → byte-identical skeleton.

Scale note: a tree has tens of segments.  Small fixed Python loops over
*growth levels* (trunk → limbs → twigs) are fine; all per-segment math
inside one call is vectorized numpy (Hard Rule 4 applies to per-element
loops over large arrays, not to a 3-call growth recipe).

Units: meters, radians, Z-up.  The trunk base is at the local origin
``(0, 0, 0)``; the renderer translates instances onto the terrain.

Example — a minimal species recipe (see ``species/gnarled_oak.py``)
-------------------------------------------------------------------
::

    import math
    import numpy as np
    from fire_engine.procedural.flora import SkeletonBuilder

    sb = SkeletonBuilder(rng)
    trunk = sb.trunk(height_m=5.0, base_radius_m=0.25, segments=4,
                     wobble_m=0.3)
    limbs = sb.branches(trunk, count=(3, 5),
                        pitch_set=(math.radians(85),),   # blocky right angles
                        length_ratio=(0.4, 0.6))
    sk = sb.skeleton()                                   # finalized arrays

Docs: docs/systems/procedural.md
"""

from __future__ import annotations

import math

import numpy as np

# TreeSkeleton and validate_skeleton live in types.py (grouping module);
# re-exported here so all historical import paths remain valid.
from fire_engine.procedural.flora.types import (
    TreeSkeleton,
    validate_skeleton,
)

__all__ = ["SkeletonBuilder", "TreeSkeleton", "validate_skeleton"]

_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
_TWO_PI = 2.0 * math.pi
# Golden angle (radians) — used by yaw_mode="spiral" so successive branches
# around one parent never stack on the same side (phyllotaxis).
_GOLDEN_ANGLE = 2.399963229728653


def _normalize(v: np.ndarray) -> np.ndarray:
    """Row-normalize ``(N, 3)`` (zero rows become +Z; no NaNs ever)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.where(n > 1e-9, v / np.maximum(n, 1e-9), _UP)
    return out.astype(np.float32)


def _frames(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthonormal frame ``(u, v)`` perpendicular to each axis — ``(N, 3)`` each.

    ``u = normalize(cross(axis, up))`` with an X-axis fallback for
    near-vertical axes; ``v = cross(axis, u)``.  Deterministic, vectorized.
    """
    ref = np.where(
        np.abs(axes[:, 2:3]) < 0.94, _UP[None, :], np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    u = _normalize(np.cross(axes, ref))
    v = _normalize(np.cross(axes, u))
    return u, v


def _rotate_toward_up(dirs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Rotate each unit direction toward +Z (positive *angles*) or away
    (negative — droop), clamped so directions never flip past vertical.

    Rodrigues rotation about ``cross(dir, up)``, fully vectorized.
    Directions already (anti)parallel to Z are returned unchanged.
    """
    dirs = dirs.astype(np.float32)
    axis = np.cross(dirs, _UP[None, :])
    axis_len = np.linalg.norm(axis, axis=1)
    ok = axis_len > 1e-6
    # Clamp: can rotate up at most to vertical, down at most to -vertical.
    to_up = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    ang = np.clip(angles, -(math.pi - to_up), to_up).astype(np.float32)
    ang = np.where(ok, ang, 0.0)
    a = np.where(ok[:, None], axis / np.maximum(axis_len[:, None], 1e-9), 0.0)
    c = np.cos(ang)[:, None]
    s = np.sin(ang)[:, None]
    dot = np.sum(a * dirs, axis=1, keepdims=True)
    rotated = dirs * c + np.cross(a, dirs) * s + a * dot * (1.0 - c)
    return _normalize(rotated)


class SkeletonBuilder:
    """
    Grow a branch skeleton level by level — the species-script "node graph".

    Each grow call returns the **segment ids it created** (``int32`` array);
    pass those ids back into :meth:`branches` to sprout the next level, and
    into ``leaves.leaves_at_tips`` to foliate the result.  Call
    :meth:`skeleton` once at the end to finalize.

    Every method consumes ``self.rng`` deterministically — same rng state in,
    byte-identical skeleton out.  All endpoints are constructed ON their
    parent segment, so canopies can never detach from trunks.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator (the species def receives it from the procedural
        registry — Hard Rule 2).

    Example
    -------
    ::

        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=5.0, base_radius_m=0.25, segments=4)
        limbs = sb.branches(trunk, count=(3, 5), length_ratio=(0.4, 0.7))
        twigs = sb.branches(limbs, count=(1, 2), length_ratio=(0.4, 0.6))
        sk = sb.skeleton()
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._parent: list[np.ndarray] = []
        self._start: list[np.ndarray] = []
        self._end: list[np.ndarray] = []
        self._r0: list[np.ndarray] = []
        self._r1: list[np.ndarray] = []
        self._depth: list[np.ndarray] = []
        self._chain_len: list[np.ndarray] = []  # full length of owning chain
        self._count: int = 0
        self._max_z: float = 1e-6

    # -- internal accessors (concatenated views; segment counts are tiny) ----

    def _all(self, parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts, axis=0) if parts else np.empty((0,))

    def _append(
        self,
        parent: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        r0: np.ndarray,
        r1: np.ndarray,
        depth: np.ndarray,
        chain_len: np.ndarray,
    ) -> np.ndarray:
        n = int(start.shape[0])
        ids = np.arange(self._count, self._count + n, dtype=np.int32)
        self._parent.append(np.asarray(parent, dtype=np.int32))
        self._start.append(np.asarray(start, dtype=np.float32))
        self._end.append(np.asarray(end, dtype=np.float32))
        self._r0.append(np.asarray(r0, dtype=np.float32))
        self._r1.append(np.asarray(r1, dtype=np.float32))
        self._depth.append(np.asarray(depth, dtype=np.int32))
        self._chain_len.append(np.asarray(chain_len, dtype=np.float32))
        self._count += n
        if n:
            self._max_z = max(self._max_z, float(end[:, 2].max()))
        return ids

    # ------------------------------------------------------------------
    # Growth API (what species scripts call)
    # ------------------------------------------------------------------

    def trunk(
        self,
        *,
        height_m: float,
        base_radius_m: float,
        tip_radius_m: float | None = None,
        segments: int = 3,
        wobble_m: float = 0.1,
        lean_rad: float = 0.0,
    ) -> np.ndarray:
        """
        Grow the trunk: ``segments`` stacked tapering segments from the origin.

        A **bush** is the same call with a stubby trunk (``height_m ≈ 0.15``,
        ``segments=1``) — one generator, both plants.

        Parameters
        ----------
        height_m : float
            Total trunk height (m).
        base_radius_m : float
            Half-thickness at the ground (m) — the rendered trunk is a square
            prism of side ``2 × radius``.
        tip_radius_m : float | None
            Half-thickness at the crown; defaults to ``base_radius_m * 0.25``.
        segments : int
            Stacked segments (more = wigglier trunk).  Default 3.
        wobble_m : float
            Max horizontal deviation of intermediate nodes (m), growing with
            height so the base stays pinned at the origin.  Default 0.1.
        lean_rad : float
            Overall lean angle from vertical (radians) in a random yaw
            direction.  Default 0 (plumb).

        Returns
        -------
        numpy.ndarray
            ``int32`` ids of the trunk segments (root first) — pass to
            :meth:`branches`.
        """
        n = max(1, int(segments))
        tip_r = base_radius_m * 0.25 if tip_radius_m is None else tip_radius_m

        z = np.linspace(0.0, float(height_m), n + 1, dtype=np.float32)
        t = z / max(float(height_m), 1e-6)
        lean_yaw = float(self.rng.uniform(0.0, _TWO_PI))
        lean_xy = np.array([math.cos(lean_yaw), math.sin(lean_yaw)], dtype=np.float32) * math.tan(
            float(lean_rad)
        )
        wob = self.rng.uniform(-wobble_m, wobble_m, size=(n + 1, 2)).astype(np.float32) * t[:, None]
        nodes = np.empty((n + 1, 3), dtype=np.float32)
        nodes[:, 0:2] = wob + lean_xy[None, :] * z[:, None]
        nodes[0, 0:2] = 0.0  # base exactly at the origin
        nodes[:, 2] = z

        radii = (float(base_radius_m) + (tip_r - float(base_radius_m)) * t).astype(np.float32)
        # Chain linkage: segment k's parent is segment k-1; the base is a root.
        parent = np.arange(-1 + self._count, n - 1 + self._count, dtype=np.int32)
        parent[0] = -1
        return self._append(
            parent=parent,
            start=nodes[:-1],
            end=nodes[1:],
            r0=radii[:-1],
            r1=radii[1:],
            depth=np.zeros(n, dtype=np.int32),
            chain_len=np.full(n, float(height_m), dtype=np.float32),
        )

    def branches(
        self,
        parents: np.ndarray,
        *,
        count: tuple[int, int] = (2, 4),
        t_range: tuple[float, float] = (0.45, 0.95),
        pitch_set: tuple[float, ...] = (math.pi / 2.0,),
        pitch_jitter_rad: float = 0.12,
        yaw_mode: str = "spiral",
        yaw_jitter_rad: float = 0.3,
        length_ratio: tuple[float, float] = (0.45, 0.7),
        length_m: tuple[float, float] | None = None,
        length_scale_by_height: tuple[float, float] = (1.0, 1.0),
        radius_ratio: float = 0.55,
        min_radius_m: float = 0.02,
        upturn_rad: float = 0.0,
        droop_rad: float = 0.0,
        bend_rad: float = 0.15,
        segments: int = 1,
    ) -> np.ndarray:
        """
        Sprout child branches from points along the *parents* segments.

        This is the workhorse of the species API — each call is one growth
        level (trunk → limbs, limbs → twigs, …).  Branch start points are
        constructed ON the parent segment (never floating), directions come
        from a script-controlled angle set, and lengths/radii derive from
        the parent so the tree tapers naturally toward the crown.

        Parameters
        ----------
        parents : numpy.ndarray
            Segment ids to grow from (a previous call's return value).
        count : tuple[int, int]
            Branches per parent segment, drawn uniformly from
            ``[count[0], count[1]]`` inclusive.  Default (2, 4).
        t_range : tuple[float, float]
            Where along the parent segment branches attach, as a fraction
            ``0 = parent start … 1 = parent end``.  Default (0.45, 0.95).
        pitch_set : tuple[float, ...]
            The **angle choice set** (radians) between the branch and its
            parent's axis.  ``(math.radians(90),)`` gives the blocky
            *Dynamic Trees* right-angle look; mixed sets give freer growth.
            Each branch picks one entry uniformly.  Default (π/2,).
        pitch_jitter_rad : float
            Uniform jitter added to the picked pitch.  Default 0.12.
        yaw_mode : str
            How branches distribute around the parent axis:
            ``"spiral"`` (golden-angle phyllotaxis — natural, never stacks),
            ``"opposite"`` (alternating 180° pairs), ``"random"``.
        yaw_jitter_rad : float
            Uniform yaw jitter on top of the mode.  Default 0.3.
        length_ratio : tuple[float, float]
            Branch length as a fraction of the parent's **chain length**
            (a trunk segment's chain length is the full trunk height), drawn
            uniformly.  Ignored when *length_m* is given.  Default (0.45, 0.7).
        length_m : tuple[float, float] | None
            Absolute branch length range (m) — use for bushes whose stub
            trunk makes ratios meaningless.
        length_scale_by_height : tuple[float, float]
            Length multiplier lerped by the attachment point's normalized
            height (0 = ground, 1 = current tree top) — ``(1.0, 0.45)``
            makes upper limbs shorter so the canopy tapers like the mod's
            trees.  Default (1.0, 1.0) (no taper).
        radius_ratio : float
            Branch start radius as a fraction of the parent's radius at the
            attachment point.  Default 0.55.
        min_radius_m : float
            Radius floor (m) so twigs never vanish.  Default 0.02.
        upturn_rad / droop_rad : float
            Rotate branch directions toward (+) / away from (−) vertical by
            this angle (radians), clamped at vertical.  Old oaks droop,
            young growth turns up.  Defaults 0.
        bend_rad : float
            Per-sub-segment random direction deviation (radians) when
            ``segments > 1`` — crooked branches.  Default 0.15.
        segments : int
            Sub-segments per branch (more = curvier).  Default 1.

        Returns
        -------
        numpy.ndarray
            ``int32`` ids of ALL created sub-segments — feed to the next
            :meth:`branches` level or ``leaves_at_tips``.
        """
        parents = np.asarray(parents, dtype=np.int64)
        if parents.size == 0:
            return np.empty(0, dtype=np.int32)
        rng = self.rng
        all_start = self._all(self._start).reshape(-1, 3)
        all_end = self._all(self._end).reshape(-1, 3)
        all_r0 = self._all(self._r0)
        all_r1 = self._all(self._r1)
        all_depth = self._all(self._depth).astype(np.int32)
        all_chain = self._all(self._chain_len)

        counts = rng.integers(count[0], count[1] + 1, size=parents.size)
        B = int(counts.sum())
        if B == 0:
            return np.empty(0, dtype=np.int32)
        pidx = np.repeat(parents, counts)  # (B,)
        # Ordinal of each branch within its parent (for spiral / opposite).
        ordinal = np.arange(B) - np.repeat(np.cumsum(counts) - counts, counts)

        # --- attachment points ON the parent segment ----------------------
        t = rng.uniform(t_range[0], t_range[1], B).astype(np.float32)
        p_start, p_end = all_start[pidx], all_end[pidx]
        attach = p_start + (p_end - p_start) * t[:, None]
        p_axis = _normalize(p_end - p_start)
        r_attach = (all_r0[pidx] + (all_r1[pidx] - all_r0[pidx]) * t).astype(np.float32)

        # --- branch directions: pitch from the angle set, yaw by mode -----
        pitch = np.asarray(pitch_set, dtype=np.float64)[
            rng.integers(0, len(pitch_set), B)
        ] + rng.uniform(-pitch_jitter_rad, pitch_jitter_rad, B)
        if yaw_mode == "spiral":
            yaw0 = rng.uniform(0.0, _TWO_PI, parents.size)
            yaw = np.repeat(yaw0, counts) + ordinal * _GOLDEN_ANGLE
        elif yaw_mode == "opposite":
            yaw0 = rng.uniform(0.0, _TWO_PI, parents.size)
            yaw = np.repeat(yaw0, counts) + ordinal * math.pi
        elif yaw_mode == "random":
            yaw = rng.uniform(0.0, _TWO_PI, B)
        else:
            raise ValueError(f"yaw_mode must be 'spiral', 'opposite' or 'random', got {yaw_mode!r}")
        yaw = yaw + rng.uniform(-yaw_jitter_rad, yaw_jitter_rad, B)

        u, v = _frames(p_axis)
        sp, cp = np.sin(pitch)[:, None], np.cos(pitch)[:, None]
        sy, cy = np.sin(yaw)[:, None], np.cos(yaw)[:, None]
        dirs = _normalize(p_axis * cp + (u * cy + v * sy) * sp)
        if upturn_rad or droop_rad:
            dirs = _rotate_toward_up(dirs, np.full(B, float(upturn_rad) - float(droop_rad)))

        # --- lengths -------------------------------------------------------
        if length_m is not None:
            length = rng.uniform(length_m[0], length_m[1], B)
        else:
            length = rng.uniform(length_ratio[0], length_ratio[1], B) * all_chain[pidx]
        h_norm = np.clip(attach[:, 2] / self._max_z, 0.0, 1.0)
        ls0, ls1 = length_scale_by_height
        length = (length * (ls0 + (ls1 - ls0) * h_norm)).astype(np.float32)

        # --- radii ----------------------------------------------------------
        r_start = np.minimum(
            np.maximum(r_attach * float(radius_ratio), min_radius_m), r_attach
        ).astype(np.float32)
        r_tip = np.maximum(r_start * 0.3, min_radius_m * 0.75).astype(np.float32)

        # --- emit sub-segments (fixed small loop over `segments`) ----------
        n_sub = max(1, int(segments))
        seg_len = (length / n_sub)[:, None]
        depth = (all_depth[pidx] + 1).astype(np.int32)
        ids_out: list[np.ndarray] = []
        cur_start = attach
        cur_dir = dirs
        prev_ids: np.ndarray | None = None
        for k in range(n_sub):
            if k > 0 and bend_rad > 0.0:
                dev = rng.uniform(-1.0, 1.0, (B, 3)).astype(np.float32)
                cur_dir = _normalize(cur_dir + dev * math.tan(float(bend_rad)))
            cur_end = cur_start + cur_dir * seg_len
            f0 = k / n_sub
            f1 = (k + 1) / n_sub
            seg_r0 = r_start + (r_tip - r_start) * f0
            seg_r1 = r_start + (r_tip - r_start) * f1
            par = pidx.astype(np.int32) if prev_ids is None else prev_ids
            ids = self._append(
                parent=par,
                start=cur_start,
                end=cur_end,
                r0=seg_r0,
                r1=seg_r1,
                depth=depth,
                chain_len=length,
            )
            ids_out.append(ids)
            prev_ids = ids
            cur_start = cur_end
        return np.concatenate(ids_out)

    def skeleton(self) -> TreeSkeleton:
        """
        Finalize the build: concatenate arrays and compute sway weights.

        Sway at a segment's end = its cumulative path length from the root
        divided by the longest path in the tree, clamped to ``[0, 1]`` — so
        the trunk base is pinned (≈0) and the outermost tips ride the wind
        (≈1).  Monotone non-decreasing along every branch path by
        construction.

        Returns
        -------
        TreeSkeleton
        """
        if self._count == 0:
            raise ValueError("SkeletonBuilder: nothing grown — call trunk() before skeleton()")
        parent = self._all(self._parent).astype(np.int32)
        start = self._all(self._start).reshape(-1, 3).astype(np.float32)
        end = self._all(self._end).reshape(-1, 3).astype(np.float32)
        r0 = self._all(self._r0).astype(np.float32)
        r1 = self._all(self._r1).astype(np.float32)
        depth = self._all(self._depth).astype(np.int32)

        # Path length from root to each segment end.  Children always have
        # larger ids than their parents (segments append in growth order),
        # so one ordered pass resolves every chain.  S is tens — fine.
        seg_len = np.linalg.norm(end - start, axis=1)
        path = np.zeros(self._count, dtype=np.float64)
        for i in range(self._count):  # tens of segments, bounded
            p = parent[i]
            path[i] = seg_len[i] + (path[p] if p >= 0 else 0.0)
        sway = np.clip(path / max(path.max(), 1e-6), 0.0, 1.0).astype(np.float32)

        return TreeSkeleton(
            parent=parent,
            start=start,
            end=end,
            radius_start=r0,
            radius_end=r1,
            depth=depth,
            sway=sway,
        )
