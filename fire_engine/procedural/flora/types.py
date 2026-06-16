"""
procedural/flora/types.py — shared data types for the flora pipeline.

Grouping module (exempt from the one-public-class rule) holding the trivial
struct/dataclass support types used across the flora sub-package:

* :class:`TreeSkeleton` — finalized branch-skeleton struct-of-arrays produced
  by :class:`~fire_engine.procedural.flora.skeleton.SkeletonBuilder`.
* :func:`validate_skeleton` — machine-check of skeleton invariants.
* :class:`TreeVariantSet` — registry-cached per-species mesh + texture bundle
  produced by :class:`~fire_engine.procedural.flora.species_def.TreeSpeciesDef`.

These types are re-exported from their original modules (``skeleton.py``,
``species_def.py``) so all existing import paths remain valid.

Docs: docs/systems/procedural.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fire_engine.procedural.flora.mesher import TreeMesh

__all__ = ["TreeSkeleton", "TreeVariantSet", "validate_skeleton"]


@dataclass
class TreeSkeleton:
    """
    A finalized branch skeleton — struct-of-arrays, one row per segment.

    Produced by :meth:`~fire_engine.procedural.flora.skeleton.SkeletonBuilder.skeleton`;
    consumed by ``mesher.mesh_branches``, ``leaves.leaves_at_tips`` and
    ``impostor.rasterize_impostor``.

    Attributes
    ----------
    parent : numpy.ndarray
        ``int32 (S,)`` — parent segment id, ``-1`` for root (trunk base)
        segments.  A child's ``start`` always lies ON its parent segment.
    start, end : numpy.ndarray
        ``float32 (S, 3)`` — segment endpoints in tree-local meters
        (trunk base at the origin, Z-up).
    radius_start, radius_end : numpy.ndarray
        ``float32 (S,)`` — half-thickness (m) at each endpoint.  The mesher
        renders a square cross-section of side ``2 × radius``.  Tapers
        root → tip (``radius_end <= radius_start`` per segment).
    depth : numpy.ndarray
        ``int32 (S,)`` — growth level: 0 = trunk, 1 = limbs, 2 = twigs …
        Sub-segments within one branch share their branch's depth.
    sway : numpy.ndarray
        ``float32 (S,)`` — wind-sway weight at the segment's END point,
        in ``[0, 1]``: 0 at the trunk base rising with path length to ≈1 at
        the outermost tips.  Baked per-vertex into mesh ``color.a`` so the
        vertex shader can bend canopies while pinning trunks.

    Docs: docs/systems/procedural.md
    """

    parent: np.ndarray
    start: np.ndarray
    end: np.ndarray
    radius_start: np.ndarray
    radius_end: np.ndarray
    depth: np.ndarray
    sway: np.ndarray

    @property
    def n_segments(self) -> int:
        """Number of segments ``S``."""
        return int(self.parent.shape[0])

    def sway_start(self) -> np.ndarray:
        """
        Sway weight at each segment's START point — ``float32 (S,)``.

        A segment's start sway is its parent's end sway (0 for roots), so
        sway is continuous along every branch path.
        """
        s = np.where(self.parent >= 0, self.sway[np.maximum(self.parent, 0)], 0.0)
        return s.astype(np.float32)

    def tip_ids(self, ids: np.ndarray | None = None) -> np.ndarray:
        """
        Segment ids that are tips — segments no other segment grows from.

        Parameters
        ----------
        ids : numpy.ndarray | None
            Restrict the answer to this id subset (e.g. the ids one
            ``branches()`` call returned).  ``None`` → all tips.
        """
        is_parent = np.zeros(self.n_segments, dtype=bool)
        is_parent[self.parent[self.parent >= 0]] = True
        tips = np.nonzero(~is_parent)[0].astype(np.int32)
        if ids is not None:
            tips = tips[np.isin(tips, np.asarray(ids))]
        return tips


def validate_skeleton(sk: TreeSkeleton, atol: float = 1e-3) -> None:
    """
    Machine-check skeleton invariants; raise ``ValueError`` on violation.

    Called by ``TreeSpeciesDef.generate`` on every variant (cheap — tens of
    segments) and by the test suite.  Catches the whole "floating canopy"
    class of bugs that 2-D sprite generation could only eyeball.

    Checks
    ------
    1. Every non-root segment's ``start`` lies ON its parent segment
       (distance to the parent's line segment < *atol* meters).
    2. Radii taper: ``radius_end <= radius_start`` per segment, and a
       child's ``radius_start`` never exceeds its parent's thickest point.
    3. ``sway`` is in ``[0, 1]`` and non-decreasing along every parent
       link (canopy sways at least as much as the wood it grows from).

    Parameters
    ----------
    sk : TreeSkeleton
        The finalized skeleton.
    atol : float
        Attachment tolerance (m).  Default 1e-3.

    Docs: docs/systems/procedural.md
    """
    child = np.nonzero(sk.parent >= 0)[0]
    if child.size:
        p = sk.parent[child]
        a = sk.start[p]
        b = sk.end[p]
        ab = b - a
        denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
        t = np.clip(np.sum((sk.start[child] - a) * ab, axis=1) / denom, 0.0, 1.0)
        nearest = a + ab * t[:, None]
        d = np.linalg.norm(sk.start[child] - nearest, axis=1)
        if (d > atol).any():
            worst = int(child[int(np.argmax(d))])
            raise ValueError(
                f"validate_skeleton: segment {worst} starts {d.max():.4f} m "
                f"off its parent segment (floating branch)"
            )

        if (sk.radius_start[child] > np.maximum(sk.radius_start[p], sk.radius_end[p]) + 1e-5).any():
            raise ValueError("validate_skeleton: a child branch is thicker than its parent")
        if (sk.sway[child] + 1e-6 < sk.sway[p]).any():
            raise ValueError("validate_skeleton: sway decreases along a branch path")

    if (sk.radius_end > sk.radius_start + 1e-5).any():
        raise ValueError("validate_skeleton: a segment's radius grows toward its tip")
    if (sk.sway < -1e-6).any() or (sk.sway > 1.0 + 1e-6).any():
        raise ValueError("validate_skeleton: sway weights outside [0, 1]")


@dataclass(frozen=True)
class TreeVariantSet:
    """
    Everything the renderer needs to draw one species — registry-cached.

    Produced by :meth:`~fire_engine.procedural.flora.species_def.TreeSpeciesDef.generate`
    and stored in the procedural registry keyed by species name.

    Attributes
    ----------
    name : str
        The species def name (``"tree_gnarled_oak"`` …).
    meshes : tuple[TreeMesh, ...]
        The variant pool — unique per world seed; instances draw from it.
    atlas : numpy.ndarray
        ``(H, W, 4) uint8`` species texture (bark left half opaque, leaf
        right half binary alpha) — bind once per species.
    impostors : numpy.ndarray
        ``(Hc, Wc × n_variants, 4) uint8`` far-LOD sprite atlas, one cell
        per variant, trunk bases on the bottom row.  All cells share ONE
        meters-per-texel scale (see ``impostor_height_m``).
    max_height_m / max_radius_m : float
        Pool-wide extents (m) — pad render bounds with these × max scale.
    impostor_width_m / impostor_height_m : float
        World size (m) of the impostor billboard quad: because the cells
        share one scale, a quad of exactly this size, base at the trunk
        base, overlays every variant's mesh at the crossfade.

    Docs: docs/systems/procedural.md
    """

    name: str
    meshes: tuple[TreeMesh, ...]
    atlas: np.ndarray
    impostors: np.ndarray
    max_height_m: float
    max_radius_m: float
    impostor_width_m: float
    impostor_height_m: float

    @property
    def n_variants(self) -> int:
        """Variant pool size."""
        return len(self.meshes)
