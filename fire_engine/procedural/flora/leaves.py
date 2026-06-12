"""
procedural/flora/leaves.py — leaf-cluster placement on branch skeletons.

Leaves follow the *Dynamic Trees* idea: foliage is placed **on branch tips**,
so the canopy shape emerges from the skeleton and can never float free of
the wood.  A :class:`LeafClusters` is a struct-of-arrays of foliage blobs
(center, radius, sway weight) that the mesher turns into crossed alpha-cutout
quads merged into the tree mesh, and the impostor rasterizer turns into
posterised discs.

Species scripts normally only call :func:`leaf_clusters_at_tips`; leafless
or near-leafless species (dead trees) pass fewer ids or use
``per_tip=(0, 1)`` for sparse tufts.

Units: meters, Z-up, tree-local space (trunk base at the origin).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.procedural.flora.skeleton import TreeSkeleton

__all__ = ["LeafClusters", "leaf_clusters_at_tips"]


@dataclass
class LeafClusters:
    """
    Foliage blobs attached to a skeleton — struct-of-arrays, one row each.

    Attributes
    ----------
    center : numpy.ndarray
        ``float32 (L, 3)`` — blob centers, tree-local meters.
    radius : numpy.ndarray
        ``float32 (L,)`` — blob radius (m); the mesher's crossed quads are
        ``2 × radius`` wide/tall.
    sway : numpy.ndarray
        ``float32 (L,)`` — wind-sway weight in ``[0, 1]`` (≈0.85–1.0:
        leaves ride gusts harder than the wood they grow on).
    """

    center: np.ndarray
    radius: np.ndarray
    sway: np.ndarray

    @property
    def n_clusters(self) -> int:
        """Number of clusters ``L``."""
        return int(self.radius.shape[0])

    @staticmethod
    def empty() -> "LeafClusters":
        """A zero-cluster instance (leafless species)."""
        return LeafClusters(center=np.empty((0, 3), dtype=np.float32),
                            radius=np.empty(0, dtype=np.float32),
                            sway=np.empty(0, dtype=np.float32))


def leaf_clusters_at_tips(
    sk: TreeSkeleton,
    ids: np.ndarray,
    rng: np.random.Generator,
    *,
    radius_m: tuple[float, float] = (0.5, 1.0),
    per_tip: tuple[int, int] = (1, 2),
    offset_frac: float = 0.35,
    sway_min: float = 0.85,
) -> LeafClusters:
    """
    Place foliage blobs on the **tip segments** among *ids*.

    Tips are segments nothing else grows from (``TreeSkeleton.tip_ids``), so
    foliating after the last ``branches()`` call puts leaves exactly where
    the youngest wood is — the canopy always connects to the tree.

    Parameters
    ----------
    sk : TreeSkeleton
        The finalized skeleton (``SkeletonBuilder.skeleton()``).
    ids : numpy.ndarray
        Candidate segment ids — typically the concatenated returns of the
        outer ``branches()`` calls.  Non-tips among them are ignored.
    rng : numpy.random.Generator
        Deterministic generator (consume the species def's rng).
    radius_m : tuple[float, float]
        Per-blob radius range (m), uniform.  Default (0.5, 1.0).
    per_tip : tuple[int, int]
        Blobs per tip, uniform inclusive.  ``(0, 1)`` gives the sparse dying
        look.  Default (1, 2).
    offset_frac : float
        Blob centers jitter around the tip end point by up to
        ``radius × offset_frac`` per axis (slight upward bias so canopies
        crown the wood).  Default 0.35.
    sway_min : float
        Floor for the blob sway weight; actual sway is
        ``max(tip sway, uniform(sway_min, 1))``.  Default 0.85.

    Returns
    -------
    LeafClusters
        May be empty (e.g. ``per_tip=(0, 0)`` or no tips in *ids*).

    Example
    -------
    ::

        leaves = leaf_clusters_at_tips(sk, np.concatenate([limbs, twigs]),
                                       rng, radius_m=(0.6, 1.1))
    """
    tips = sk.tip_ids(ids)
    if tips.size == 0:
        return LeafClusters.empty()

    counts = rng.integers(per_tip[0], per_tip[1] + 1, size=tips.size)
    n = int(counts.sum())
    if n == 0:
        return LeafClusters.empty()
    rep = np.repeat(tips, counts)

    radius = rng.uniform(radius_m[0], radius_m[1], n).astype(np.float32)
    jitter = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    jitter[:, 2] = np.abs(jitter[:, 2]) * 0.6 + 0.1   # bias blobs upward
    center = (sk.end[rep]
              + jitter * (radius * float(offset_frac))[:, None]) \
        .astype(np.float32)

    sway = np.maximum(sk.sway[rep],
                      rng.uniform(sway_min, 1.0, n)).astype(np.float32)
    return LeafClusters(center=center, radius=radius,
                        sway=np.clip(sway, 0.0, 1.0))
