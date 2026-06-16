"""
procedural/flora/leaves.py — along-wood leaf placement on skeletons.

Leaves follow the *Dynamic Trees* idea, but the canopy is **hundreds of
individual leaves anchored ALONG the branch wood** rather than a volumetric
blob floating around the tips.  Every leaf is grown on a leaf-bearing
segment (the finest twigs first): a parameter ``t`` is drawn along the
segment axis, the point on the axis is computed, and the leaf is pushed
RADIALLY off the wood by a small bounded amount (≈ the segment radius at
``t`` plus a fraction of a leaf size).  A leaf therefore always sits just
off the surface of a real branch — it can never float in empty space.

Because the count scales with the branch structure (more twigs ⇒ more
leaves) and biases toward the thinner / outer segments and toward segment
ends, adding more or finer twigs in a species script directly produces a
denser, leafier silhouette that hugs the wood.

The result is a :class:`Leaves` struct-of-arrays — one row per **individual
leaf** — that the mesher turns into small oriented alpha-cutout quads merged
into the variant mesh (one draw per variant; the GPU batches the whole
canopy), and the impostor rasterizer turns into posterised dots.

Species scripts normally only call :func:`leaves_at_tips`; leafless or
near-leafless species (dead trees) pass fewer ids / lower ``density``, or
return ``Leaves.empty()``.

Units: meters, Z-up, tree-local space (trunk base at the origin).

Docs: docs/systems/procedural.flora.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.procedural.flora.skeleton import TreeSkeleton

__all__ = ["Leaves", "leaves_at_tips"]

# Practical ceiling on the leaf count we will materialize before thinning, so
# a runaway density × length never blows memory.  Species may request up to a
# few thousand leaves; this guards only against absurd inputs.
_MAX_RAW_LEAVES = 200_000


@dataclass
class Leaves:
    """
    Individual leaves attached to a skeleton — struct-of-arrays, one row
    per leaf card.

    Attributes
    ----------
    center : numpy.ndarray
        ``float32 (L, 3)`` — leaf centers, tree-local meters.  Every center
        sits just off the surface of a real branch segment (see
        :func:`leaves_at_tips`).
    radius : numpy.ndarray
        ``float32 (L,)`` — leaf half-size (m); the mesher's card is
        ``2 × radius`` across.
    sway : numpy.ndarray
        ``float32 (L,)`` — wind-sway weight in ``[0, 1]`` (≈0.85–1.0:
        leaves ride gusts harder than the wood they grow on).

    Docs: docs/systems/procedural.flora.md
    """

    center: np.ndarray
    radius: np.ndarray
    sway: np.ndarray

    @property
    def n_leaves(self) -> int:
        """Number of leaves ``L``.

        Docs: docs/systems/procedural.flora.md
        """
        return int(self.radius.shape[0])

    @staticmethod
    def empty() -> Leaves:
        """A zero-leaf instance (leafless species).

        Docs: docs/systems/procedural.flora.md
        """
        return Leaves(
            center=np.empty((0, 3), dtype=np.float32),
            radius=np.empty(0, dtype=np.float32),
            sway=np.empty(0, dtype=np.float32),
        )


def _perp_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthonormal frame ``(u, v)`` perpendicular to each unit *axis* — ``(N, 3)``.

    ``u = normalize(cross(axis, ref))`` with ``ref`` switched from +Z to +X
    for near-vertical axes (so the cross never degenerates); ``v =
    cross(axis, u)``.  Deterministic and vectorized.
    """
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ref = np.where(
        np.abs(axis[:, 2:3]) < 0.94, up[None, :], np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    u = np.cross(axis, ref)
    un = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.where(un > 1e-9, u / np.maximum(un, 1e-9), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    v = np.cross(axis, u)
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.where(vn > 1e-9, v / np.maximum(vn, 1e-9), np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
    return u.astype(np.float32), v.astype(np.float32)


def leaves_at_tips(
    sk: TreeSkeleton,
    ids: np.ndarray,
    rng: np.random.Generator,
    *,
    cell_m: float = 0.25,  # kept for call-site compatibility (now unused)
    rounds: int = 3,  # legacy CA knob — now only gates "no foliage" (rounds<=0)
    density: float = 0.6,
    per_cell: tuple[int, int] = (1, 2),  # kept for compatibility (now unused)
    leaf_size_m: tuple[float, float] = (0.09, 0.14),
    sway_min: float = 0.85,
    max_leaves: int = 600,
    leaves_per_m: float | None = None,
    max_offset_m: float | None = None,
) -> Leaves:
    """
    Grow individual leaves **along the wood** of the leaf-bearing segments
    in *ids* (twigs and outer limbs), anchoring every leaf to a real branch.

    For each leaf a host segment is chosen (biased toward thinner / outer
    twigs), a parameter ``t`` is drawn along it (biased toward the segment
    end so the silhouette stays leafy at the rim), the point on the segment
    axis at ``t`` is computed, and the leaf center is offset RADIALLY off
    the wood by ``segment_radius(t) + a small fraction of a leaf size`` in a
    deterministic pseudo-random perpendicular direction.  The offset is
    bounded (``max_offset_m``) so leaves hug the branch — they never float.

    Leaf *count* scales with the branch structure: it is
    ``density × leaves_per_m × Σ segment_length`` over the leaf-bearing
    segments (so more / finer twigs ⇒ denser canopy), then capped at
    ``max_leaves``.

    Parameters
    ----------
    sk : TreeSkeleton
        The finalized skeleton (``SkeletonBuilder.skeleton()``).
    ids : numpy.ndarray
        Leaf-bearing segment ids — typically the concatenated returns of the
        outer ``branches()`` calls (limbs + twigs).  Every segment gets
        leaves, but the per-segment share is weighted toward the thinner
        twigs (∝ length ÷ radius) so foliage clusters on the fine growth.
        Empty / no segments ⇒ ``Leaves.empty()``.
    rng : numpy.random.Generator
        Deterministic generator (consume the species def's rng).
    cell_m : float
        Deprecated CA cell size — accepted for call-site compatibility and
        ignored.
    rounds : int
        Deprecated CA hydration radius — accepted for compatibility; only
        ``rounds <= 0`` still matters (it means "no foliage" ⇒ empty).
    density : float
        Overall leaf-count multiplier (``0 ⇒`` empty).  Default 0.6.
    per_cell : tuple[int, int]
        Deprecated CA knob — accepted for compatibility and ignored.
    leaf_size_m : tuple[float, float]
        Per-leaf half-size range (m), uniform.  Default (0.09, 0.14).
    sway_min : float
        Floor for the leaf sway weight; actual sway is
        ``max(host-segment sway at t, uniform(sway_min, 1))``.  Default 0.85.
    max_leaves : int
        Deterministic thinning cap (vertex budget: 4 verts/leaf).  May be a
        few thousand.  Default 600.
    leaves_per_m : float | None
        Target leaves per meter of branch length (before the ``density``
        multiplier).  ``None`` → a default of 60/m, tuned so the default oak
        call yields a full canopy.  Raise it for denser foliage.
    max_offset_m : float | None
        Hard cap on how far a leaf center sits off the wood axis (m).
        ``None`` → ``segment_radius(t) + 1.5 × max(leaf_size_m)`` per leaf,
        which keeps leaves visually attached.  This is the anti-floating
        guarantee: every leaf is within ``max_offset_m`` of its host segment.

    Returns
    -------
    Leaves
        May be empty (no segments in *ids*, ``rounds<=0``, ``density<=0``).

    Example
    -------
    ::

        leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng,
                                density=0.85, leaf_size_m=(0.12, 0.18),
                                max_leaves=1600)

    Docs: docs/systems/procedural.flora.md
    """
    del cell_m, per_cell  # deprecated CA knobs — kept only for call sites
    seg = np.asarray(ids, dtype=np.int64).ravel()
    seg = np.unique(seg)  # de-dup limbs/twigs overlap; keeps determinism
    if seg.size == 0 or rounds <= 0 or density <= 0.0:
        return Leaves.empty()

    start = sk.start[seg].astype(np.float64)  # (M, 3)
    end = sk.end[seg].astype(np.float64)
    axis_vec = end - start
    length = np.linalg.norm(axis_vec, axis=1)  # (M,)
    live = length > 1e-6
    if not live.any():
        return Leaves.empty()
    seg, start, end, axis_vec, length = (
        seg[live],
        start[live],
        end[live],
        axis_vec[live],
        length[live],
    )

    r0 = sk.radius_start[seg].astype(np.float64)  # (M,)
    r1 = sk.radius_end[seg].astype(np.float64)
    sway0 = sk.sway_start()[seg].astype(np.float64)
    sway1 = sk.sway[seg].astype(np.float64)
    radius_mid = 0.5 * (r0 + r1)

    # --- bias toward the thinner / outer segments (twigs) ------------------
    # Weight each segment's share of the leaf budget by its length AND by how
    # thin it is (inverse radius), so finer twigs get proportionally more
    # leaves and the silhouette reads as foliage, not bare limbs.
    thin_w = 1.0 / (radius_mid + 0.02)  # +2 cm so trunk-ish segs aren't zero
    seg_weight = length * thin_w
    seg_weight = seg_weight / seg_weight.sum()

    # --- total leaf budget scales with branch length ----------------------
    per_m = 60.0 if leaves_per_m is None else float(leaves_per_m)
    total_len = float(length.sum())
    raw = round(density * per_m * total_len)
    raw = max(0, min(raw, max_leaves, _MAX_RAW_LEAVES))
    if raw == 0:
        return Leaves.empty()

    # --- assign each leaf to a host segment (weighted, deterministic) ------
    host = rng.choice(seg.size, size=raw, p=seg_weight)  # indices into seg arrays

    # --- parameter t along the host, biased toward the segment END --------
    # sqrt(U) skews t toward 1.0 → leaves cluster toward the twig tip / rim.
    t = np.sqrt(rng.random(raw))

    a = start[host]
    ab = axis_vec[host]
    axis_hat = ab / length[host][:, None]
    p_axis = a + ab * t[:, None]  # point ON the wood

    # radius at t and host-interpolated sway
    r_t = r0[host] + (r1[host] - r0[host]) * t
    sway_t = sway0[host] + (sway1[host] - sway0[host]) * t

    # --- radial offset off the wood, bounded so leaves stay attached ------
    u, v = _perp_frame(axis_hat)
    phi = rng.uniform(0.0, 2.0 * np.pi, raw)
    radial = u * np.cos(phi)[:, None] + v * np.sin(phi)[:, None]

    leaf_r = rng.uniform(leaf_size_m[0], leaf_size_m[1], raw)
    big_leaf = float(max(leaf_size_m))
    # Nominal: sit on the bark surface plus ~0.6 leaf so the card peeks out.
    offset = r_t + 0.6 * leaf_r + rng.uniform(0.0, 0.4 * big_leaf, raw)
    cap = (r_t + 1.5 * big_leaf) if max_offset_m is None else float(max_offset_m)
    offset = np.minimum(offset, cap)

    centers = (p_axis + radial * offset[:, None]).astype(np.float32)

    sway = np.maximum(sway_t, rng.uniform(sway_min, 1.0, raw)).astype(np.float32)
    radius = leaf_r.astype(np.float32)

    if centers.shape[0] > max_leaves:  # deterministic unbiased thinning
        pick = np.sort(rng.permutation(centers.shape[0])[:max_leaves])
        centers, radius, sway = centers[pick], radius[pick], sway[pick]

    return Leaves(
        center=np.ascontiguousarray(centers, np.float32),
        radius=np.ascontiguousarray(radius, np.float32),
        sway=np.clip(sway, 0.0, 1.0),
    )
