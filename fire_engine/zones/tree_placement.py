"""
zones/tree_placement.py — CPU-baked placement for 3-D trees and bushes.

Unlike grass/flowers (whose instances are derived ON the GPU from
``gl_InstanceID`` because the counts are huge), trees and bushes are placed
**on the CPU**: counts are small (≤ a few thousand per volume), the game
needs knowable trunk positions (future collision/forage), trees must not
overlap, and each instance must pick a species + mesh variant for the
per-variant instanced draws.  The renderer
(``world/tree_renderer.py``) packs the result of :func:`bake_tree_instances`
into a small RGBA32F **data texture** the vertex shader reads with
``texelFetch(u_inst_tex, ivec2(col, gl_InstanceID), 0)``.

Placement is a **jittered grid**: one candidate per cell of edge
``c = max(min_spacing_m, 1/√density)``, jittered by ±0.35·c and kept with
probability ``density·c²`` — guaranteeing any two trees are at least
``0.3·c`` apart (no twin trunks) while reading as natural scatter.  Z comes
from the SAME height-field bake grass uses; sentinel texels (no ground)
drop the instance, so trees never float over craters.

Determinism: all randomness from ``for_domain("zones", "tree_placement",
kind, volume.id)`` (Hard Rule 2) — same world seed + volume → identical
forest, zero save bytes.

Data-texture layout (the GLSL contract — keep in sync with tree.vert /
tree_impostor.vert)
--------------------------------------------------------------------------
``instances_data_block`` returns ``(N, 2, 4) float32``; texel ``(0, i)`` =
``(x, y, z, yaw)`` and texel ``(1, i)`` = ``(scale, phase, tint, variant)``.
``tests/test_tree_placement.py`` pins this layout.

Units: meters, radians, Z-up.

Example
-------
::

    from fire_engine.zones import tree_placement as tp

    mix = tp.species_mix_from_params(vol.params, "tree_gnarled_oak")
    inst = tp.bake_tree_instances(vol, cfg, chunk_manager.chunks, mix,
                                  {"tree_gnarled_oak": 8}, kind="trees")
    block = tp.instances_data_block(inst, inst.species_idx == 0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from fire_engine.core import Config
from fire_engine.core.rng import for_domain
from fire_engine.zones.grass_placement import (
    HEIGHT_SENTINEL,
    bake_grass_height_field,
)
from fire_engine.zones.volume import ZoneVolume

__all__ = [
    "TREE_KINDS",
    "SCALE_JITTER",
    "TreeInstances",
    "species_mix_from_params",
    "bake_tree_instances",
    "instances_data_block",
]

# The zone tags the tree renderer consumes.  A "trees" volume ALSO feeds the
# wind system's leaf litter (zones/grass_placement.leaf_hash_seed) — one
# volume, trees + their fallen leaves.
TREE_KINDS: tuple[str, ...] = ("trees", "bushes")

# Per-instance scale jitter [min, min + span) — trees jitter wider than
# bushes.  PUBLIC: the renderer pads its culling bounds by max scale.
SCALE_JITTER: dict[str, tuple[float, float]] = {
    "trees": (0.8, 0.8),
    "bushes": (0.7, 0.6),
}

# Per-kind config field prefix (tree_density_per_m2, bush_min_spacing_m, …).
_KIND_PREFIX: dict[str, str] = {"trees": "tree", "bushes": "bush"}

# Grid jitter as a fraction of the cell edge.  0.35 leaves a guaranteed
# (1 − 2·0.35) = 0.3·cell minimum distance between any two instances.
_JITTER_FRAC = 0.35

# Per-instance albedo tint range (multiplies the atlas in the shader).
_TINT_MIN, _TINT_SPAN = 0.85, 0.30


@dataclass
class TreeInstances:
    """
    CPU-baked per-instance placement for one volume — struct-of-arrays.

    Attributes
    ----------
    x, y, z : numpy.ndarray
        ``float32 (N,)`` trunk-base world positions (m).  Z sits on the
        baked terrain surface.
    yaw : numpy.ndarray
        ``float32 (N,)`` rotation around Z, radians ``[0, 2π)``.
    scale : numpy.ndarray
        ``float32 (N,)`` per-instance size multiplier.
    phase : numpy.ndarray
        ``float32 (N,)`` wind sway phase, radians ``[0, 2π)``.
    tint : numpy.ndarray
        ``float32 (N,)`` albedo multiplier ``[0.85, 1.15)``.
    species_idx : numpy.ndarray
        ``int32 (N,)`` index into :attr:`species_names`.
    variant : numpy.ndarray
        ``int32 (N,)`` mesh-pool variant index within the species.
    species_names : tuple[str, ...]
        The species mix this bake drew from, in index order.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    scale: np.ndarray
    phase: np.ndarray
    tint: np.ndarray
    species_idx: np.ndarray
    variant: np.ndarray
    species_names: tuple[str, ...]

    @property
    def count(self) -> int:
        """Instance count ``N``."""
        return int(self.x.shape[0])


def species_mix_from_params(params: Mapping, default: str) -> list[tuple[str, float]]:
    """
    Resolve a volume's species mix from its ``params``.

    Recognised params (msgpack-primitive strings, per the zones contract):

    - ``params["species"] = "tree_gnarled_oak"`` — single species.
    - ``params["species_mix"] = "tree_gnarled_oak:3,tree_dead:1"`` —
      weighted mix (``name:weight`` pairs, comma-separated; weight
      defaults to 1).
    - neither → ``[(default, 1.0)]``.

    Returns
    -------
    list[tuple[str, float]]
        ``(species_name, weight)`` pairs, weights > 0.

    Example
    -------
    >>> species_mix_from_params({"species_mix": "a:3, b"}, "c")
    [('a', 3.0), ('b', 1.0)]
    """
    if "species_mix" in params:
        mix: list[tuple[str, float]] = []
        for entry in str(params["species_mix"]).split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, _, w = entry.partition(":")
            weight = float(w) if w.strip() else 1.0
            if weight > 0.0:
                mix.append((name.strip(), weight))
        if not mix:
            raise ValueError(f"species_mix parsed to nothing: {params['species_mix']!r}")
        return mix
    if "species" in params:
        return [(str(params["species"]), 1.0)]
    return [(default, 1.0)]


def bake_tree_instances(
    volume: ZoneVolume,
    config: Config,
    chunks: Mapping[tuple[int, int, int], object],
    species_weights: list[tuple[str, float]],
    variant_counts: Mapping[str, int],
    kind: str = "trees",
) -> TreeInstances:
    """
    Bake the deterministic instance set for one ``"trees"``/``"bushes"``
    volume.

    Jittered-grid candidates → density keep-probability → terrain-surface Z
    (height-field bake; sentinel = drop) → cap → per-instance attributes +
    species/variant assignment.  Pure function of (volume, config, chunks,
    mix) — re-baked by the renderer when terrain edits dirty the volume.

    Parameters
    ----------
    volume : ZoneVolume
        The tree/bush volume (AABB defines the grid and the Z window).
    config : Config
        Provides ``<kind>_density_per_m2`` / ``<kind>_min_spacing_m`` /
        ``<kind>_max_instances`` (``tree_``/``bush_`` prefixes) plus the
        height-field fields.  Density may be overridden per volume via
        ``params["density"]``.
    chunks : Mapping
        Loaded chunks (``ChunkManager.chunks``) for the height-field bake.
    species_weights : list[tuple[str, float]]
        From :func:`species_mix_from_params`.
    variant_counts : Mapping[str, int]
        Mesh-pool size per species name (``TreeVariantSet.n_variants``).
    kind : str
        ``"trees"`` or ``"bushes"``.

    Returns
    -------
    TreeInstances
        May be empty (no ground in the Z window, density 0, …).
    """
    prefix = _KIND_PREFIX[kind]
    scale_min, scale_span = SCALE_JITTER[kind]
    density = float(volume.params.get("density", getattr(config, f"{prefix}_density_per_m2")))
    min_spacing = float(getattr(config, f"{prefix}_min_spacing_m"))
    cap = int(getattr(config, f"{prefix}_max_instances"))
    rng = for_domain("zones", "tree_placement", kind, volume.id)

    x0, y0, z0 = volume.min_corner
    x1, y1, z1 = volume.max_corner
    names = tuple(n for n, _ in species_weights)

    if density <= 0.0 or cap <= 0:
        return _empty_instances(names)

    # --- jittered grid candidates ---------------------------------------
    cell = max(min_spacing, 1.0 / math.sqrt(density))
    nx = max(1, int(math.ceil((x1 - x0) / cell)))
    ny = max(1, int(math.ceil((y1 - y0) / cell)))
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx = x0 + (gx.ravel() + 0.5) * cell
    cy = y0 + (gy.ravel() + 0.5) * cell
    n_cells = cx.shape[0]
    jit = rng.uniform(-_JITTER_FRAC * cell, _JITTER_FRAC * cell, size=(n_cells, 2))
    px = (cx + jit[:, 0]).astype(np.float32)
    py = (cy + jit[:, 1]).astype(np.float32)

    keep_p = min(density * cell * cell, 1.0)
    keep = rng.random(n_cells) < keep_p
    keep &= (px >= x0) & (px < x1) & (py >= y0) & (py < y1)

    # --- terrain surface Z (same bake grass stands on) -------------------
    field = bake_grass_height_field(volume, chunks, config)
    H, W = field.shape[:2]
    vs = float(config.voxel_size)
    ix = np.clip(((px - x0) / vs).astype(np.int64), 0, W - 1)
    iy = np.clip(((py - y0) / vs).astype(np.int64), 0, H - 1)
    r = field[iy, ix, 0]
    keep &= r != HEIGHT_SENTINEL

    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return _empty_instances(names)
    if idx.size > cap:
        # Deterministic spatially-unbiased thinning to the cap.
        idx = idx[np.sort(rng.permutation(idx.size)[:cap])]

    n = idx.size
    z = np.float32(z0) + r[idx].astype(np.float32) / np.float32(254.0) * np.float32(z1 - z0)

    # --- per-instance attributes -----------------------------------------
    weights = np.asarray([w for _, w in species_weights], dtype=np.float64)
    weights /= weights.sum()
    species_idx = rng.choice(len(names), size=n, p=weights).astype(np.int32)
    pool = np.asarray([int(variant_counts[nm]) for nm in names], dtype=np.int32)
    variant = (rng.random(n) * pool[species_idx]).astype(np.int32)

    two_pi = 2.0 * math.pi
    return TreeInstances(
        x=px[idx],
        y=py[idx],
        z=z,
        yaw=rng.uniform(0.0, two_pi, n).astype(np.float32),
        scale=(scale_min + scale_span * rng.random(n)).astype(np.float32),
        phase=rng.uniform(0.0, two_pi, n).astype(np.float32),
        tint=(_TINT_MIN + _TINT_SPAN * rng.random(n)).astype(np.float32),
        species_idx=species_idx,
        variant=variant,
        species_names=names,
    )


def instances_data_block(inst: TreeInstances, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Pack instances into the renderer's data-texture block.

    Layout (the GLSL ``texelFetch`` contract — tree.vert and
    tree_impostor.vert read EXACTLY this; the placement test pins it):

    - texel ``(column 0, row i)`` = ``(x, y, z, yaw)``
    - texel ``(column 1, row i)`` = ``(scale, phase, tint, variant)``

    Parameters
    ----------
    inst : TreeInstances
        A volume's baked instances.
    mask : numpy.ndarray | None
        Optional boolean selector — the renderer passes
        ``(species_idx == s) & (variant == v)`` to build one mesh draw's
        texture, and a species-only mask for the impostor draw.

    Returns
    -------
    numpy.ndarray
        ``float32 (n, 2, 4)`` — row-major rows = instances, columns =
        texels.  May be ``(0, 2, 4)``.
    """
    sel = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    block = np.stack(
        [
            np.stack([inst.x[sel], inst.y[sel], inst.z[sel], inst.yaw[sel]], axis=1),
            np.stack(
                [
                    inst.scale[sel],
                    inst.phase[sel],
                    inst.tint[sel],
                    inst.variant[sel].astype(np.float32),
                ],
                axis=1,
            ),
        ],
        axis=1,
    )
    return np.ascontiguousarray(block, dtype=np.float32)


def _empty_instances(names: tuple[str, ...]) -> TreeInstances:
    """Zero-instance result (keeps dtypes/fields consistent)."""
    f = np.empty(0, dtype=np.float32)
    i = np.empty(0, dtype=np.int32)
    return TreeInstances(
        x=f,
        y=f.copy(),
        z=f.copy(),
        yaw=f.copy(),
        scale=f.copy(),
        phase=f.copy(),
        tint=f.copy(),
        species_idx=i,
        variant=i.copy(),
        species_names=names,
    )
