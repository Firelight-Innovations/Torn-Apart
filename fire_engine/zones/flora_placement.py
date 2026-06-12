"""
zones/flora_placement.py — headless math for GPU-only flora placement.

The flora renderer (``world/flora_renderer.py``) draws sprite flora via
hardware instancing, exactly like grass: the CPU stores **no per-plant
data** — each instance derives its world position, rotation, scale, sway
phase and **atlas variant** in the vertex shader from ``gl_InstanceID``
through an integer hash.  This module is the testable, panda3d-free half of
that contract:

- :func:`flora_instance_attribs` — a **line-for-line python mirror of the
  GLSL hash chain** in ``world/shaders/flora.vert``.  It extends the grass
  chain (``grass_placement.instance_attribs``) with one more hash link that
  picks the sprite-atlas variant, and parameterises the scale-jitter range.
  The two must stay byte-identical; the headless tests pin this mirror so
  any GLSL edit that forgets it fails.
- :func:`flora_hash_seed` — per-(volume, kind) hash seed via
  ``core.rng.for_domain`` (Hard Rule 2).
- :func:`flora_instance_count` — plant count from volume area × density,
  with per-kind config defaults and caps.

Flora kinds are the zone tags — today just ``"flowers"``.  Trees and bushes
graduated to real 3-D meshes with their own CPU-baked placement
(``zones/tree_placement.py``); a ``"trees"`` volume still also feeds the
wind system's leaf litter (``leaf_hash_seed`` / ``leaf_instance_count``).

The terrain height field flora stands on is the SAME bake grass uses
(:func:`zones.grass_placement.bake_grass_height_field`) — it is generic over
any volume.

Units: meters, Z-up.  All bulk work is numpy (Hard Rule 4).

Example
-------
    from fire_engine.zones import flora_placement as fp

    count = fp.flora_instance_count(vol, cfg, "flowers")
    seed  = fp.flora_hash_seed(vol, "flowers")          # uniform u_hash_seed
    attrs = fp.flora_instance_attribs(np.arange(count), seed,
                                      vol.min_corner, vol.max_corner,
                                      n_variants=4)
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.core import Config
from fire_engine.core.rng import for_domain
from fire_engine.zones.grass_placement import hash_lowbias32, _K1, _K2, _K3, _K4
from fire_engine.zones.volume import ZoneVolume

__all__ = [
    "FLORA_KINDS",
    "flora_hash_seed",
    "flora_instance_count",
    "flora_instance_attribs",
]

# The zone tags the flora renderer consumes, in render order.  "bushes" and
# "trees" volumes belong to zones/tree_placement.py (3-D meshes) now.
FLORA_KINDS: tuple[str, ...] = ("flowers",)

# Per-kind (density config field, max-instances config field).  Density may be
# overridden per volume via params["density"].
_KIND_CONFIG: dict[str, tuple[str, str]] = {
    "flowers": ("flora_flower_density_per_m2", "flora_flower_max_instances"),
}

# Fifth hash-chain XOR constant (variant link) — MUST match flora.vert.
_K5 = np.uint32(0x165667B1)


def flora_hash_seed(volume: ZoneVolume, kind: str) -> int:
    """
    Deterministic per-(volume, kind) hash seed for the instance chain.

    Derived through ``for_domain("zones", "flora", kind, volume.id)``
    (Hard Rule 2) so the same world seed + volume id + kind always places
    identical plants.  Bounded to ``[0, 2**31)`` (Panda3D passes shader
    ints as signed).

    Parameters
    ----------
    volume : ZoneVolume
        The flora volume.
    kind : str
        One of :data:`FLORA_KINDS` (``"flowers"``).
    """
    return int(for_domain("zones", "flora", kind, volume.id)
               .integers(0, 2 ** 31))


def flora_instance_count(volume: ZoneVolume, config: Config, kind: str) -> int:
    """
    Number of plant instances a flora volume spawns for one kind.

    ``density × footprint area``, where density (plants/m²) comes from the
    volume's ``params["density"]`` or the kind's config default
    (``flora_<kind>_density_per_m2``), clamped to the kind's
    ``flora_<kind>_max_instances`` cap.  Pure function.

    Example
    -------
    >>> from fire_engine.core import Config
    >>> from fire_engine.zones import ZoneVolume
    >>> v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (20.0, 20.0, 8.0))
    >>> flora_instance_count(v, Config(), "flowers")  # 400 m² × 1.5
    600
    """
    density_field, max_field = _KIND_CONFIG[kind]
    density = float(volume.params.get("density", getattr(config, density_field)))
    count = int(volume.area_xy_m2 * max(density, 0.0))
    return max(0, min(count, int(getattr(config, max_field))))


def flora_instance_attribs(
    indices: np.ndarray,
    seed: int,
    min_corner: tuple[float, float, float],
    max_corner: tuple[float, float, float],
    n_variants: int,
    scale_min: float = 0.7,
    scale_span: float = 0.6,
) -> dict[str, np.ndarray]:
    """
    Per-instance flora placement attributes — python mirror of the GLSL chain.

    ``world/shaders/flora.vert`` computes EXACTLY this from ``gl_InstanceID``;
    this mirror exists so placement is testable headlessly (determinism,
    bounds, variant distribution) without a GPU.  It is the grass chain
    (``instance_attribs``) plus one more lowbias32 link (``h5``) that selects
    the sprite-atlas variant, with the scale-jitter range parameterised
    (the renderer passes the kind's ``u_scale_min`` / ``u_scale_span``).

    Parameters
    ----------
    indices : numpy.ndarray
        Instance ids (``gl_InstanceID`` values), any integer dtype.
    seed : int
        Per-(volume, kind) hash seed (:func:`flora_hash_seed`).
    min_corner / max_corner : tuple[float, float, float]
        The flora volume's AABB corners (world meters).
    n_variants : int
        Atlas cell count of the kind's sprite texture (4 for flowers).
    scale_min, scale_span : float
        Per-instance size multiplier range ``[scale_min, scale_min +
        scale_span)`` — flowers keep the grass default 0.7–1.3.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``"x"``/``"y"`` world-space plant base positions (float32, meters),
        ``"rot"`` yaw radians [0, 2π), ``"scale"`` size multiplier,
        ``"phase"`` sway phase radians [0, 2π), ``"variant"`` atlas cell
        index (int32, ``h5 % n_variants``).
    """
    i = np.asarray(indices).astype(np.uint32, copy=False)
    h0 = hash_lowbias32(i ^ np.uint32(seed))
    h1 = hash_lowbias32(h0 ^ _K1)
    h2 = hash_lowbias32(h1 ^ _K2)
    h3 = hash_lowbias32(h2 ^ _K3)
    h4 = hash_lowbias32(h3 ^ _K4)
    h5 = hash_lowbias32(h4 ^ _K5)

    inv = np.float32(1.0 / 4294967296.0)   # 1 / 2^32 — matches GLSL u2f()
    size_x = np.float32(max_corner[0] - min_corner[0])
    size_y = np.float32(max_corner[1] - min_corner[1])
    two_pi = np.float32(2.0 * math.pi)
    return {
        "x": np.float32(min_corner[0]) + h0.astype(np.float32) * inv * size_x,
        "y": np.float32(min_corner[1]) + h1.astype(np.float32) * inv * size_y,
        "rot": h2.astype(np.float32) * inv * two_pi,
        "scale": np.float32(scale_min)
        + h3.astype(np.float32) * inv * np.float32(scale_span),
        "phase": h4.astype(np.float32) * inv * two_pi,
        "variant": (h5 % np.uint32(max(n_variants, 1))).astype(np.int32),
    }
