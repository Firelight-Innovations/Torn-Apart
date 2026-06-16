"""
Trivial support types (frozen dataclasses / NamedTuples) for fire_engine.lighting.

Grouping module — may define more than one public type.  All types are
re-exported from the originating parent modules so historical import paths
remain unchanged.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fire_engine.lighting.occluders import TreeOccluderSet
from fire_engine.lighting.palette import MaterialPalette

__all__ = [
    "AreaLight",
    "AssemblyJob",
    "AssemblyResult",
    "GeometryVolume",
    "PointLight",
    "SpotLight",
]


# ---------------------------------------------------------------------------
# assembly_worker types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssemblyJob:
    """
    One cascade-volume reassembly request.

    Attributes
    ----------
    cascade_index : int
        Which cascade (0 or 1) this volume is for.
    origin_cell : tuple[int, int, int]
        Window origin (in cells) the volume is assembled for — committed to the
        cascade window only when the result is uploaded.
    cells, cell_m : int, float
        Window dimensions (texels per axis, meters per cell).
    chunk_size, voxel_size : int, float
        Terrain constants for the slice/downsample math.
    materials : dict[tuple[int, int, int], numpy.ndarray]
        Snapshot of ``uint8 (S, S, S)`` material arrays for the chunks the
        gather will read (references, not copies — see
        ``gpu.py``), submits it, and continues rendering with the previously
        committed volume.
    palette : MaterialPalette
        Immutable material → albedo/emission lookup (safe to share read-only).
    seq : int
        Monotonic id; lets the consumer drop a superseded result.
    occluders : TreeOccluderSet | None
        Static tree/bush occluder snapshot splatted into the volume (see
        ``lighting/occluders.py``).  Immutable struct-of-arrays — safe to
        share read-only across the thread boundary.  ``None`` → chunks only.
    trunk_occ : float
        Trunk splat opacity (``config.light_tree_trunk_occ``).
    canopy_gain : float
        Multiplier on the per-instance leaf-derived canopy extinction
        (``config.light_tree_canopy_extinction_gain``).

    Docs: docs/systems/lighting._impl.md
    """

    cascade_index: int
    origin_cell: tuple[int, int, int]
    cells: int
    cell_m: float
    chunk_size: int
    voxel_size: float
    materials: dict[tuple[int, int, int], Any]
    palette: MaterialPalette
    seq: int
    occluders: TreeOccluderSet | None = None
    trunk_occ: float = 0.0
    canopy_gain: float = 0.0


@dataclass(frozen=True)
class AssemblyResult:
    """
    A finished cascade volume, packed and ready for ``Texture.set_ram_image``.

    Attributes
    ----------
    cascade_index : int
    origin_cell : tuple[int, int, int]
        The origin the volume was assembled for (commit this to the window).
    albedo_bytes, emis_bytes : bytes
        Page-major BGRA 3-D-texture RAM images (see ``volume.pack_volume``).
    seq : int
        Echoes the job's ``seq``.

    Docs: docs/systems/lighting._impl.md
    """

    cascade_index: int
    origin_cell: tuple[int, int, int]
    albedo_bytes: bytes
    emis_bytes: bytes
    seq: int


# ---------------------------------------------------------------------------
# lights types
# ---------------------------------------------------------------------------


@dataclass
class PointLight:
    """
    Omnidirectional punctual light.

    Attributes
    ----------
    position : tuple[float, float, float]
        World position in meters.
    color : tuple[float, float, float]
        Linear RGB in [0, 1] (hue only; brightness lives in ``intensity``).
    intensity : float
        HDR radiant intensity (a cosy torch ≈ 2–4, an explosion flash ≈ 30).
    radius : float
        Falloff window in meters — zero contribution beyond this.
    ttl_s : float | None
        Lifetime in seconds; ``None`` = permanent.  Transient lights fade
        linearly over their lifetime and are removed at expiry.

    Docs: docs/systems/lighting._impl.md
    """

    position: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    radius: float
    ttl_s: float | None = None


@dataclass
class AreaLight:
    """
    Axis-aligned emissive box light (windows, lava pools, shafts).

    Attributes
    ----------
    center : tuple[float, float, float]
        Box centre, world meters.
    half_extents : tuple[float, float, float]
        Box half sizes per axis, meters.
    color : tuple[float, float, float]
        Linear RGB in [0, 1].
    intensity : float
        HDR radiant intensity at the box surface.
    radius : float
        Falloff window in meters measured from the box *surface*.
    ttl_s : float | None
        Lifetime in seconds; ``None`` = permanent.

    Docs: docs/systems/lighting._impl.md
    """

    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    radius: float
    ttl_s: float | None = None


@dataclass
class SpotLight:
    """
    Cone-restricted punctual light (flashlight, lighthouse, vehicle beam).

    Same windowed inverse-square falloff and occupancy-march shadowing as
    :class:`PointLight`, multiplied by a smooth cone window: full strength
    inside the cone core, fading to zero at the cone edge.

    Attributes
    ----------
    position : tuple[float, float, float]
        World position in meters (e.g. the camera position).
    direction : tuple[float, float, float]
        Unit beam direction, world space (e.g. camera forward).  Normalised
        defensively at pack time.
    color : tuple[float, float, float]
        Linear RGB in [0, 1].
    intensity : float
        HDR radiant intensity (a hand flashlight ≈ 10–18).
    radius : float
        Falloff window in meters — zero contribution beyond this.
    cone_deg : float
        FULL cone angle in degrees (a typical flashlight ≈ 35–50).
    ttl_s : float | None
        Lifetime in seconds; ``None`` = permanent.

    Example
    -------
    >>> ls = LightSet()
    >>> torch = SpotLight(position=(0, 0, 10), direction=(0, 1, 0),
    ...                   color=(1.0, 0.95, 0.8), intensity=14.0,
    ...                   radius=30.0, cone_deg=42.0)
    >>> lid = ls.add(torch)
    >>> torch.position = (0, 2, 10)   # follow the camera...
    >>> ls.notify_changed()           # ...then mark the packed data stale

    Docs: docs/systems/lighting._impl.md
    """

    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    radius: float
    cone_deg: float = 42.0
    ttl_s: float | None = None


# ---------------------------------------------------------------------------
# volume types
# ---------------------------------------------------------------------------


@dataclass
class GeometryVolume:
    """
    Packed world-geometry block for one cascade, ready for GPU upload.

    Attributes
    ----------
    albedo_occ : numpy.ndarray
        ``uint8 (N, N, N, 4)`` indexed ``[x, y, z]``: RGB = surface albedo
        (linear, 0–255), A = **solid sub-voxel fraction ×255**: the fraction
        of the cell's ``k³`` terrain voxels that are solid, rounded to a byte.
        At cascade 0 (``cell_m == voxel_size``, ``k == 1``) this is exactly
        255 (solid) or 0 (air), identical to a binary occupancy flag.  At the
        coarse cascades it is a partial value, so a hollow room reads air
        (A == 0) in its interior and only its 1-voxel walls read partly-solid
        — the GPU probes no longer treat a hollow box as a solid block.
    emission : numpy.ndarray
        ``uint8 (N, N, N, 4)``: RGB = emitted radiance / ``EMISSION_SCALE``
        (clipped to 255), A unused (255).
    origin_cell : tuple[int, int, int]
        World cell index of texel (0,0,0) at assembly time.
    cell_m : float
        Cell edge in meters.

    Docs: docs/systems/lighting._impl.md
    """

    albedo_occ: np.ndarray
    emission: np.ndarray
    origin_cell: tuple[int, int, int]
    cell_m: float
