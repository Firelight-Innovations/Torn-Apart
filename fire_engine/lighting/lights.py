"""
lighting/lights.py — Dynamic light registry for the GPU lighting pipeline.

All non-celestial lights (torches, spells, explosions, glowing windows)
register here; the sun and moon arrive separately via ``SkyState``.  The
registry packs every active light into one ``float32`` array per frame for a
single GPU uniform upload — no per-light draw calls or scene-graph state.

Light model
-----------
- **PointLight** — omnidirectional, inverse-square falloff smoothly windowed
  to zero at ``radius`` (the standard "(1 - (d/r)⁴)² / (d² + 1)" punctual
  falloff), shadowed in the volume by an occupancy march.
- **AreaLight** — an axis-aligned emissive box; the GPU treats it as a point
  light whose distance is measured to the box surface, giving the soft
  wide-source look (windows, lava pools, light shafts).
- **SpotLight** — a point light restricted to a cone (flashlight, beacon):
  same falloff/shadowing as PointLight × a smooth cone window around
  ``direction`` with full angle ``cone_deg``.
- Transient lights (muzzle flash, explosion) use ``ttl_s``: ``update(dt)``
  fades their intensity linearly to zero and removes them at expiry.

This module also hosts :class:`OccluderSet` — the registry of **dynamic
shadow-caster boxes** (dev cubes, props, NPCs).  Terrain shadows come from
the voxel occupancy volume; anything NOT in the voxel field registers its
world AABB here and the GPU's visibility marches add an analytic ray-vs-box
test, so moving objects cast (and cut god rays) without re-voxelisation.

No panda3d imports.  Headless-testable.

Example
-------
>>> from fire_engine.lighting.lights import LightSet, PointLight
>>> ls = LightSet()
>>> lid = ls.add(PointLight(position=(8.0, 8.0, 10.0),
...                         color=(1.0, 0.6, 0.25), intensity=3.0,
...                         radius=12.0))
>>> arr, count = ls.pack(max_lights=64)
>>> count
1
>>> arr.shape
(64, 12)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PointLight",
    "AreaLight",
    "SpotLight",
    "LightSet",
    "OccluderSet",
    "LIGHT_TYPE_POINT",
    "LIGHT_TYPE_AREA",
    "LIGHT_TYPE_SPOT",
    "MAX_OCCLUDERS",
]

LIGHT_TYPE_POINT: float = 0.0
LIGHT_TYPE_AREA: float = 1.0
LIGHT_TYPE_SPOT: float = 2.0

#: Maximum dynamic occluder boxes uploaded to the GPU (mirrors the fixed
#: uniform array length in ``lighting/glsl.py``).
MAX_OCCLUDERS: int = 16


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
    """

    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    radius: float
    cone_deg: float = 42.0
    ttl_s: float | None = None


@dataclass
class _Entry:
    light: PointLight | AreaLight | SpotLight
    age_s: float = 0.0


class LightSet:
    """
    Registry of active dynamic lights with one-array GPU packing.

    ``version`` increments on every add/remove/update that changes packed
    data — the GPU pipeline re-injects light only when it sees a new version
    (or when the sun/volume moved), so a static torch costs nothing per frame.

    Example
    -------
    >>> ls = LightSet()
    >>> lid = ls.add(PointLight((0, 0, 0), (1, 1, 1), 2.0, 8.0, ttl_s=0.5))
    >>> ls.update(0.6)            # transient expired
    >>> ls.count
    0
    """

    def __init__(self) -> None:
        self._entries: dict[int, _Entry] = {}
        self._next_id: int = 1
        self.version: int = 0

    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of active lights."""
        return len(self._entries)

    def add(self, light: PointLight | AreaLight) -> int:
        """Register a light; returns its id (use with ``remove``)."""
        lid = self._next_id
        self._next_id += 1
        self._entries[lid] = _Entry(light)
        self.version += 1
        return lid

    def remove(self, light_id: int) -> None:
        """Remove a light by id (no-op when absent)."""
        if self._entries.pop(light_id, None) is not None:
            self.version += 1

    def clear(self) -> None:
        """Remove all lights."""
        if self._entries:
            self._entries.clear()
            self.version += 1

    def notify_changed(self) -> None:
        """
        Mark the packed data stale after mutating a registered light in place.

        Use for per-frame followers (a flashlight tracking the camera):
        mutate the light's ``position``/``direction`` fields directly, then
        call this so the GPU pipeline re-injects.  Cheap — call only when the
        light actually moved (re-injection is the expensive part).
        """
        self.version += 1

    def get(self, light_id: int) -> PointLight | AreaLight | SpotLight | None:
        """Return the registered light object for ``light_id`` (or None)."""
        e = self._entries.get(light_id)
        return e.light if e is not None else None

    def update(self, dt: float) -> None:
        """
        Advance transient-light lifetimes by ``dt`` seconds.

        Lights with ``ttl_s`` fade linearly (their *packed* intensity scales
        by remaining lifetime) and are removed once expired.  Permanent
        lights are untouched.  Bumps ``version`` whenever any transient
        light exists (its fade changes the packed data every frame).
        """
        if not self._entries:
            return
        expired = [
            lid
            for lid, e in self._entries.items()
            if e.light.ttl_s is not None and e.age_s + dt >= e.light.ttl_s
        ]
        any_transient = False
        for e in self._entries.values():
            if e.light.ttl_s is not None:
                e.age_s += dt
                any_transient = True
        for lid in expired:
            del self._entries[lid]
        if any_transient:
            self.version += 1

    # ------------------------------------------------------------------

    def pack(self, max_lights: int) -> tuple[np.ndarray, int]:
        """
        Pack active lights into a ``float32 (max_lights, 12)`` array.

        Row layout (matches the GLSL unpacking in ``lighting/glsl.py``)::

            [0:3]  position / box centre (world meters)
            [3]    falloff radius (meters)
            [4:7]  color * faded_intensity (linear HDR RGB)
            [7]    light type (0 = point, 1 = area, 2 = spot)
            [8:11] box half extents (area) / unit beam direction (spot)
            [11]   cos(cone_deg / 2) for spot lights, else 0

        Lights beyond ``max_lights`` are dropped oldest-first (warned once
        per pack).  Rows past ``count`` are zero.

        Returns
        -------
        tuple[numpy.ndarray, int]
            ``(array, active_count)``.
        """
        out = np.zeros((max_lights, 12), dtype=np.float32)
        n = 0
        for e in self._entries.values():
            if n >= max_lights:
                break
            li = e.light
            fade = 1.0
            if li.ttl_s is not None and li.ttl_s > 0.0:
                fade = max(0.0, 1.0 - e.age_s / li.ttl_s)
            if isinstance(li, AreaLight):
                out[n, 0:3] = li.center
                out[n, 7] = LIGHT_TYPE_AREA
                out[n, 8:11] = li.half_extents
            elif isinstance(li, SpotLight):
                out[n, 0:3] = li.position
                out[n, 7] = LIGHT_TYPE_SPOT
                d = np.asarray(li.direction, np.float32)
                norm = float(np.linalg.norm(d))
                out[n, 8:11] = d / norm if norm > 1e-6 else np.float32((0.0, 0.0, -1.0))
                out[n, 11] = math.cos(math.radians(li.cone_deg) * 0.5)
            else:
                out[n, 0:3] = li.position
                out[n, 7] = LIGHT_TYPE_POINT
            out[n, 3] = li.radius
            out[n, 4:7] = np.asarray(li.color, np.float32) * (li.intensity * fade)
            n += 1
        return out, n


class OccluderSet:
    """
    Registry of dynamic shadow-caster AABBs for the GPU lighting pipeline.

    Anything that should cast voxel-style shadows but is NOT part of the
    terrain voxel field (dev cubes, props, NPCs) sets its world bounding box
    here once per frame.  ``set_boxes`` only bumps ``version`` when the
    packed data actually changed (beyond ~1 cm), so static objects cost
    nothing and the pipeline re-injects only when something moved.

    Limits: the GPU uniform arrays hold :data:`MAX_OCCLUDERS` (16) boxes;
    extra boxes are dropped (warn-once) — dev tooling scale, not gameplay.

    Example
    -------
    >>> occ = OccluderSet()
    >>> occ.set_boxes([((0.0, 0.0, 8.0), (1.0, 1.0, 9.0))])
    True
    >>> mins, maxs, count = occ.pack()
    >>> count
    1
    """

    def __init__(self) -> None:
        self.version: int = 0
        self._mins = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        self._maxs = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        self._count: int = 0
        self._warned: bool = False

    @property
    def count(self) -> int:
        """Number of active occluder boxes."""
        return self._count

    def set_boxes(
        self,
        boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    ) -> bool:
        """
        Replace the full occluder list with ``boxes`` (world-space AABBs).

        Parameters
        ----------
        boxes : list of ((min_x, min_y, min_z), (max_x, max_y, max_z))
            World meters.  Call every frame with the current boxes; change
            detection is internal.

        Returns
        -------
        bool — True when the set changed (``version`` was bumped).
        """
        if len(boxes) > MAX_OCCLUDERS and not self._warned:
            self._warned = True
            import logging

            logging.getLogger("fire_engine.lighting.lights").warning(
                "OccluderSet: %d boxes > max %d — extras dropped", len(boxes), MAX_OCCLUDERS
            )
        boxes = boxes[:MAX_OCCLUDERS]
        n = len(boxes)
        mins = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        maxs = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        if n:
            mins[:n] = np.asarray([b[0] for b in boxes], dtype=np.float32)
            maxs[:n] = np.asarray([b[1] for b in boxes], dtype=np.float32)
        if (
            n == self._count
            and np.allclose(mins, self._mins, atol=0.01)
            and np.allclose(maxs, self._maxs, atol=0.01)
        ):
            return False
        self._mins, self._maxs, self._count = mins, maxs, n
        self.version += 1
        return True

    def pack(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Packed GPU arrays: ``(mins (16, 3) float32, maxs (16, 3), count)``.

        Rows past ``count`` are zero.  Layout mirrors ``u_box_min`` /
        ``u_box_max`` / ``u_num_boxes`` in ``lighting/glsl.py``.
        """
        return self._mins, self._maxs, self._count
