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
from dataclasses import dataclass

import numpy as np

from fire_engine.lighting._impl.occluder_set import (
    MAX_OCCLUDERS,
    OccluderSet,
)
from fire_engine.lighting._impl.types import (
    AreaLight,
    PointLight,
    SpotLight,
)

__all__ = [
    "LIGHT_TYPE_AREA",
    "LIGHT_TYPE_POINT",
    "LIGHT_TYPE_SPOT",
    "MAX_OCCLUDERS",
    "AreaLight",
    "LightSet",
    "OccluderSet",
    "PointLight",
    "SpotLight",
]

LIGHT_TYPE_POINT: float = 0.0
LIGHT_TYPE_AREA: float = 1.0
LIGHT_TYPE_SPOT: float = 2.0


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
                out[n, 8:11] = d / norm if norm > 1e-6 else np.asarray((0.0, 0.0, -1.0), np.float32)
                out[n, 11] = math.cos(math.radians(li.cone_deg) * 0.5)
            else:
                out[n, 0:3] = li.position
                out[n, 7] = LIGHT_TYPE_POINT
            out[n, 3] = li.radius
            out[n, 4:7] = np.asarray(li.color, np.float32) * (li.intensity * fade)
            n += 1
        return out, n
