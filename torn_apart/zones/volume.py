"""
zones/volume.py — ZoneVolume: a tagged axis-aligned box volume in world space.

A ZoneVolume marks a rectangular region of the world for some system to act
on: ``tag="grass"`` volumes tell the GPU grass renderer where blades grow;
``tag="biome"`` volumes (future) will retag the terrain surface (snow, bare
dirt, ...).  Volumes are pure data — frozen, serialisable to primitives, no
behaviour beyond geometry queries — so they ride through delta saves
unchanged (Hard Rule 3: no pickle, dicts of primitives only).

Units: world-space **meters**, Z-up, corners are (x, y, z) tuples with
``min_corner < max_corner`` on every axis.

Example
-------
    from torn_apart.zones import ZoneVolume

    vol = ZoneVolume(
        id=1, tag="grass",
        min_corner=(-12.0, -5.0, 6.0),
        max_corner=( 12.0, 25.0, 10.0),
        params={"density": 12.0},        # blades per square meter
    )
    vol.area_xy_m2          # 720.0
    vol.to_dict()           # plain primitives — save-safe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ZoneVolume"]


@dataclass(frozen=True)
class ZoneVolume:
    """
    Immutable tagged box volume (world meters).

    Attributes
    ----------
    id : int
        Unique id assigned by :class:`~torn_apart.zones.store.ZoneStore`.
    tag : str
        What the volume means: ``"grass"`` (grass spawn region) or
        ``"biome"`` (surface-material region, future).
    min_corner / max_corner : tuple[float, float, float]
        World-space AABB corners in meters, ``min < max`` per axis.
    biome : str | None
        For ``tag="biome"`` volumes: the biome name (``"snow"``, ``"dirt"``).
        ``None`` for other tags.
    params : dict[str, float]
        Per-volume tuning values (e.g. grass ``"density"`` in blades/m²).
        Keys/values must stay msgpack-primitive for saves.

    Example
    -------
    >>> v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 20.0, 4.0))
    >>> v.area_xy_m2
    200.0
    """

    id: int
    tag: str
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]
    biome: str | None = None
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("ZoneVolume.tag must be a non-empty string")
        lo = tuple(float(v) for v in self.min_corner)
        hi = tuple(float(v) for v in self.max_corner)
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError("ZoneVolume corners must be 3-tuples (x, y, z)")
        if not all(a < b for a, b in zip(lo, hi)):
            raise ValueError(
                f"ZoneVolume min_corner {lo} must be < max_corner {hi} "
                "on every axis")
        # Normalise to float tuples (frozen dataclass → object.__setattr__).
        object.__setattr__(self, "min_corner", lo)
        object.__setattr__(self, "max_corner", hi)

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    @property
    def size_m(self) -> tuple[float, float, float]:
        """Edge lengths (x, y, z) in meters."""
        return (self.max_corner[0] - self.min_corner[0],
                self.max_corner[1] - self.min_corner[1],
                self.max_corner[2] - self.min_corner[2])

    @property
    def area_xy_m2(self) -> float:
        """Footprint area in square meters (XY plane)."""
        sx, sy, _ = self.size_m
        return sx * sy

    def contains_xy(self, world_x: np.ndarray,
                    world_y: np.ndarray) -> np.ndarray:
        """
        Vectorized XY containment test (Z ignored).

        Parameters
        ----------
        world_x, world_y : numpy.ndarray
            Broadcastable world coordinate arrays (meters).

        Returns
        -------
        numpy.ndarray
            Boolean array, True where ``min <= coord < max`` on both axes.
        """
        wx = np.asarray(world_x)
        wy = np.asarray(world_y)
        return ((wx >= self.min_corner[0]) & (wx < self.max_corner[0]) &
                (wy >= self.min_corner[1]) & (wy < self.max_corner[1]))

    def intersects_chunk(self, coord: tuple[int, int, int],
                         chunk_meters: float) -> bool:
        """
        True when this volume's AABB overlaps chunk ``coord``'s world box.

        Parameters
        ----------
        coord : tuple[int, int, int]
            Integer chunk coordinate ``(cx, cy, cz)``.
        chunk_meters : float
            World-space chunk edge length (``config.chunk_meters``, 16.0 m).
        """
        return all(
            coord[i] * chunk_meters < self.max_corner[i]
            and (coord[i] + 1) * chunk_meters > self.min_corner[i]
            for i in range(3))

    # ------------------------------------------------------------------
    # Serialisation (save-safe primitives only)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Reduce to a plain dict of primitives (msgpack/save-safe)."""
        return {
            "id": int(self.id),
            "tag": str(self.tag),
            "min_corner": list(self.min_corner),
            "max_corner": list(self.max_corner),
            "biome": self.biome,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneVolume":
        """Inverse of :meth:`to_dict`."""
        return cls(
            id=int(data["id"]),
            tag=str(data["tag"]),
            min_corner=tuple(float(v) for v in data["min_corner"]),
            max_corner=tuple(float(v) for v in data["max_corner"]),
            biome=data.get("biome"),
            params=dict(data.get("params") or {}),
        )
