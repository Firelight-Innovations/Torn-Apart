"""
devtools/picking.py — CPU ray/AABB object picking for click-to-select.

The overlay computes a world-space ray from the mouse cursor through the camera
(a panda3d operation, done in ``world/``) and hands the ray here.  This module
intersects it against the axis-aligned bounding box of every registered
:class:`Selectable` and returns the nearest hit.  Keeping the math here (pure
numpy, no panda3d) makes selection deterministic and unit-testable, and avoids
standing up a Panda3D collision graph just to click on a handful of dev objects.

AABBs are world-axis-aligned and derived from the object's world position ±
(half-extents × per-axis scale); object rotation is ignored for v1 picking
(boxes stay axis-aligned).  That is plenty accurate for selecting dev props and
spawned primitives; tighten later if rotated props need pixel-exact picking.

No panda3d imports — headless-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from fire_engine.core.math3d import Vec3

if TYPE_CHECKING:
    from fire_engine.world.gameobject import GameObject


@dataclass
class Selectable:
    """
    A GameObject that can be clicked in the viewport, plus its local AABB.

    Parameters
    ----------
    game_object : GameObject
        The object this entry selects.  Read duck-typed (``.transform``); no
        world/ import happens at runtime.
    half_extents : Vec3
        Half the box size along each local axis, in **meters**, before scale.
        A 1 m cube uses ``Vec3(0.5, 0.5, 0.5)``.

    Notes
    -----
    The world AABB is recomputed on demand from the object's current transform,
    so the box follows the object as it moves (e.g. when you edit its position
    in the Inspector).
    """

    game_object: "GameObject"
    half_extents: Vec3

    def world_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(min_corner, max_corner)`` of this object's world-space AABB.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Two length-3 float64 arrays (world meters).
        """
        tf = self.game_object.transform
        center = tf.position.to_numpy().astype(np.float64)
        scale = tf.local_scale.to_numpy().astype(np.float64)
        ext = self.half_extents.to_numpy().astype(np.float64) * np.abs(scale)
        return center - ext, center + ext


def ray_aabb(
    origin: Vec3,
    direction: Vec3,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> Optional[float]:
    """
    Slab-method ray vs. axis-aligned box intersection.

    Parameters
    ----------
    origin : Vec3
        Ray origin in world meters.
    direction : Vec3
        Ray direction (need not be normalised; must be non-zero).
    box_min, box_max : np.ndarray
        Length-3 arrays giving the AABB corners (world meters).

    Returns
    -------
    float | None
        The ray parameter ``t`` (distance along ``direction``) of the entry
        point if the ray hits the box at ``t >= 0``; ``None`` on a miss.  When
        the origin is inside the box, returns ``0.0``.

    Notes
    -----
    Division by a zero direction component is handled via numpy's inf semantics
    (a ray parallel to a slab misses only if its origin is outside that slab).
    """
    o = origin.to_numpy().astype(np.float64)
    d = direction.to_numpy().astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (box_min - o) * inv
        t2 = (box_max - o) * inv

    t_near = np.nanmax(np.minimum(t1, t2))
    t_far = np.nanmin(np.maximum(t1, t2))

    if t_far < 0.0 or t_near > t_far:
        return None
    return float(max(t_near, 0.0))


def pick(
    origin: Vec3,
    direction: Vec3,
    selectables: list[Selectable],
) -> "Optional[GameObject]":
    """
    Return the nearest GameObject whose AABB the ray hits, or ``None``.

    Parameters
    ----------
    origin : Vec3
        Ray origin in world meters (typically the near-plane point under the
        mouse cursor).
    direction : Vec3
        Ray direction in world meters (cursor-through-camera direction).
    selectables : list[Selectable]
        Candidate objects.

    Returns
    -------
    GameObject | None
        The closest intersected object's GameObject, or ``None`` on a full miss.

    Example
    -------
        go = pick(ray_origin, ray_dir, manager.selectables)
        if go is not None:
            selection.set(go)
    """
    best_t: Optional[float] = None
    best_go: "Optional[GameObject]" = None
    for sel in selectables:
        bmin, bmax = sel.world_aabb()
        t = ray_aabb(origin, direction, bmin, bmax)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_go = sel.game_object
    return best_go
