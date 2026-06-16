"""
world/transform.py — Quaternion-based Transform component for the Torn Apart engine.

Every GameObject owns exactly one Transform.  The Transform holds local TRS
(translation, rotation, scale) state and derives world-space position and
rotation through the parent chain.  A dirty flag + propagation to descendants
ensures world matrices are re-computed lazily and only when the hierarchy
actually changes.

Coordinate system (Z-up, Panda3D native, matches core.math3d):
    forward = +Y
    right   = +X
    up      = +Z

All distances are in **meters**.  Rotations are stored as Quat (quaternions);
Euler angles are a presentation layer only — never stored as state.

No panda3d imports — this module is fully headless-testable.
The world/ sync layer (app.py) converts math3d types to Panda3D types at the
scene-graph boundary.

Example
-------
    from fire_engine.render.transform import Transform, Space
    from fire_engine.core.math3d import Vec3, Quat
    from math import pi

    parent = Transform()
    child  = Transform()
    child.set_parent(parent, keep_world=False)

    parent.local_position = Vec3(10, 0, 0)
    child.local_position  = Vec3(0, 5, 0)

    # child world position follows parent
    print(child.position)   # Vec3(10, 5, 0)

    # look_at — forward points toward origin
    child.look_at(Vec3.ZERO)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from fire_engine.core.math3d import Quat, Vec3

if TYPE_CHECKING:
    pass  # no circular imports needed currently


# ---------------------------------------------------------------------------
# Space enum
# ---------------------------------------------------------------------------


class Space(Enum):
    """
    Reference-frame selector used by Transform.translate and Transform.rotate.

    SELF  — operations are expressed in the transform's own local frame.
    WORLD — operations are expressed in world space.

    Example
    -------
        t.translate(Vec3(0, 1, 0), relative_to=Space.SELF)   # move 1 m forward
        t.translate(Vec3(0, 1, 0), relative_to=Space.WORLD)  # move 1 m along world +Y
    """

    SELF = auto()
    WORLD = auto()


# ---------------------------------------------------------------------------
# 4×4 matrix helpers (pure numpy, float64 internally for precision)
# ---------------------------------------------------------------------------


def _trs_matrix(pos: Vec3, rot: Quat, scale: Vec3) -> np.ndarray:
    """
    Build a 4×4 TRS (translation × rotation × scale) matrix.

    Uses float64 internally; callers may cast if needed.

    Parameters
    ----------
    pos   : Vec3 — translation in meters
    rot   : Quat — rotation quaternion (unit)
    scale : Vec3 — scale per axis

    Returns
    -------
    np.ndarray shape (4, 4) float64
    """
    w, x, y, z = (float(c) for c in rot._data)
    sx, sy, sz = float(scale.x), float(scale.y), float(scale.z)

    # Rotation matrix from quaternion
    m = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0.0],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0.0],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Apply scale to rotation columns
    m[0, :3] *= sx
    m[1, :3] *= sy
    m[2, :3] *= sz

    # Translation
    m[0, 3] = float(pos.x)
    m[1, 3] = float(pos.y)
    m[2, 3] = float(pos.z)

    return m


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class Transform:
    """
    Hierarchy transform (parent/children), local TRS state, and derived
    world-space position/rotation.

    World-space properties are computed via the parent chain and cached with
    a dirty flag.  Any change to local TRS state or the parent pointer
    propagates the dirty flag to all descendants immediately (O(subtree size)),
    so the next world-property read triggers at most one matrix multiply per
    ancestor.

    Attributes
    ----------
    local_position : Vec3
        Position relative to the parent (or world origin if no parent).
    local_rotation : Quat
        Rotation relative to the parent (or world frame if no parent).
    local_scale    : Vec3
        Scale relative to the parent.  Default (1, 1, 1).
    parent         : Transform | None  (read via property; write via set_parent)
    children       : tuple[Transform, ...]  (read-only)

    World-space (read-only, derived)
    --------------------------------
    position : Vec3   — world-space position in meters
    rotation : Quat   — world-space rotation

    Direction vectors (world-space, from rotation)
    -----------------------------------------------
    forward : Vec3   — +Y axis in world space (facing direction)
    right   : Vec3   — +X axis in world space
    up      : Vec3   — +Z axis in world space
    """

    __slots__ = (
        "_children",
        "_dirty",
        "_local_position",
        "_local_rotation",
        "_local_scale",
        "_parent",
        "_world_matrix",  # (4,4) float64 or None when dirty
        "game_object",  # back-reference set by GameObject after construction
    )

    def __init__(self) -> None:
        self._local_position: Vec3 = Vec3(0.0, 0.0, 0.0)
        self._local_rotation: Quat = Quat.identity()
        self._local_scale: Vec3 = Vec3(1.0, 1.0, 1.0)

        self._parent: Transform | None = None
        self._children: list[Transform] = []

        self._world_matrix: np.ndarray | None = None
        self._dirty: bool = True

        self.game_object = None  # filled in by GameObject.__init__

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    @property
    def parent(self) -> Transform | None:
        """Parent transform, or None if this is a root transform."""
        return self._parent

    @property
    def children(self) -> tuple[Transform, ...]:
        """Read-only tuple of immediate child transforms."""
        return tuple(self._children)

    def set_parent(
        self,
        p: Transform | None,
        keep_world: bool = True,
    ) -> None:
        """
        Reparent this transform.

        Parameters
        ----------
        p          : Transform | None — new parent, or None to detach.
        keep_world : bool — if True (default), preserve world-space position
                     and rotation after reparenting.  If False, the local TRS
                     values stay unchanged (world-space position will jump).

        Example
        -------
            child.set_parent(parent)          # child keeps world position
            child.set_parent(None)            # detach from parent
            child.set_parent(other, False)    # keep local coords, world jumps
        """
        if p is self:
            raise ValueError("A Transform cannot be its own parent.")

        old_parent = self._parent
        _parent_changing = old_parent is not p

        # Save world TRS before reparenting (so we can restore it after)
        if keep_world and _parent_changing:
            world_pos = self.position
            world_rot = self.rotation

        # Detach from old parent
        if old_parent is not None:
            old_parent._children.remove(self)

        self._parent = p

        if p is not None:
            p._children.append(self)

        if keep_world and _parent_changing:
            # Recompute local TRS so world position/rotation is preserved.
            # self._parent is now p so position/rotation setters use the new parent.
            self.position = world_pos
            self.rotation = world_rot

        self._mark_dirty()

    # ------------------------------------------------------------------
    # Local TRS accessors
    # ------------------------------------------------------------------

    @property
    def local_position(self) -> Vec3:
        """Position relative to the parent frame (meters)."""
        return self._local_position

    @local_position.setter
    def local_position(self, value: Vec3) -> None:
        self._local_position = value
        self._mark_dirty()

    @property
    def local_rotation(self) -> Quat:
        """Rotation relative to the parent frame (unit quaternion)."""
        return self._local_rotation

    @local_rotation.setter
    def local_rotation(self, value: Quat) -> None:
        self._local_rotation = value.normalized()
        self._mark_dirty()

    @property
    def local_scale(self) -> Vec3:
        """Scale relative to the parent frame."""
        return self._local_scale

    @local_scale.setter
    def local_scale(self, value: Vec3) -> None:
        self._local_scale = value
        self._mark_dirty()

    # ------------------------------------------------------------------
    # World-space position/rotation (derived through parent chain)
    # ------------------------------------------------------------------

    def _world_mat(self) -> np.ndarray:
        """Return the cached (or freshly computed) world matrix."""
        if self._dirty or self._world_matrix is None:
            local = _trs_matrix(
                self._local_position,
                self._local_rotation,
                self._local_scale,
            )
            if self._parent is not None:
                self._world_matrix = self._parent._world_mat() @ local
            else:
                self._world_matrix = local
            self._dirty = False
        return self._world_matrix

    @property
    def position(self) -> Vec3:
        """
        World-space position in meters.

        Setting this recomputes local_position relative to the current parent.
        """
        m = self._world_mat()
        return Vec3(float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))

    @position.setter
    def position(self, world_pos: Vec3) -> None:
        if self._parent is None:
            self._local_position = world_pos
        else:
            self._local_position = self._parent.inverse_transform_point(world_pos)
        self._mark_dirty()

    @property
    def rotation(self) -> Quat:
        """
        World-space rotation (unit quaternion).

        Setting this recomputes local_rotation relative to the current parent.
        """
        if self._parent is None:
            return self._local_rotation
        return self._parent.rotation * self._local_rotation

    @rotation.setter
    def rotation(self, world_rot: Quat) -> None:
        if self._parent is None:
            self._local_rotation = world_rot.normalized()
        else:
            parent_inv = self._parent.rotation.inverse()
            self._local_rotation = (parent_inv * world_rot).normalized()
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Direction vectors (world-space)
    # ------------------------------------------------------------------

    @property
    def forward(self) -> Vec3:
        """World-space forward direction (+Y in local space, Z-up convention)."""
        return self.rotation.rotate(Vec3.FORWARD)

    @property
    def right(self) -> Vec3:
        """World-space right direction (+X in local space)."""
        return self.rotation.rotate(Vec3.RIGHT)

    @property
    def up(self) -> Vec3:
        """World-space up direction (+Z in local space)."""
        return self.rotation.rotate(Vec3.UP)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def translate(self, v: Vec3, relative_to: Space = Space.SELF) -> None:
        """
        Move the transform by vector *v*.

        Parameters
        ----------
        v           : Vec3 — displacement in meters
        relative_to : Space — Space.SELF (default) moves along local axes;
                               Space.WORLD moves along world axes.

        Example
        -------
            t.translate(Vec3(0, 1, 0))               # 1 m forward (local)
            t.translate(Vec3(0, 0, 1), Space.WORLD)  # 1 m up (world)
        """
        if relative_to is Space.SELF:
            self._local_position = self._local_position + self.rotation.rotate(v)
        else:  # Space.WORLD
            self._local_position = self._local_position + v
        self._mark_dirty()

    def rotate(self, q: Quat, relative_to: Space = Space.SELF) -> None:
        """
        Apply an additional rotation *q* to this transform.

        Parameters
        ----------
        q           : Quat — rotation to apply (unit quaternion)
        relative_to : Space — Space.SELF applies *q* in the local frame
                               (post-multiply: local_rotation = local_rotation * q).
                               Space.WORLD applies *q* in world space
                               (pre-multiply in world, then convert to local).

        Example
        -------
            t.rotate(Quat.from_axis_angle(Vec3.UP, pi/4))          # yaw 45° local
            t.rotate(Quat.from_axis_angle(Vec3.UP, pi/4), Space.WORLD)  # yaw 45° world
        """
        if relative_to is Space.SELF:
            self._local_rotation = (self._local_rotation * q).normalized()
        else:  # Space.WORLD
            # world rotation after = q * current world rotation
            new_world = q * self.rotation
            self.rotation = new_world
            return
        self._mark_dirty()

    def look_at(self, target: Vec3, up: Vec3 = Vec3.UP) -> None:
        """
        Orient this transform so that its +Y axis (forward) points toward *target*.

        Uses a right-handed look-at construction in Z-up (forward=+Y) space.

        Parameters
        ----------
        target : Vec3 — world-space position to look at (meters)
        up     : Vec3 — hint for the up direction (default Vec3.UP = +Z).
                        Degrades gracefully if forward ≈ up by using an
                        alternate reference.

        Example
        -------
            camera_transform.look_at(Vec3(10, 20, 0))
            # camera.forward ≈ direction toward (10,20,0)
        """
        world_pos = self.position
        fwd = target - world_pos
        if fwd.length < 1e-8:
            return  # target coincides with self — no change
        fwd = fwd.normalized()

        # Guard against forward ≈ up
        up_ref = up
        if abs(fwd.dot(up_ref)) > 0.999:
            # Use an alternate up reference
            if abs(fwd.dot(Vec3.FORWARD)) < 0.999:
                up_ref = Vec3.FORWARD
            else:
                up_ref = Vec3.RIGHT

        # Build orthonormal basis: in Z-up space forward=+Y, right=+X, up=+Z
        # right = fwd × up_ref  (cross: (Y)×(Z) = +X in right-hand system)
        right = fwd.cross(up_ref)
        if right.length < 1e-8:
            return
        right = right.normalized()
        true_up = fwd.cross(right)  # fwd × right = -up... need right × fwd = up? Let's use:
        # Actually: right × fwd gives the "down" so:
        # In RH system with fwd=+Y, right=+X: up = right × fwd = +X × +Y = +Z. Correct.
        true_up = (right.cross(fwd)).normalized()

        # Build rotation matrix: columns are where local axes (right/fwd/up) go in world.
        # This is a 3×3 column-major matrix:
        #   col 0 = right   (where local +X goes)
        #   col 1 = fwd     (where local +Y goes)
        #   col 2 = true_up (where local +Z goes)
        # Row-major storage (numpy default), so rows = destination per column:
        m3 = np.array(
            [
                [right.x, fwd.x, true_up.x],
                [right.y, fwd.y, true_up.y],
                [right.z, fwd.z, true_up.z],
            ],
            dtype=np.float64,
        )

        # Convert rotation matrix to quaternion
        # Using Shepperd's method
        world_rot = _mat3_to_quat(m3)
        self.rotation = world_rot

    def transform_point(self, p: Vec3) -> Vec3:
        """
        Transform a point from local space to world space.

        Parameters
        ----------
        p : Vec3 — position in local (object) space (meters)

        Returns
        -------
        Vec3 — corresponding world-space position (meters)

        Example
        -------
            # An object at (10,0,0) world, rotated 90° yaw:
            # its local +Y forward maps to world −X
            world_pt = t.transform_point(Vec3(0, 1, 0))
        """
        m = self._world_mat()
        hp = np.array([p.x, p.y, p.z, 1.0], dtype=np.float64)
        wp = m @ hp
        return Vec3(float(wp[0]), float(wp[1]), float(wp[2]))

    def inverse_transform_point(self, p: Vec3) -> Vec3:
        """
        Transform a point from world space to local space.

        Parameters
        ----------
        p : Vec3 — position in world space (meters)

        Returns
        -------
        Vec3 — corresponding local-space position (meters)

        Example
        -------
            local_pt = t.inverse_transform_point(world_pt)
            # round-trip: t.transform_point(local_pt) ≈ world_pt
        """
        m = self._world_mat()
        inv = np.linalg.inv(m)
        hp = np.array([p.x, p.y, p.z, 1.0], dtype=np.float64)
        lp = inv @ hp
        return Vec3(float(lp[0]), float(lp[1]), float(lp[2]))

    # ------------------------------------------------------------------
    # Dirty-flag propagation
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        """Mark this transform and all descendants dirty (world matrix stale)."""
        if not self._dirty:
            self._dirty = True
            for child in self._children:
                child._mark_dirty()

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pos = self._local_position
        return (
            f"Transform(pos={pos}, "
            f"children={len(self._children)}, "
            f"parent={'yes' if self._parent else 'no'})"
        )


# ---------------------------------------------------------------------------
# Helper: 3×3 rotation matrix → Quat  (Shepperd's method)
# ---------------------------------------------------------------------------


def _mat3_to_quat(m: np.ndarray) -> Quat:
    """
    Convert a 3×3 rotation matrix to a unit quaternion (Shepperd's method).

    Parameters
    ----------
    m : np.ndarray shape (3, 3) — orthonormal rotation matrix.
        Rows = output axes in the *input* frame:
            m[0] = new X-axis (right)
            m[1] = new Y-axis (forward)
            m[2] = new Z-axis (up)

    Returns
    -------
    Quat — unit quaternion

    Note
    ----
    The matrix is stored with rows as the destination axes for each local
    basis vector (i.e. the *transpose* of a column-basis matrix).
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = Quat.__new__(Quat)
    q._data = np.array([w, x, y, z], dtype=np.float32)
    return q.normalized()
