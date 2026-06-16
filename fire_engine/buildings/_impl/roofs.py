"""
buildings/_impl/roofs.py — pitched-roof geometry (gable / hip / shed).

Iteration-2 roofs (see ``docs/systems/buildings.md`` Roadmap).  ``set_roof``
now carries a :class:`~fire_engine.buildings.enums.RoofKind`; this module turns
a non-flat :class:`~fire_engine.buildings.types.RoofSlab` into solid roof
geometry on a shared :class:`~fire_engine.buildings._impl.soup.Soup`.

Generation model
----------------
Pitched roofs are built over the footprint outline's **ridge-aligned bounding
rectangle** (true-outline / concave pitched roofs are Iteration 3).  Work
happens in a 2-D frame whose ``u`` axis runs along ``ridge_dir_rad`` and ``v``
across it.  Eaves are anchored at the footprint rectangle perimeter at the
wall-top height ``top_z``; the ridge rises by ``halfspan · tan(pitch)``.  An
``overhang_m`` extends each plane outward in plan and down its own slope by
``overhang · tan(pitch)`` (so it stays coplanar — a real eave projecting past
the wall).  Each roof plane is a thin prism (``thickness_m`` deep); gable ends
and shed sides get vertical infill so the envelope is closed at the walls.

Vectorized emission via ``Soup`` (Hard Rule 4): the only loops are over a
handful of roof planes / gable ends.  Deterministic (no RNG).

Docs: docs/systems/buildings._impl.md
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.buildings._impl.soup import Soup
from fire_engine.buildings.enums import RoofKind, SurfaceMaterial
from fire_engine.buildings.types import RoofSlab

_ROOF = int(SurfaceMaterial.ROOF)


def add_roof(soup: Soup, roof: RoofSlab, top_z: float, qpq: int) -> None:
    """
    Emit ``roof`` capping the top storey at local z ``top_z`` into ``soup``.

    ``RoofKind.FLAT`` is a horizontal slab in ``[top_z, top_z + thickness_m]``
    (unchanged Iteration-1 behaviour); ``SHED`` / ``GABLE`` / ``HIP`` are
    pitched per the module docstring.  ``qpq`` is accepted for signature
    parity with the other mesh stages (roofs use the polygon outline directly).

    Docs: docs/systems/buildings._impl.md
    """
    del qpq  # roofs use the outline directly; no arc tessellation needed
    if roof.kind is RoofKind.FLAT:
        soup.add_slab(roof.polygon, top_z, top_z + roof.thickness_m, _ROOF)
        return
    frame = _ridge_frame(roof.polygon, roof.ridge_dir_rad)
    tan_p = math.tan(math.radians(roof.pitch_deg))
    args = (soup, frame, float(top_z), tan_p, float(roof.overhang_m), float(roof.thickness_m))
    if roof.kind is RoofKind.SHED:
        _shed(*args)
    elif roof.kind is RoofKind.GABLE:
        _gable(*args)
    else:  # RoofKind.HIP
        _hip(*args)


class _Frame:
    """Ridge-aligned 2-D frame + footprint bounds in ``(u, v)`` (meters)."""

    def __init__(self, uhat: np.ndarray, vhat: np.ndarray, bounds: tuple[float, ...]) -> None:
        self.uhat = uhat
        self.vhat = vhat
        self.u0, self.u1, self.v0, self.v1 = bounds

    def xyz(self, u: float, v: float, z: float) -> list[float]:
        """Plan ``(u, v)`` + height ``z`` → building-local ``[x, y, z]``."""
        xy = u * self.uhat + v * self.vhat
        return [float(xy[0]), float(xy[1]), float(z)]


def _ridge_frame(polygon: np.ndarray, ridge_dir_rad: float) -> _Frame:
    """Build the ridge-aligned frame and the footprint's ``(u, v)`` bounds."""
    uhat = np.array([math.cos(ridge_dir_rad), math.sin(ridge_dir_rad)], dtype=np.float64)
    vhat = np.array([-uhat[1], uhat[0]], dtype=np.float64)
    pts = np.asarray(polygon, dtype=np.float64)
    u = pts @ uhat
    v = pts @ vhat
    return _Frame(uhat, vhat, (float(u.min()), float(u.max()), float(v.min()), float(v.max())))


def _shed(soup: Soup, f: _Frame, top_z: float, tan_p: float, oh: float, thick: float) -> None:
    """One mono-pitch plane low at ``v0`` → high at ``v1``, with side/end infill."""
    full = (f.v1 - f.v0) * tan_p  # height gain across the footprint
    low_h = top_z - oh * tan_p
    high_h = top_z + full + oh * tan_p
    plane = np.array(
        [
            f.xyz(f.u0 - oh, f.v0 - oh, low_h),
            f.xyz(f.u1 + oh, f.v0 - oh, low_h),
            f.xyz(f.u1 + oh, f.v1 + oh, high_h),
            f.xyz(f.u0 - oh, f.v1 + oh, high_h),
        ]
    )
    soup.add_prism(plane, thick, _ROOF)
    # Triangular side-wall infill at each footprint u edge (rises v0→v1).
    for u, sign in ((f.u0, -1.0), (f.u1, 1.0)):
        tri = np.array(
            [[f.xyz(u, f.v0, top_z), f.xyz(u, f.v1, top_z), f.xyz(u, f.v1, top_z + full)]]
        )
        soup.add_tris(tri, np.array([[sign * f.uhat[0], sign * f.uhat[1], 0.0]]))
    # Vertical infill closing the tall (v1) end.
    quad = np.array(
        [
            [
                f.xyz(f.u0, f.v1, top_z),
                f.xyz(f.u1, f.v1, top_z),
                f.xyz(f.u1, f.v1, top_z + full),
                f.xyz(f.u0, f.v1, top_z + full),
            ]
        ]
    )
    soup.add_quads(quad, np.array([[f.vhat[0], f.vhat[1], 0.0]]))


def _gable(soup: Soup, f: _Frame, top_z: float, tan_p: float, oh: float, thick: float) -> None:
    """Two planes meeting at a central ridge (along u), with gable-end infill."""
    vmid = 0.5 * (f.v0 + f.v1)
    half = 0.5 * (f.v1 - f.v0)
    ridge_h = top_z + half * tan_p
    eave_h = top_z - oh * tan_p
    uo0, uo1 = f.u0 - oh, f.u1 + oh
    low = np.array(
        [
            f.xyz(uo0, vmid, ridge_h),
            f.xyz(uo1, vmid, ridge_h),
            f.xyz(uo1, f.v0 - oh, eave_h),
            f.xyz(uo0, f.v0 - oh, eave_h),
        ]
    )
    high = np.array(
        [
            f.xyz(uo0, vmid, ridge_h),
            f.xyz(uo0, f.v1 + oh, eave_h),
            f.xyz(uo1, f.v1 + oh, eave_h),
            f.xyz(uo1, vmid, ridge_h),
        ]
    )
    soup.add_prism(low, thick, _ROOF)
    soup.add_prism(high, thick, _ROOF)
    # Gable infill triangles at the footprint u edges (eave top_z → ridge).
    for u, sign in ((f.u0, -1.0), (f.u1, 1.0)):
        tri = np.array([[f.xyz(u, f.v0, top_z), f.xyz(u, f.v1, top_z), f.xyz(u, vmid, ridge_h)]])
        soup.add_tris(tri, np.array([[sign * f.uhat[0], sign * f.uhat[1], 0.0]]))


def _hip(soup: Soup, f: _Frame, top_z: float, tan_p: float, oh: float, thick: float) -> None:
    """Four planes (two trapezoids + two end triangles) on a shortened ridge."""
    vmid = 0.5 * (f.v0 + f.v1)
    half = 0.5 * (f.v1 - f.v0)
    ridge_h = top_z + half * tan_p
    eave_h = top_z - oh * tan_p
    # Ridge inset from the u ends by `half` so the hip ends slope at the pitch;
    # if the footprint is narrower in u than v it degenerates to a pyramid.
    ur0, ur1 = f.u0 + half, f.u1 - half
    if ur1 < ur0:
        ur0 = ur1 = 0.5 * (f.u0 + f.u1)
    uo0, uo1 = f.u0 - oh, f.u1 + oh
    vo0, vo1 = f.v0 - oh, f.v1 + oh
    low = np.array(
        [
            f.xyz(ur0, vmid, ridge_h),
            f.xyz(ur1, vmid, ridge_h),
            f.xyz(uo1, vo0, eave_h),
            f.xyz(uo0, vo0, eave_h),
        ]
    )
    high = np.array(
        [
            f.xyz(ur0, vmid, ridge_h),
            f.xyz(uo0, vo1, eave_h),
            f.xyz(uo1, vo1, eave_h),
            f.xyz(ur1, vmid, ridge_h),
        ]
    )
    end_a = np.array([f.xyz(ur0, vmid, ridge_h), f.xyz(uo0, vo0, eave_h), f.xyz(uo0, vo1, eave_h)])
    end_b = np.array([f.xyz(ur1, vmid, ridge_h), f.xyz(uo1, vo1, eave_h), f.xyz(uo1, vo0, eave_h)])
    for poly in (low, high, end_a, end_b):
        soup.add_prism(poly, thick, _ROOF)
