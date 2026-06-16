"""
render/sky/_impl/sky_geom — sky-dome geometry builders, texture helpers, and constants.

Module-level utilities extracted from ``sky_renderer`` to satisfy the 500-line
limit (C0302).  Imported by ``sky_build`` (which contains the component build
functions) and re-exported through ``sky_build`` for backward compat.

No circular imports: this module never imports from sky_renderer or sky_build.

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from panda3d.core import (
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    Texture,
)

from fire_engine.core import get_logger
from fire_engine.core.rng import for_domain

__all__ = [
    "_CAMERA_FAR_M",
    "_DEFAULT_STAR_COUNT",
    "_DOME_RADIUS_M",
    "_DOME_SLICES",
    "_DOME_STACKS",
    "_VCLOUD_ALT_M",
    "_VCLOUD_DETAIL_SIZE",
    "_VCLOUD_DETAIL_STR",
    "_VCLOUD_DETAIL_TILE_M",
    "_VCLOUD_HG",
    "_VCLOUD_LIGHT_STEP_M",
    "_VCLOUD_SHAPE_SIZE",
    "_VCLOUD_SHAPE_TILE_M",
    "_VCLOUD_SIGMA",
    "_VCLOUD_THICK_M",
    "_build_dome_node",
    "_clamp01",
    "_fallback_moon",
    "_fallback_star_cube",
    "_load_or_bake_cloud_noise",
    "_make_geom_node",
    "_sky_texture",
]

_log = get_logger("world.sky_renderer")

# ---------------------------------------------------------------------------
# Renderer tuning constants (visual tuning, not world config — config-backed
# values [cloud altitude/thickness/cell, star count] come from core.config).
# ---------------------------------------------------------------------------

_DOME_RADIUS_M: float = 800.0  # sky-dome sphere radius (meters)
_DOME_STACKS: int = 24  # dome latitude divisions
_DOME_SLICES: int = 48  # dome longitude divisions
_CAMERA_FAR_M: float = 4000.0  # minimum camera far plane (meters)
# Volumetric cloud layer (raymarched; replaces the boxy slab quads).
_VCLOUD_ALT_M: float = 500.0  # cloud slab bottom altitude (world Z, m)
_VCLOUD_THICK_M: float = 400.0  # slab thickness (m)
_VCLOUD_SHAPE_TILE_M: float = 3000.0  # world span of one shape-noise tile (m)
_VCLOUD_DETAIL_TILE_M: float = 320.0  # world span of one detail-noise tile (m)
_VCLOUD_DETAIL_STR: float = 0.22  # edge-erosion strength from the detail vol
_VCLOUD_SIGMA: float = 0.09  # extinction per meter at full density
_VCLOUD_LIGHT_STEP_M: float = 28.0  # sun light-march step length (m)
_VCLOUD_HG: float = 0.62  # Henyey-Greenstein anisotropy (fwd scatter)
_VCLOUD_SHAPE_SIZE: int = 64  # baked shape volume edge (voxels)
_VCLOUD_DETAIL_SIZE: int = 32  # baked detail volume edge (voxels)

_DEFAULT_STAR_COUNT: int = 2500


# ---------------------------------------------------------------------------
# Bulk geometry builders (numpy → one memoryview write, Hard Rule 7)
# ---------------------------------------------------------------------------


def _make_geom_node(
    vertex_block: np.ndarray, fmt: GeomVertexFormat, indices: np.ndarray, name: str
) -> GeomNode:
    """
    Build a GeomNode from an interleaved float32 vertex block + uint32 indices.

    Parameters
    ----------
    vertex_block : np.ndarray
        ``(N, K) float32`` — rows must exactly match *fmt*'s interleaved layout
        (e.g. K=3 for ``get_v3()``, K=5 for ``get_v3t2()``).
    fmt : GeomVertexFormat
        A registered single-array format.
    indices : np.ndarray
        ``(M,) uint32`` triangle indices.
    name : str
        Node/vertex-data name (debugging).

    Returns
    -------
    panda3d.core.GeomNode
        One bulk memoryview write per buffer — no per-vertex loops.
    """
    block = np.ascontiguousarray(vertex_block, dtype=np.float32)
    n_verts = int(block.shape[0])

    vdata = GeomVertexData(name, fmt, Geom.UH_static)
    vdata.set_num_rows(n_verts)
    varray = vdata.modify_array(0)
    view = memoryview(varray).cast("B")
    view[:] = memoryview(block.data).cast("B")

    prim = GeomTriangles(Geom.UH_static)
    prim.set_index_type(GeomEnums.NT_uint32)
    idx = np.ascontiguousarray(indices, dtype=np.uint32)
    iarray = prim.modify_vertices()
    iarray.set_num_rows(int(idx.shape[0]))
    iview = memoryview(iarray).cast("B")
    iview[:] = memoryview(idx.data).cast("B")

    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode(name)
    node.add_geom(geom)
    return node


def _build_dome_node(radius_m: float, stacks: int, slices: int) -> GeomNode:
    """
    Build an inverted (inward-facing) UV-sphere GeomNode for the sky dome.

    Vertex positions double as view directions in the dome shader (the dome
    follows the camera by translation only), so the format is position-only
    (``get_v3``).  Winding is verified numerically and flipped if needed so
    the inside of the sphere is front-facing under default backface culling.

    Parameters
    ----------
    radius_m : float — sphere radius in meters.
    stacks   : int   — latitude divisions (>= 3).
    slices   : int   — longitude divisions (>= 3).
    """
    phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, stacks + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, slices + 1)
    pgrid, tgrid = np.meshgrid(phi, theta, indexing="ij")
    pos = np.stack(
        [
            np.cos(pgrid) * np.cos(tgrid),
            np.cos(pgrid) * np.sin(tgrid),
            np.sin(pgrid),
        ],
        axis=-1,
    ).reshape(-1, 3).astype(np.float32) * np.float32(radius_m)

    i = np.arange(stacks)[:, None]
    j = np.arange(slices)[None, :]
    v00 = (i * (slices + 1) + j).astype(np.uint32)
    v10 = v00 + np.uint32(slices + 1)
    v01 = v00 + np.uint32(1)
    v11 = v10 + np.uint32(1)
    tris = np.stack([v00, v01, v11, v00, v11, v10], axis=-1).reshape(-1, 3)

    # Ensure inward winding: find one non-degenerate triangle, test its normal
    # against the outward radial; flip every triangle if it faces outward.
    for tri in tris:
        a, b, c = pos[tri[0]], pos[tri[1]], pos[tri[2]]
        nrm = np.cross(b - a, c - a)
        if float(np.dot(nrm, nrm)) > 1e-8:
            centroid = (a + b + c) / 3.0
            if float(np.dot(nrm, centroid)) > 0.0:  # outward → flip
                tris = tris[:, ::-1]
            break

    return _make_geom_node(pos, GeomVertexFormat.get_v3(), tris.reshape(-1), "sky_dome")


def _load_or_bake_cloud_noise(
    seed: int, shape_size: int, detail_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the baked volumetric-cloud noise volumes, baking + caching on a miss.

    The bake (``sky.cloud_noise``) is deterministic in the world seed, so a
    disk cache keyed by ``(seed, size, version)`` is always valid — the ~1.7 s
    64³ bake then happens only on the very first run for a seed.  Cache lives
    under ``saves/cloud_cache/`` (gitignored); any I/O failure silently falls
    back to baking in-process (never fatal).

    Returns ``(shape_arr, detail_arr)`` — both ``(N,N,N,4) uint8``.
    """
    from fire_engine.world.sky.cloud_noise import bake_detail_noise, bake_shape_noise

    cache_dir = Path("saves") / "cloud_cache"
    version = 1

    def _load_or(kind: str, size: int, baker: Callable[[int], np.ndarray]) -> np.ndarray:
        path = cache_dir / f"{kind}_{seed}_{size}_v{version}.npy"
        try:
            if path.exists():
                return np.asarray(np.load(path))
        except Exception as exc:
            _log.warning("cloud noise cache read failed (%s); rebaking", exc)
        arr: np.ndarray = baker(size)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
        except Exception as exc:
            _log.debug("cloud noise cache write failed: %s", exc)
        return arr

    return (
        _load_or("cloud_shape", shape_size, bake_shape_noise),
        _load_or("cloud_detail", detail_size, bake_detail_noise),
    )


def _fallback_star_cube(star_count: int) -> np.ndarray:
    """
    Deterministic stand-in for the registry ``"night_sky_cube"`` def: six
    64² faces of deep-indigo floor + point stars; alpha = luminance.

    Used only when ``procedural.get("night_sky_cube")`` is unavailable.
    All randomness via ``for_domain``.
    """
    rng = for_domain("sky", "star_cube_fallback")
    size = 64
    rgb = np.full((6, size, size, 3), 0.012, dtype=np.float32)
    rgb[..., 2] = 0.035
    n = max(int(star_count), 1)
    face = rng.integers(0, 6, n)
    row = rng.integers(0, size, n)
    col = rng.integers(0, size, n)
    b = (rng.random(n).astype(np.float32) ** 3) * 0.8 + 0.08
    np.maximum.at(rgb, (face, row, col), np.repeat(b[:, None], 3, axis=1))
    out = np.empty((6, size, size, 4), dtype=np.uint8)
    out[..., :3] = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[..., 3] = out[..., :3].max(axis=-1)
    return out


def _fallback_moon() -> np.ndarray:
    """
    Deterministic stand-in for the registry ``"moon_surface"`` def: a flat
    pale-gray 64x64 disc (alpha 255 inside the unit circle, 0 outside).
    """
    size = 64
    ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    disc = (xx * xx + yy * yy) <= 1.0
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 0] = 168
    out[..., 1] = 166
    out[..., 2] = 158
    out[..., 3] = np.where(disc, 255, 0).astype(np.uint8)
    return out


def _sky_texture(name: str, fallback: np.ndarray | None = None) -> Texture:
    """
    Fetch a procedural sky texture by registry *name* and bridge it to Panda3D.

    Falls back to *fallback* (already-generated RGBA array) with a logged
    warning if the registry def is missing — keeps the renderer working while
    the headless sky package is still landing.
    """
    rgba: Any = None
    try:
        from fire_engine.procedural import get as get_procedural

        rgba = get_procedural(name)
    except Exception as exc:
        _log.warning("procedural texture %r unavailable (%s) — using fallback", name, exc)
    if rgba is None:
        if fallback is None:
            raise RuntimeError(f"no texture and no fallback for {name!r}")
        rgba = fallback
    from fire_engine.render.bridges.texture_bridge import to_panda_texture

    return to_panda_texture(rgba)


def _clamp01(x: float) -> float:
    """Clamp a float to [0, 1]."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
