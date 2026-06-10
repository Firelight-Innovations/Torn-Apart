"""
world/sky_renderer.py — SkyRendererComponent: the render half of the sky system.

Draws everything the headless ``torn_apart.sky`` simulation describes through
its per-frame ``SkyState``:

* **Sky dome** — an inverted UV-sphere (radius ~800 m) camera-centred via
  translation-only follow, painted by ``sky_shaders.SKY_DOME_FRAGMENT``:
  atmosphere gradient, sun disc + halo, moon with phase terminator, the
  ``"night_sky"`` equirect galaxy texture with per-star twinkle, slow star
  rotation, and deterministic shooting stars.
* **Boxy raymarched clouds** — two camera-following horizontal quads (slab
  bottom + top) DDA-raymarching a grid of ``sky_cloud_cell_m`` cells in the
  slab ``[sky_cloud_altitude_m, +sky_cloud_thickness_m]``; crisp Minecraft-style
  boxes, flat-face lit, wind-drifted, correct from below/inside/above.
* **Rain** — three nested camera-following open cylinders textured with the
  ``"rain_streak"`` texture, UV-scrolled at per-layer rates for parallax,
  additively blended, hidden when ``rain_intensity < 0.05``.
* **Fog + global light** — exponential ``panda3d.core.Fog`` on the terrain
  root (density ``SkyState.fog_density`` 1/m, colour ``fog_color``), window
  clear colour blended toward fog, and ``terrain_root.set_color_scale`` driven
  by ``SkyState.terrain_light_scale`` (day/night modulation of the baked
  vertex sunlight).

Units: meters / seconds / radians, world Z-up.  All scene-graph geometry is
built ONCE in ``start()`` with bulk numpy → memoryview writes (Hard Rule 7);
per-frame work is a fixed handful of ``set_shader_input`` / ``set_pos`` /
``set_color_scale`` calls.

This module imports panda3d and therefore lives in ``world/`` (Hard Rule 1).
``torn_apart.world.__init__`` exports ``SkyRendererComponent`` behind the same
try/except guard as the other panda3d bridges.

Example
-------
::

    from torn_apart.sky import SkySystem
    from torn_apart.world import instantiate
    from torn_apart.world.sky_renderer import SkyRendererComponent

    sky_system = SkySystem(cfg, clock, bus)
    sky_go = instantiate()
    sky_go.name = "Sky"
    sky_go.add_component(
        SkyRendererComponent,
        base=app,                       # the world.app.App (ShowBase)
        sky_system=sky_system,          # drives + reads SkyState
        terrain_root=app.terrain_root,  # receives fog + light colour scale
        clock=clock,                    # optional; shooting-star scheduling
    )
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Panda3D imports are allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    ColorBlendAttrib,
    Fog,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    LVecBase2f,
    LVecBase3f,
    LVecBase4f,
    SamplerState,
    Shader,
    Texture,
    TextureStage,
    TransparencyAttrib,
)

from torn_apart.core import get_logger
from torn_apart.core.rng import for_domain
from torn_apart.world.component import Component
from torn_apart.world import sky_shaders

__all__ = ["SkyRendererComponent"]

_log = get_logger("world.sky_renderer")

# ---------------------------------------------------------------------------
# Renderer tuning constants (visual tuning, not world config — config-backed
# values [cloud altitude/thickness/cell, star count] come from core.config).
# ---------------------------------------------------------------------------

_DOME_RADIUS_M:        float = 800.0   # sky-dome sphere radius (meters)
_DOME_STACKS:          int   = 24      # dome latitude divisions
_DOME_SLICES:          int   = 48      # dome longitude divisions
_CAMERA_FAR_M:         float = 4000.0  # minimum camera far plane (meters)
_CLOUD_QUAD_SIDE_M:    float = 2400.0  # cloud slab quad side length (meters)
_CLOUD_FADE_FRACTION:  float = 0.92    # fade distance as a fraction of half-side

# Rain layers: (cylinder radius m, scroll-rate multiplier).  Different scroll
# rates per layer give cheap parallax depth.
_RAIN_LAYERS: tuple[tuple[float, float], ...] = ((4.0, 1.6), (7.0, 1.15), (11.0, 0.85))
_RAIN_HEIGHT_M:        float = 14.0    # cylinder height (meters, camera-centred)
_RAIN_SEGMENTS:        int   = 32      # cylinder circumference divisions
_RAIN_TEX_METERS_U:    float = 3.0     # streak texture horizontal world span (m)
_RAIN_TEX_METERS_V:    float = 12.0    # streak texture vertical world span (m)
_RAIN_HIDE_THRESHOLD:  float = 0.05    # rain_intensity below which rain hides
_RAIN_BASE_SCROLL:     float = 1.4     # base UV scroll rate (v-units / s)
_RAIN_MAX_TILT_DEG:    float = 14.0    # max wind tilt of the rain cylinders

# Shooting stars: deterministic schedule (see _update_shooting_star).
_SS_SLOT_GAME_S:       float = 1800.0  # one slot = 30 game-minutes (game seconds)
_SS_DURATION_REAL_S:   float = 1.2     # streak animation length (real seconds)
_SS_SPAWN_P:           float = 0.5     # spawn probability per slot
_SS_MIN_STAR_VIS:      float = 0.5     # only spawn when stars are visible

_GAME_SECONDS_PER_DAY: float = 86400.0

# Contract defaults for the sky config keys (the sky package adds them to
# core.config; fall back to the frozen-contract values when running against a
# pre-sky Config so the renderer works mid-integration).
_DEFAULT_CLOUD_ALTITUDE_M:  float = 96.0
_DEFAULT_CLOUD_THICKNESS_M: float = 8.0
_DEFAULT_CLOUD_CELL_M:      float = 12.0
_DEFAULT_STAR_COUNT:        int   = 2500


# ---------------------------------------------------------------------------
# Bulk geometry builders (numpy → one memoryview write, Hard Rule 7)
# ---------------------------------------------------------------------------

def _make_geom_node(vertex_block: np.ndarray,
                    fmt: GeomVertexFormat,
                    indices: np.ndarray,
                    name: str) -> GeomNode:
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
    view[:] = memoryview(block).cast("B")

    prim = GeomTriangles(Geom.UH_static)
    prim.set_index_type(GeomEnums.NT_uint32)
    idx = np.ascontiguousarray(indices, dtype=np.uint32)
    iarray = prim.modify_vertices()
    iarray.set_num_rows(int(idx.shape[0]))
    iview = memoryview(iarray).cast("B")
    iview[:] = memoryview(idx).cast("B")

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
            if float(np.dot(nrm, centroid)) > 0.0:   # outward → flip
                tris = tris[:, ::-1]
            break

    return _make_geom_node(pos, GeomVertexFormat.get_v3(),
                           tris.reshape(-1), "sky_dome")


def _build_cloud_node(side_m: float, thickness_m: float) -> GeomNode:
    """
    Build the cloud-layer coverage geometry: two horizontal quads in one
    GeomNode — model z=0 (slab bottom) and z=*thickness_m* (slab top).  The
    owning NodePath is placed at z = ``sky_cloud_altitude_m`` so world heights
    line up with the raymarch slab; the fragment shader identifies the plane
    from the interpolated world z and discards duplicate coverage.

    Parameters
    ----------
    side_m      : float — quad side length in meters (camera-following in XY).
    thickness_m : float — slab thickness in meters (top-quad model height).
    """
    h = side_m * 0.5
    t = thickness_m
    pos = np.array(
        [
            [-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0],
            [-h, -h, t],   [h, -h, t],   [h, h, t],   [-h, h, t],
        ],
        dtype=np.float32,
    )
    idx = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=np.uint32)
    return _make_geom_node(pos, GeomVertexFormat.get_v3(), idx, "cloud_layer")


def _build_rain_cylinder(radius_m: float, height_m: float,
                         segments: int) -> GeomNode:
    """
    Build one open (uncapped) vertical cylinder for a rain layer.

    Centred on the model origin (z spans ±height/2); UVs tile the
    ``"rain_streak"`` texture so streaks are ~world-scaled: u covers
    ``_RAIN_TEX_METERS_U`` meters of circumference per tile, v covers
    ``_RAIN_TEX_METERS_V`` meters of height per tile.  Rendered two-sided.

    V orientation (do NOT "simplify"): ``texture_bridge.to_panda_texture``
    vertically flips the numpy array on upload (docs/systems/world.md gotcha
    #8).  The ``"rain_streak"`` def paints each streak's bright head with the
    motion-blur tail at HIGHER array rows, so post-flip the tail sits at
    LOWER texture v.  We therefore map **v = v_tiles at the cylinder bottom
    and v = 0 at the top**: world-up = decreasing v puts the tail ABOVE the
    bright head (correct falling-rain look), and a DECREASING per-frame V
    offset (see ``_update_rain``) translates the pattern DOWNWARD.

    Parameters
    ----------
    radius_m : float — cylinder radius in meters.
    height_m : float — cylinder height in meters.
    segments : int   — circumference divisions.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    x = (radius_m * np.cos(theta)).astype(np.float32)
    y = (radius_m * np.sin(theta)).astype(np.float32)
    u_tiles = (2.0 * np.pi * radius_m) / _RAIN_TEX_METERS_U
    v_tiles = height_m / _RAIN_TEX_METERS_V
    u = np.linspace(0.0, u_tiles, segments + 1).astype(np.float32)

    n = segments + 1
    block = np.empty((2 * n, 5), dtype=np.float32)   # [x, y, z, u, v]
    block[:n, 0] = x
    block[:n, 1] = y
    block[:n, 2] = -0.5 * height_m
    block[:n, 3] = u
    block[:n, 4] = v_tiles          # bottom ring = HIGH v (see docstring)
    block[n:, 0] = x
    block[n:, 1] = y
    block[n:, 2] = 0.5 * height_m
    block[n:, 3] = u
    block[n:, 4] = 0.0              # top ring = v 0

    j = np.arange(segments, dtype=np.uint32)
    b0, b1 = j, j + 1
    t0, t1 = j + n, j + 1 + n
    idx = np.stack([b0, b1, t1, b0, t1, t0], axis=-1).reshape(-1)
    return _make_geom_node(block, GeomVertexFormat.get_v3t2(), idx, "rain_layer")


# ---------------------------------------------------------------------------
# Texture acquisition (registry first, deterministic fallback second)
# ---------------------------------------------------------------------------

def _cloud_value_quantiles(seed: float, sample_cells: int = 192) -> np.ndarray:
    """
    Empirical quantile table of the cloud shader's per-cell occupancy value.

    The GLSL ``cell_value`` (2-octave value noise + per-cell hash; see
    ``sky_shaders.CLOUD_FRAGMENT``) is bell-shaped around ~0.5, NOT uniform —
    thresholding it directly with ``cloud_coverage`` would give almost no
    clouds below ~0.3 coverage.  This numpy float32 port evaluates the same
    formulas over a ``sample_cells``² cell grid and returns the SORTED values;
    indexing the array at ``coverage * (n-1)`` yields the threshold that makes
    *coverage* the actual fill fraction.

    Parameters
    ----------
    seed : float
        The same world-seed-derived value passed to the shader as ``u_seed``.
    sample_cells : int
        Sample grid edge (cells); 192² ≈ 37k samples is plenty for quantiles.

    Returns
    -------
    np.ndarray — sorted float64 occupancy values (ascending).
    """
    s = np.float32(seed)
    c33 = np.float32(33.33)

    def hash21(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # GLSL: p3 = fract(p.xyx * 0.1031 + seed); p3 += dot(p3, p3.yzx + 33.33);
        #       return fract((p3.x + p3.y) * p3.z);
        px = x * np.float32(0.1031) + s
        px -= np.floor(px)
        py = y * np.float32(0.1031) + s
        py -= np.floor(py)
        pz = px                                   # p.xyx → z component == x component
        d = px * (py + c33) + py * (pz + c33) + pz * (px + c33)
        r = ((px + d) + (py + d)) * (pz + d)
        return r - np.floor(r)

    def vnoise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix, iy = np.floor(x), np.floor(y)
        fx, fy = x - ix, y - iy
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        a = hash21(ix, iy)
        b = hash21(ix + 1.0, iy)
        c = hash21(ix, iy + 1.0)
        e = hash21(ix + 1.0, iy + 1.0)
        return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + \
               (c * (1.0 - fx) + e * fx) * fy

    half = sample_cells // 2
    cx, cy = np.meshgrid(
        np.arange(-half, half, dtype=np.float32),
        np.arange(-half, half, dtype=np.float32),
        indexing="ij",
    )
    vals = (0.55 * vnoise(cx / np.float32(6.0), cy / np.float32(6.0))
            + 0.30 * vnoise(cx / np.float32(2.2), cy / np.float32(2.2))
            + 0.15 * hash21(cx + np.float32(17.0), cy + np.float32(17.0)))
    return np.sort(vals.ravel().astype(np.float64))


def _fallback_night_sky(star_count: int) -> np.ndarray:
    """
    Deterministic stand-in for the registry ``"night_sky"`` def: a 1024x512
    equirect RGBA galaxy band + *star_count* point stars; alpha = luminance.

    Used only when ``procedural.get("night_sky")`` is unavailable (the sky
    package not yet landed).  All randomness via ``for_domain``.
    """
    rng = for_domain("sky", "night_sky_fallback")
    height, width = 512, 1024
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    yy = np.arange(height, dtype=np.float32)[:, None]
    xx = np.arange(width, dtype=np.float32)[None, :]
    band_center = height * 0.5 + (height * 0.22) * np.sin(
        xx / width * 2.0 * np.pi + 1.3)
    band = np.exp(-(((yy - band_center) / (height * 0.10)) ** 2))
    mottle = np.kron(rng.random((height // 8, width // 8)).astype(np.float32),
                     np.ones((8, 8), dtype=np.float32))
    band *= 0.45 + 0.75 * mottle
    rgb[..., 0] += band * 0.12
    rgb[..., 1] += band * 0.12
    rgb[..., 2] += band * 0.19

    sx = rng.integers(0, width, star_count)
    sy = rng.integers(0, height, star_count)
    brightness = (rng.random(star_count).astype(np.float32) ** 3) * 0.9 + 0.08
    rgb[sy, sx, :] = np.maximum(rgb[sy, sx, :], brightness[:, None])

    rgb = np.clip(rgb, 0.0, 1.0)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., :3] = (rgb * 255.0).astype(np.uint8)
    out[..., 3] = (rgb.max(axis=-1) * 255.0).astype(np.uint8)   # alpha = luminance
    return out


def _fallback_rain_streak() -> np.ndarray:
    """
    Deterministic stand-in for the registry ``"rain_streak"`` def: a 128x512
    vertically tileable RGBA texture of faint vertical streaks.
    """
    rng = for_domain("sky", "rain_streak_fallback")
    height, width = 512, 128
    seeds = ((rng.random((height, width)) < 0.0045).astype(np.float32)
             * (0.35 + 0.65 * rng.random((height, width)).astype(np.float32)))
    streak = np.zeros_like(seeds)
    length = 40
    for k in range(length):                      # 40 bulk array ops, tileable via roll
        streak += np.roll(seeds, k, axis=0) * (1.0 - k / float(length))
    streak = np.clip(streak * 0.9, 0.0, 1.0)
    v = (streak * 255.0).astype(np.uint8)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., 0] = (streak * 0.72 * 255.0).astype(np.uint8)
    out[..., 1] = (streak * 0.80 * 255.0).astype(np.uint8)
    out[..., 2] = v
    out[..., 3] = v
    return out


def _sky_texture(name: str, fallback: np.ndarray | None = None) -> Texture:
    """
    Fetch a procedural sky texture by registry *name* and bridge it to Panda3D.

    Falls back to *fallback* (already-generated RGBA array) with a logged
    warning if the registry def is missing — keeps the renderer working while
    the headless sky package is still landing.
    """
    rgba: np.ndarray | None = None
    try:
        from torn_apart.procedural import get as get_procedural
        rgba = get_procedural(name)
    except Exception as exc:  # noqa: BLE001 — registry may predate the sky defs
        _log.warning("procedural texture %r unavailable (%s) — using fallback",
                     name, exc)
    if rgba is None:
        if fallback is None:
            raise RuntimeError(f"no texture and no fallback for {name!r}")
        rgba = fallback
    from torn_apart.world.texture_bridge import to_panda_texture
    return to_panda_texture(rgba)


def _clamp01(x: float) -> float:
    """Clamp a float to [0, 1]."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ---------------------------------------------------------------------------
# SkyRendererComponent
# ---------------------------------------------------------------------------

class SkyRendererComponent(Component):
    """
    Render component for the procedural sky + weather system.

    Lifecycle
    ---------
    * ``start()`` — builds all scene-graph nodes ONCE (dome, cloud quads, rain
      cylinders, Fog object), compiles the GLSL shaders, uploads textures, and
      extends the camera far plane to cover the sky geometry.
    * ``update(dt)`` — drives ``sky_system.update()`` (the registry runs all
      updates before any late_update, so the state is fresh for every reader
      this frame without touching App).
    * ``late_update(dt)`` — reads the frame's ``SkyState`` and writes the
      per-frame render state: a fixed handful of ``set_shader_input`` /
      ``set_pos`` / ``set_color_scale`` calls (never per-element loops).

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App (ShowBase)
        The application — provides ``render``, ``camLens``, ``camera_go``,
        ``set_background_color`` and (optionally) ``_config``.
    sky_system : torn_apart.sky.SkySystem (or duck-typed stub)
        ``update() -> SkyState`` once per frame; only SkyState attributes are
        read, so any object with the contract fields works (see
        ``tools/screenshot.py --stub-sky``).
    terrain_root : panda3d NodePath
        Receives the exponential Fog and the day/night ``set_color_scale``
        (baked vertex sunlight × global scale = the lighting integration).
    clock : core.clock.Clock | None
        Optional; used for star rotation + the deterministic shooting-star
        schedule.  Defaults to ``sky_system.clock`` / ``sky_system._clock``
        when available, else those features idle.

    Units: meters, seconds, radians.  All directions world-space Z-up.
    """

    def __init__(self, base: Any = None, sky_system: Any = None,
                 terrain_root: Any = None, clock: Any = None) -> None:
        super().__init__()
        self.base = base
        self.sky_system = sky_system
        self.terrain_root = terrain_root
        self.clock = clock
        if self.clock is None and sky_system is not None:
            self.clock = getattr(sky_system, "clock", None) or \
                getattr(sky_system, "_clock", None)

        # Per-frame state
        self._state: Any = None            # last SkyState from sky_system.update()
        self._time_s: float = 0.0          # real seconds since start (twinkle)
        self._wind_x_m: float = 0.0        # accumulated wind drift (meters)
        self._wind_y_m: float = 0.0

        # Scene-graph nodes (built in start())
        self._dome_np = None
        self._cloud_np = None
        self._rain_root = None
        self._rain_layers: list = []       # (NodePath, scroll_mult)
        self._rain_scroll: list[float] = []
        self._rain_visible: bool = False
        self._fog: Fog | None = None

        # coverage → threshold quantiles (filled by _build_clouds; identity ramp
        # until then so late_update is safe pre-start)
        self._cloud_quantiles: np.ndarray = np.linspace(0.0, 1.0, 2)

        # Cloud slab parameters (resolved from config in start())
        self._cloud_alt_m: float = _DEFAULT_CLOUD_ALTITUDE_M
        self._cloud_thick_m: float = _DEFAULT_CLOUD_THICKNESS_M
        self._cloud_cell_m: float = _DEFAULT_CLOUD_CELL_M

        # Shooting-star animation state
        self._ss_slot: tuple[int, int] | None = None   # (game_day, slot)
        self._ss_progress: float = -1.0                # < 0 → inactive

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build all sky scene-graph nodes, shaders, and textures (once)."""
        if self.base is None:
            _log.warning("SkyRendererComponent.start: no base — disabled")
            self.enabled = False
            return

        cfg = getattr(self.base, "_config", None)
        self._cloud_alt_m = float(getattr(cfg, "sky_cloud_altitude_m",
                                          _DEFAULT_CLOUD_ALTITUDE_M))
        self._cloud_thick_m = float(getattr(cfg, "sky_cloud_thickness_m",
                                            _DEFAULT_CLOUD_THICKNESS_M))
        self._cloud_cell_m = float(getattr(cfg, "sky_cloud_cell_m",
                                           _DEFAULT_CLOUD_CELL_M))
        star_count = int(getattr(cfg, "sky_star_count", _DEFAULT_STAR_COUNT))

        # Far plane must cover the dome (800 m) and the cloud-quad corners
        # (~1.7 km horizontally).
        lens = self.base.camLens
        if lens.get_far() < _CAMERA_FAR_M:
            lens.set_far(_CAMERA_FAR_M)

        self._build_dome(star_count)
        self._build_clouds()
        self._build_rain()

        # Exponential fog on the terrain (density/colour driven per frame).
        self._fog = Fog("sky_fog")
        self._fog.set_exp_density(0.0008)
        if self.terrain_root is not None:
            self.terrain_root.set_fog(self._fog)

        _log.info("Sky renderer ready (cloud slab z=[%.0f, %.0f] m, cell=%.0f m)",
                  self._cloud_alt_m, self._cloud_alt_m + self._cloud_thick_m,
                  self._cloud_cell_m)

    def update(self, dt: float) -> None:
        """
        Advance the sky simulation for this frame.

        The component registry runs ALL update() calls before any
        late_update(), so calling ``sky_system.update()`` here guarantees a
        fresh ``SkyState`` for ``late_update`` without modifying App.
        """
        self._time_s += dt
        if self.sky_system is not None:
            self._state = self.sky_system.update()

    def late_update(self, dt: float) -> None:
        """Write this frame's SkyState to the GPU (bulk uniform/state writes)."""
        st = self._state if self._state is not None else \
            getattr(self.sky_system, "state", None)
        if st is None or self._dome_np is None:
            return

        cx, cy, cz = self._camera_pos()

        # Wind drift accumulates in meters; the cloud shader shifts its grid by it.
        wx, wy = st.wind_dir
        self._wind_x_m += float(wx) * float(st.wind_speed) * dt
        self._wind_y_m += float(wy) * float(st.wind_speed) * dt

        self._update_dome(st, cx, cy, cz)
        self._update_shooting_star(st, dt)
        self._update_clouds(st, cx, cy, cz)
        self._update_rain(st, cx, cy, cz, dt)
        self._update_fog_and_light(st)

    def on_destroy(self) -> None:
        """Detach all sky nodes and clear the terrain fog."""
        for np_node in (self._dome_np, self._cloud_np, self._rain_root):
            if np_node is not None:
                np_node.remove_node()
        self._dome_np = self._cloud_np = self._rain_root = None
        self._rain_layers.clear()
        if self.terrain_root is not None and self._fog is not None:
            self.terrain_root.clear_fog()
            self.terrain_root.clear_color_scale()

    # ------------------------------------------------------------------
    # Build helpers (start-time only)
    # ------------------------------------------------------------------

    def _build_dome(self, star_count: int) -> None:
        """Build the inverted sky-dome sphere + dome shader + night texture."""
        node = _build_dome_node(_DOME_RADIUS_M, _DOME_STACKS, _DOME_SLICES)
        dome = self.base.render.attach_new_node(node)
        dome.set_bin("background", 10)
        dome.set_depth_write(False)
        dome.set_depth_test(False)
        dome.set_light_off()
        dome.set_color_off()

        night_tex = _sky_texture("night_sky",
                                 fallback=_fallback_night_sky(star_count))
        night_tex.set_wrap_u(Texture.WM_repeat)
        night_tex.set_wrap_v(Texture.WM_clamp)
        dome.set_texture(night_tex)

        shader = Shader.make(Shader.SL_GLSL,
                             vertex=sky_shaders.SKY_DOME_VERTEX,
                             fragment=sky_shaders.SKY_DOME_FRAGMENT)
        dome.set_shader(shader)
        # Neutral defaults so the first frame renders with every uniform defined.
        dome.set_shader_input("u_sun_dir", LVecBase3f(0.0, 0.0, 1.0))
        dome.set_shader_input("u_sun_color", LVecBase3f(1.0, 0.95, 0.85))
        dome.set_shader_input("u_sun_intensity", 0.0)
        dome.set_shader_input("u_moon_dir", LVecBase3f(0.0, 0.0, -1.0))
        dome.set_shader_input("u_moon_phase", 0.5)
        dome.set_shader_input("u_zenith_color", LVecBase3f(0.2, 0.35, 0.6))
        dome.set_shader_input("u_horizon_color", LVecBase3f(0.6, 0.7, 0.8))
        dome.set_shader_input("u_star_visibility", 0.0)
        dome.set_shader_input("u_star_rotation", 0.0)
        dome.set_shader_input("u_time", 0.0)
        dome.set_shader_input("u_fog_color", LVecBase3f(0.6, 0.65, 0.7))
        dome.set_shader_input("u_fog_blend", 0.0)
        dome.set_shader_input("u_ss_active", 0.0)
        dome.set_shader_input("u_ss_start", LVecBase3f(0.0, 1.0, 0.3))
        dome.set_shader_input("u_ss_travel", LVecBase3f(1.0, 0.0, 0.0))
        dome.set_shader_input("u_ss_progress", 0.0)
        self._dome_np = dome

    def _build_clouds(self) -> None:
        """Build the cloud slab quads + raymarch shader (static uniforms once)."""
        node = _build_cloud_node(_CLOUD_QUAD_SIDE_M, self._cloud_thick_m)
        clouds = self.base.render.attach_new_node(node)
        clouds.set_bin("background", 20)          # after the dome, before terrain
        clouds.set_depth_write(False)
        clouds.set_depth_test(False)
        clouds.set_two_sided(True)                # visible from below AND above
        clouds.set_light_off()
        clouds.set_transparency(TransparencyAttrib.M_alpha)

        shader = Shader.make(Shader.SL_GLSL,
                             vertex=sky_shaders.CLOUD_VERTEX,
                             fragment=sky_shaders.CLOUD_FRAGMENT)
        clouds.set_shader(shader)

        # World-seed-derived hash offset (deterministic cloud field).
        seed = float(for_domain("sky", "clouds").random())
        clouds.set_shader_input("u_seed", seed)
        # coverage → noise-threshold quantile table (see _cloud_value_quantiles).
        self._cloud_quantiles = _cloud_value_quantiles(seed)
        clouds.set_shader_input("u_altitude", self._cloud_alt_m)
        clouds.set_shader_input("u_thickness", self._cloud_thick_m)
        clouds.set_shader_input("u_cell", self._cloud_cell_m)
        clouds.set_shader_input("u_fade_dist",
                                _CLOUD_QUAD_SIDE_M * 0.5 * _CLOUD_FADE_FRACTION)
        clouds.set_shader_input("u_cam_pos", LVecBase3f(0.0, 0.0, 0.0))
        clouds.set_shader_input("u_coverage", 0.0)
        clouds.set_shader_input("u_opacity", 0.0)
        clouds.set_shader_input("u_wind_offset", LVecBase2f(0.0, 0.0))
        clouds.set_shader_input("u_top_color", LVecBase3f(1.0, 1.0, 1.0))
        clouds.set_shader_input("u_side_color", LVecBase3f(0.7, 0.7, 0.72))
        clouds.set_shader_input("u_bottom_color", LVecBase3f(0.45, 0.45, 0.5))
        clouds.set_pos(0.0, 0.0, self._cloud_alt_m)
        self._cloud_np = clouds

    def _build_rain(self) -> None:
        """Build three nested rain cylinders under one camera-following root."""
        rain_tex = _sky_texture("rain_streak", fallback=_fallback_rain_streak())
        rain_tex.set_wrap_u(Texture.WM_repeat)
        rain_tex.set_wrap_v(Texture.WM_repeat)
        # Override the bridge's retro nearest filter: magnified nearest streaks
        # become chunky bright bars; linear keeps them thin and soft.
        rain_tex.set_minfilter(SamplerState.FT_linear)
        rain_tex.set_magfilter(SamplerState.FT_linear)

        root = self.base.render.attach_new_node("rain_root")
        for radius_m, scroll_mult in _RAIN_LAYERS:
            node = _build_rain_cylinder(radius_m, _RAIN_HEIGHT_M, _RAIN_SEGMENTS)
            layer = root.attach_new_node(node)
            layer.set_texture(rain_tex)
            layer.set_two_sided(True)
            layer.set_light_off()
            layer.set_depth_write(False)
            layer.set_transparency(TransparencyAttrib.M_alpha)
            # Additive-ish: contribution = rgb * alpha, added — streaks brighten
            # the scene subtly instead of smearing gray over it.
            layer.set_attrib(ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_incoming_alpha,
                ColorBlendAttrib.O_one))
            self._rain_layers.append((layer, scroll_mult))
            self._rain_scroll.append(0.0)
        root.hide()
        self._rain_root = root
        self._rain_visible = False

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _camera_pos(self) -> tuple[float, float, float]:
        """World-space camera position in meters (engine transform authority)."""
        go = getattr(self.base, "camera_go", None)
        if go is not None:
            p = go.transform.position
            return float(p.x), float(p.y), float(p.z)
        cp = self.base.camera.get_pos(self.base.render)
        return float(cp.x), float(cp.y), float(cp.z)

    def _update_dome(self, st: Any, cx: float, cy: float, cz: float) -> None:
        """Follow the camera (translation only) and push the dome uniforms."""
        dome = self._dome_np
        dome.set_pos(cx, cy, cz)   # NEVER parented under the camera: world-oriented

        sun = st.sun_dir
        moon = st.moon_dir
        dome.set_shader_input("u_sun_dir",
                              LVecBase3f(float(sun.x), float(sun.y), float(sun.z)))
        dome.set_shader_input("u_sun_color", LVecBase3f(*st.sun_color))
        dome.set_shader_input("u_sun_intensity", float(st.sun_intensity))
        dome.set_shader_input("u_moon_dir",
                              LVecBase3f(float(moon.x), float(moon.y), float(moon.z)))
        dome.set_shader_input("u_moon_phase", float(st.moon_phase))
        dome.set_shader_input("u_zenith_color", LVecBase3f(*st.zenith_color))
        dome.set_shader_input("u_horizon_color", LVecBase3f(*st.horizon_color))
        dome.set_shader_input("u_star_visibility", float(st.star_visibility))
        dome.set_shader_input("u_time", float(self._time_s))
        dome.set_shader_input("u_fog_color", LVecBase3f(*st.fog_color))
        dome.set_shader_input("u_fog_blend", self._fog_blend(st))

        # Slow whole-sky star rotation: one revolution per game day.
        rot = 0.0
        if self.clock is not None:
            rot = (float(self.clock.game_time_of_day) / _GAME_SECONDS_PER_DAY
                   ) * 2.0 * math.pi
        dome.set_shader_input("u_star_rotation", rot)

    def _update_shooting_star(self, st: Any, dt: float) -> None:
        """
        Animate + deterministically schedule shooting stars.

        Game time is divided into 30-game-minute slots; per slot,
        ``for_domain("sky", "shooting_stars", game_day, slot)`` decides spawn
        (p≈0.5) plus start/travel directions, so every run of the same seed
        shows the same meteors at the same in-game moments.  The streak
        animates over ~1.2 real seconds and only spawns while
        ``star_visibility > 0.5``.
        """
        dome = self._dome_np
        # Animate the active streak.
        if self._ss_progress >= 0.0:
            self._ss_progress += dt / _SS_DURATION_REAL_S
            if self._ss_progress >= 1.0:
                self._ss_progress = -1.0
                dome.set_shader_input("u_ss_active", 0.0)
            else:
                dome.set_shader_input("u_ss_progress", float(self._ss_progress))

        if self.clock is None:
            return
        slot = int(float(self.clock.game_time_of_day) // _SS_SLOT_GAME_S)
        key = (int(self.clock.game_day), slot)
        if key == self._ss_slot:
            return
        self._ss_slot = key
        if float(st.star_visibility) <= _SS_MIN_STAR_VIS or self._ss_progress >= 0.0:
            return

        rng = for_domain("sky", "shooting_stars", key[0], key[1])
        if float(rng.random()) >= _SS_SPAWN_P:
            return
        # Start direction: random azimuth, elevation 20°–70°.
        az = float(rng.random()) * 2.0 * math.pi
        el = math.radians(20.0 + 50.0 * float(rng.random()))
        s = np.array([math.cos(el) * math.cos(az),
                      math.cos(el) * math.sin(az),
                      math.sin(el)], dtype=np.float64)
        # Travel direction: random vector orthogonalised against the start dir.
        az2 = float(rng.random()) * 2.0 * math.pi
        raw = np.array([math.cos(az2), math.sin(az2),
                        -0.6 * float(rng.random())], dtype=np.float64)
        trav = raw - s * float(np.dot(raw, s))
        norm = float(np.linalg.norm(trav))
        if norm < 1e-6:
            return
        trav /= norm
        dome.set_shader_input("u_ss_start",
                              LVecBase3f(float(s[0]), float(s[1]), float(s[2])))
        dome.set_shader_input("u_ss_travel",
                              LVecBase3f(float(trav[0]), float(trav[1]),
                                         float(trav[2])))
        dome.set_shader_input("u_ss_active", 1.0)
        dome.set_shader_input("u_ss_progress", 0.0)
        self._ss_progress = 0.0

    def _update_clouds(self, st: Any, cx: float, cy: float, cz: float) -> None:
        """Follow the camera (cell-snapped XY) and push the cloud uniforms."""
        clouds = self._cloud_np
        cell = self._cloud_cell_m
        # Snap to cell multiples so the grid never swims under the quad.
        clouds.set_pos(math.floor(cx / cell) * cell,
                       math.floor(cy / cell) * cell,
                       self._cloud_alt_m)
        clouds.set_shader_input("u_cam_pos", LVecBase3f(cx, cy, cz))
        # Map coverage (fill fraction) to the matching noise threshold so the
        # requested fraction of cells is actually occupied.
        q = self._cloud_quantiles
        cov = _clamp01(float(st.cloud_coverage))
        clouds.set_shader_input("u_coverage",
                                float(q[int(cov * (len(q) - 1))]))
        clouds.set_shader_input("u_opacity",
                                _clamp01(0.55 + 0.45 * float(st.cloud_density)))
        clouds.set_shader_input("u_wind_offset",
                                LVecBase2f(self._wind_x_m, self._wind_y_m))

        # Flat-face lighting, computed CPU-side from the sky state:
        # tops catch sun + sky ambient, sides are medium, bottoms darkest and
        # pulled toward storm-gray as density rises; at night a faint moon/sky
        # term keeps them barely readable.
        sun_r, sun_g, sun_b = st.sun_color
        si = float(st.sun_intensity)
        hr, hg, hb = st.horizon_color          # bright weather-graded sky ambient
        night = 0.05 * float(st.star_visibility)
        top = (
            _clamp01(sun_r * si * 0.55 + hr * 0.58 + night),
            _clamp01(sun_g * si * 0.55 + hg * 0.58 + night),
            _clamp01(sun_b * si * 0.55 + hb * 0.58 + night * 1.5),
        )
        side = tuple(c * 0.80 for c in top)
        bottom_scale = 0.66 - 0.34 * float(st.cloud_density)   # storm-dark bottoms
        bottom = tuple(c * bottom_scale for c in top)
        clouds.set_shader_input("u_top_color", LVecBase3f(*top))
        clouds.set_shader_input("u_side_color", LVecBase3f(*side))
        clouds.set_shader_input("u_bottom_color", LVecBase3f(*bottom))

    def _update_rain(self, st: Any, cx: float, cy: float, cz: float,
                     dt: float) -> None:
        """Follow the camera, scroll the streak UVs, tilt with the wind."""
        ri = float(st.rain_intensity)
        if ri < _RAIN_HIDE_THRESHOLD:
            if self._rain_visible:
                self._rain_root.hide()
                self._rain_visible = False
            return
        if not self._rain_visible:
            self._rain_root.show()
            self._rain_visible = True

        root = self._rain_root
        root.set_pos(cx, cy, cz)
        # Slight wind tilt: heading faces the wind, pitch leans the streaks.
        wx, wy = st.wind_dir
        heading_deg = math.degrees(math.atan2(-float(wx), float(wy)))
        tilt_deg = min(_RAIN_MAX_TILT_DEG, float(st.wind_speed) * 1.1)
        root.set_hpr(heading_deg, tilt_deg, 0.0)
        # Subtle is better than spammy: fade contribution with intensity.
        root.set_color_scale(1.0, 1.0, 1.0, _clamp01(0.12 + 0.38 * ri))

        # Scroll DOWNWARD: with the cylinder's mirrored V (v grows toward the
        # ground — see _build_rain_cylinder), a DECREASING offset translates
        # the streak pattern toward -Z.  Per-layer rates give parallax.
        stage = TextureStage.get_default()
        rate = _RAIN_BASE_SCROLL * (0.5 + 1.5 * ri)
        for i, (layer, mult) in enumerate(self._rain_layers):
            self._rain_scroll[i] = (self._rain_scroll[i] - rate * mult * dt) % 1.0
            layer.set_tex_offset(stage, 0.0, self._rain_scroll[i])

    def _update_fog_and_light(self, st: Any) -> None:
        """Exponential fog + clear colour + global terrain light scale."""
        fr, fg, fb = st.fog_color
        if self._fog is not None:
            self._fog.set_exp_density(float(st.fog_density))
            self._fog.set_color(LVecBase4f(fr, fg, fb, 1.0))

        # Clear colour behind everything: horizon blended toward fog.
        blend = self._fog_blend(st)
        hr, hg, hb = st.horizon_color
        self.base.set_background_color(hr + (fr - hr) * blend,
                                       hg + (fg - hg) * blend,
                                       hb + (fb - hb) * blend, 1.0)

        # Lighting integration: baked vertex sunlight × global day/night scale.
        if self.terrain_root is not None:
            sr, sg, sb = st.terrain_light_scale
            self.terrain_root.set_color_scale(float(sr), float(sg),
                                              float(sb), 1.0)

    @staticmethod
    def _fog_blend(st: Any) -> float:
        """
        Map ``fog_density`` (1/m, ~0.0008 clear … 0.025 heavy) to a 0-1 factor
        for blending the horizon band / clear colour toward the fog colour.
        """
        return _clamp01((float(st.fog_density) - 0.0008) / (0.020 - 0.0008))
