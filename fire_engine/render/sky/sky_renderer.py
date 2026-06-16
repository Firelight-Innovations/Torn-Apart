"""
world/sky_renderer.py — SkyRendererComponent: the render half of the sky system.

Draws everything the headless ``fire_engine.world.sky`` simulation describes through
its per-frame ``SkyState``:

* **Sky dome** — an inverted UV-sphere (radius ~800 m) camera-centred via
  translation-only follow, painted by ``sky_shaders.SKY_DOME_FRAGMENT``:
  physical atmosphere, sun disc + halo, moon with phase terminator, the
  ``"night_sky_cube"`` star/galaxy CUBE MAP (per-star twinkle, no pole
  distortion) rotating about a seed-derived TILTED celestial axis, and
  deterministic shooting stars.
* **Volumetric clouds** — a second camera-centred inverted sphere whose
  fragment shader (``sky_shaders.CLOUD_VOLUMETRIC_FRAGMENT``) raymarches the
  cloud slab sampling the baked tileable 3-D noise (``sky.cloud_noise``):
  self-shadowed (Beer + powder), HG forward-scatter silver lining, premultiplied
  OVER the sky so a bright sun bleeds through thin cloud.  Gated by ``gfx_clouds``.
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
``fire_engine.render.__init__`` exports ``SkyRendererComponent`` behind the same
try/except guard as the other panda3d bridges.

Example
-------
::

    from fire_engine.world.sky import SkySystem
    from fire_engine.render import instantiate
    from fire_engine.render.sky.sky_renderer import SkyRendererComponent

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
from panda3d.core import (
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
    NodePath,
    SamplerState,
    Shader,
    Texture,
    TransparencyAttrib,
)

from fire_engine.core import get_logger
from fire_engine.core.rng import for_domain
from fire_engine.render.component import Component
from fire_engine.render.sky import sky_shaders

__all__ = ["SkyRendererComponent"]

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

# Shooting stars: deterministic schedule (see _update_shooting_star).
_SS_SLOT_GAME_S: float = 1800.0  # one slot = 30 game-minutes (game seconds)
_SS_DURATION_REAL_S: float = 1.2  # streak animation length (real seconds)
_SS_SPAWN_P: float = 0.5  # spawn probability per slot
_SS_MIN_STAR_VIS: float = 0.5  # only spawn when stars are visible

_GAME_SECONDS_PER_DAY: float = 86400.0

# Contract defaults for the sky config keys (the sky package adds them to
# core.config; fall back to the frozen-contract values when running against a
# pre-sky Config so the renderer works mid-integration).
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
    from pathlib import Path

    from fire_engine.world.sky.cloud_noise import bake_detail_noise, bake_shape_noise

    cache_dir = Path("saves") / "cloud_cache"
    version = 1

    from collections.abc import Callable

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


# ---------------------------------------------------------------------------
# Texture acquisition (registry first, deterministic fallback second)
# ---------------------------------------------------------------------------


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
    rgba: np.ndarray | None = None
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


# ---------------------------------------------------------------------------
# SkyRendererComponent
# ---------------------------------------------------------------------------


class SkyRendererComponent(Component):
    """
    Render component for the procedural sky + weather system.

    Lifecycle
    ---------
    * ``start()`` — builds all scene-graph nodes ONCE (dome, cloud quads, Fog
      object), compiles the GLSL shaders, uploads textures, and extends the
      camera far plane to cover the sky geometry.  (Rain now lives in
      ``world/rain_renderer.py`` — M6.)
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
    sky_system : fire_engine.world.sky.SkySystem (or duck-typed stub)
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

    def __init__(
        self,
        base: Any = None,
        sky_system: Any = None,
        terrain_root: Any = None,
        clock: Any = None,
        external_lighting: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.sky_system = sky_system
        self.terrain_root = terrain_root
        self.clock = clock
        # True when the GPU volumetric lighting pipeline owns terrain
        # shading: the sky renderer must then NOT apply Panda3D Fog, NOT
        # set terrain_root colour-scale, and NOT blend the clear colour
        # toward fog (the froxel fog composites in the terrain/sky shaders).
        self.external_lighting = bool(external_lighting)
        if self.clock is None and sky_system is not None:
            self.clock = getattr(sky_system, "clock", None) or getattr(sky_system, "_clock", None)

        # Per-frame state
        self._state: Any = None  # last SkyState from sky_system.update()
        self._time_s: float = 0.0  # real seconds since start (twinkle)
        self._wind_x_m: float = 0.0  # accumulated wind drift (meters)
        self._wind_y_m: float = 0.0

        # Scene-graph nodes (built in start())
        self._dome_np: NodePath | None = None
        self._cloud_np: NodePath | None = None
        self._fog: Fog | None = None

        # Shooting-star animation state
        self._ss_slot: tuple[int, int] | None = None  # (game_day, slot)
        self._ss_progress: float = -1.0  # < 0 → inactive

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
        star_count = int(getattr(cfg, "sky_star_count", _DEFAULT_STAR_COUNT))

        # Far plane must cover the sky dome (800 m); the volumetric cloud slab
        # is intersected analytically in the shader, so it needs no geometry.
        lens = self.base.camLens
        if lens.get_far() < _CAMERA_FAR_M:
            lens.set_far(_CAMERA_FAR_M)

        self._build_dome(star_count)
        self._build_clouds()

        # Exponential fog on the terrain (density/colour driven per frame).
        # Skipped under external (GPU volumetric) lighting — the froxel fog
        # composites inside the terrain shader instead.
        if not self.external_lighting:
            self._fog = Fog("sky_fog")
            self._fog.set_exp_density(0.0008)
            if self.terrain_root is not None:
                self.terrain_root.set_fog(self._fog)

        _log.info(
            "Sky renderer ready (volumetric cloud slab z=[%.0f, %.0f] m)",
            _VCLOUD_ALT_M,
            _VCLOUD_ALT_M + _VCLOUD_THICK_M,
        )

    def update(self, dt: float) -> None:
        """
        Advance the sky simulation for this frame.

        The component registry runs ALL update() calls before any
        late_update(), so calling ``sky_system.update()`` here guarantees a
        fresh ``SkyState`` for ``late_update`` without modifying App.

        This is the single driver of ``sky_system.update(player_pos)`` — it
        threads the camera world XY through so distant storms sample at the
        camera (M4).  Other readers (wind, the WeatherMapComponent's raster)
        consume the already-advanced weather system, so nothing else calls
        ``update`` (no double-advance).
        """
        self._time_s += dt
        if self.sky_system is not None:
            cx, cy, _ = self._camera_pos()
            self._state = self.sky_system.update((cx, cy))

    def late_update(self, dt: float) -> None:
        """Write this frame's SkyState to the GPU (bulk uniform/state writes)."""
        st = self._state if self._state is not None else getattr(self.sky_system, "state", None)
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
        self._update_fog_and_light(st)

    def on_destroy(self) -> None:
        """Detach all sky nodes and clear the terrain fog."""
        for np_node in (self._dome_np, self._cloud_np):
            if np_node is not None:
                np_node.remove_node()
        self._dome_np = self._cloud_np = None
        if self.terrain_root is not None and self._fog is not None:
            self.terrain_root.clear_fog()
            self.terrain_root.clear_color_scale()

    # ------------------------------------------------------------------
    # Build helpers (start-time only)
    # ------------------------------------------------------------------

    def _build_dome(self, star_count: int) -> None:
        """Build the inverted sky-dome sphere + dome shader + star cubemap."""
        node = _build_dome_node(_DOME_RADIUS_M, _DOME_STACKS, _DOME_SLICES)
        dome = self.base.render.attach_new_node(node)
        dome.set_bin("background", 10)
        dome.set_depth_write(False)
        dome.set_depth_test(False)
        dome.set_light_off()
        dome.set_color_off()

        shader = Shader.make(
            Shader.SL_GLSL,
            vertex=sky_shaders.SKY_DOME_VERTEX,
            fragment=sky_shaders.SKY_DOME_FRAGMENT,
        )
        dome.set_shader(shader)

        # Night-sky star/galaxy CUBE MAP (no equirect pole distortion) and
        # the tilted celestial axis it rotates about: Polaris elevation ==
        # the world's latitude, seed-derived per world — so the night sky
        # wheels properly across the sky instead of pinwheeling at zenith.
        star_cube: np.ndarray | None = None
        try:
            from fire_engine.procedural import get as get_procedural

            star_cube = get_procedural("night_sky_cube", star_count=star_count)
        except Exception as exc:
            _log.warning("night_sky_cube unavailable (%s) — using fallback", exc)
        if star_cube is None:
            star_cube = _fallback_star_cube(star_count)
        from fire_engine.render.bridges.texture_bridge import to_panda_cubemap

        dome.set_shader_input("u_star_cube", to_panda_cubemap(star_cube))
        lat_rad = math.radians(
            28.0 + 27.0 * float(for_domain("sky", "celestial_latitude").random())
        )
        dome.set_shader_input(
            "u_celestial_axis", LVecBase3f(0.0, math.cos(lat_rad), math.sin(lat_rad))
        )
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

        # Physical-atmosphere additions: procedural moon texture, tonemap
        # exposure (matches the terrain shader), night-floor/weather inputs,
        # and the froxel-fog defaults (a 1x1x1 dummy keeps the sampler3D
        # bound; the GPU pipeline's real texture replaces it in late_update).
        moon_tex = _sky_texture("moon_surface", fallback=_fallback_moon())
        moon_tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
        moon_tex.set_magfilter(SamplerState.FT_linear)
        dome.set_shader_input("u_moon_tex", moon_tex)
        dome.set_shader_input("u_moon_glow", 0.0)
        dome.set_shader_input("u_daylight", 1.0)
        dome.set_shader_input("u_weather_gray", 0.0)
        cfg = getattr(self.base, "_config", None)
        dome.set_shader_input("u_exposure", float(getattr(cfg, "light_exposure", 0.9)))
        # Config-exposed sky/sun tuning (static — set once; see core/config.py).
        dome.set_shader_input(
            "u_sun_disc_intensity", float(getattr(cfg, "gfx_sun_disc_intensity", 45.0))
        )
        dome.set_shader_input(
            "u_sun_halo_intensity", float(getattr(cfg, "gfx_sun_halo_intensity", 1.8))
        )
        dome.set_shader_input(
            "u_sun_min_brightness", float(getattr(cfg, "gfx_sun_min_brightness", 0.25))
        )
        dome.set_shader_input(
            "u_sky_inscatter_scale", float(getattr(cfg, "gfx_sky_inscatter_scale", 0.9))
        )
        dummy_fog = Texture("dome_fog_dummy")
        dummy_fog.setup_3d_texture(1, 1, 1, Texture.T_float, Texture.F_rgba16)
        dummy_fog.set_clear_color((0.0, 0.0, 0.0, 1.0))
        dome.set_shader_input("u_fog_integrated", dummy_fog)
        dome.set_shader_input("u_fog_enabled", 0.0)
        dome.set_shader_input("u_viewport", LVecBase2f(1280.0, 720.0))
        self._fog_tex_bound = False
        self._dome_np = dome

    def _build_clouds(self) -> None:
        """
        Build the volumetric cloud "dome" + raymarch shader (static uniforms).

        Reuses the inverted-sphere dome geometry purely to get a per-pixel world
        view direction; the fragment shader analytically intersects + marches
        the cloud slab in world space, so clouds fill the sky to the horizon.
        Drawn after the sky dome (bin 15) and before terrain, composited with a
        premultiplied-over blend (``src + dst·srcAlpha``, srcAlpha =
        transmittance).  Gated by ``gfx_clouds``.
        """
        cfg = getattr(self.base, "_config", None)
        if not bool(getattr(cfg, "gfx_clouds", True)):
            self._cloud_np = None
            _log.info("Volumetric clouds disabled (gfx_clouds=false)")
            return

        from fire_engine.render.bridges.texture_bridge import to_panda_texture_3d

        node = _build_dome_node(_DOME_RADIUS_M, _DOME_STACKS, _DOME_SLICES)
        clouds = self.base.render.attach_new_node(node)
        clouds.set_bin("background", 15)  # after dome (10), before terrain
        clouds.set_depth_write(False)
        clouds.set_depth_test(False)
        clouds.set_light_off()
        clouds.set_color_off()
        # Premultiplied OVER: out = src.rgb + dst.rgb · src.a, with src.a =
        # transmittance — a bright sun bleeds through thin cloud, thick occludes.
        clouds.set_transparency(TransparencyAttrib.M_none)
        clouds.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add, ColorBlendAttrib.O_one, ColorBlendAttrib.O_incoming_alpha
            )
        )

        shader = Shader.make(
            Shader.SL_GLSL,
            vertex=sky_shaders.CLOUD_VOLUMETRIC_VERTEX,
            fragment=sky_shaders.CLOUD_VOLUMETRIC_FRAGMENT,
        )
        clouds.set_shader(shader)

        # Baked, tileable density volumes (disk-cached — deterministic per seed).
        seed = int(getattr(cfg, "world_seed", 0))
        shape_arr, detail_arr = _load_or_bake_cloud_noise(
            seed, _VCLOUD_SHAPE_SIZE, _VCLOUD_DETAIL_SIZE
        )
        clouds.set_shader_input("u_shape", to_panda_texture_3d(shape_arr))
        clouds.set_shader_input("u_detail", to_panda_texture_3d(detail_arr))

        # Static uniforms.
        clouds.set_shader_input("u_altitude", _VCLOUD_ALT_M)
        clouds.set_shader_input("u_thickness", _VCLOUD_THICK_M)
        clouds.set_shader_input("u_max_dist", float(getattr(cfg, "gfx_cloud_max_dist_m", 2400.0)))
        clouds.set_shader_input("u_shape_scale", 1.0 / _VCLOUD_SHAPE_TILE_M)
        clouds.set_shader_input("u_detail_scale", 1.0 / _VCLOUD_DETAIL_TILE_M)
        clouds.set_shader_input("u_detail_strength", _VCLOUD_DETAIL_STR)
        clouds.set_shader_input("u_sigma", _VCLOUD_SIGMA)
        clouds.set_shader_input("u_hg", _VCLOUD_HG)
        clouds.set_shader_input("u_light_step_m", _VCLOUD_LIGHT_STEP_M)
        clouds.set_shader_input("u_steps", int(getattr(cfg, "gfx_cloud_steps", 48)))
        clouds.set_shader_input("u_light_steps", int(getattr(cfg, "gfx_cloud_light_steps", 6)))
        clouds.set_shader_input("u_exposure", float(getattr(cfg, "light_exposure", 0.9)))

        # Per-frame defaults (overwritten in _update_clouds).
        clouds.set_shader_input("u_cam_pos", LVecBase3f(0.0, 0.0, 0.0))
        clouds.set_shader_input("u_sun_dir", LVecBase3f(0.0, 0.0, 1.0))
        clouds.set_shader_input("u_moon_dir", LVecBase3f(0.0, 0.0, -1.0))
        clouds.set_shader_input("u_sun_radiance", LVecBase3f(3.0, 2.9, 2.6))
        clouds.set_shader_input("u_moon_radiance", LVecBase3f(0.06, 0.07, 0.10))
        clouds.set_shader_input("u_sky_ambient", LVecBase3f(0.4, 0.5, 0.7))
        clouds.set_shader_input("u_coverage", 0.5)
        clouds.set_shader_input("u_cloud_density", 1.0)
        clouds.set_shader_input("u_wind", LVecBase2f(0.0, 0.0))
        clouds.set_shader_input("u_time", 0.0)

        # M4 weather-map contract defaults: the WeatherMapComponent binds the
        # real texture + origin + enable on ``render`` (inherited here), but a
        # dummy 1x1 sampler and disabled state keep the shader valid even when
        # that component is absent / the feature is off (pre-M4 flat-ambient
        # look).  A bound sampler2D is required (an unbound one is UB).
        dummy_wmap = Texture("weather_map_dummy")
        dummy_wmap.setup_2d_texture(1, 1, Texture.T_half_float, Texture.F_rgba16)
        dummy_wmap.set_clear_color((0.0, 0.0, 0.0, 0.0))
        clouds.set_shader_input("u_weather_map", dummy_wmap)
        clouds.set_shader_input("u_wmap_origin", LVecBase2f(0.0, 0.0))
        clouds.set_shader_input("u_wmap_cell_m", 1.0)
        clouds.set_shader_input("u_wmap_cells", 1.0)
        clouds.set_shader_input("u_weather_map_enabled", 0)
        clouds.set_shader_input("u_weather_ambient", LVecBase2f(0.0, 0.0))
        clouds.set_shader_input("u_virga_enabled", 0)

        # M9 WMO cloud genera: layered high/mid/low altitude bands derived
        # in-shader from the weather map (no new texture data).  All static —
        # config tunables pushed once here; the band selection is per-step in
        # the shader from the existing coverage/density/precip channels.  Gated
        # by gfx_cloud_genera (requires gfx_weather_map; off ⇒ single slab, the
        # pre-M9 look — the shader's u_cloud_genera_enabled==0 path).
        genera_on = bool(getattr(cfg, "gfx_cloud_genera", False)) and bool(
            getattr(cfg, "gfx_weather_map", False)
        )
        clouds.set_shader_input("u_cloud_genera_enabled", 1 if genera_on else 0)
        clouds.set_shader_input(
            "u_genera_high_alt", float(getattr(cfg, "cloud_genera_high_alt_m", 1400.0))
        )
        clouds.set_shader_input(
            "u_genera_high_thick", float(getattr(cfg, "cloud_genera_high_thick_m", 120.0))
        )
        clouds.set_shader_input(
            "u_genera_mid_alt", float(getattr(cfg, "cloud_genera_mid_alt_m", 850.0))
        )
        clouds.set_shader_input(
            "u_genera_mid_thick", float(getattr(cfg, "cloud_genera_mid_thick_m", 220.0))
        )
        clouds.set_shader_input(
            "u_genera_high_floor", float(getattr(cfg, "cloud_genera_high_cov_floor", 0.06))
        )
        clouds.set_shader_input(
            "u_genera_high_cov_w", float(getattr(cfg, "cloud_genera_high_cov_weight", 0.35))
        )
        clouds.set_shader_input(
            "u_genera_high_density", float(getattr(cfg, "cloud_genera_high_density", 0.30))
        )
        clouds.set_shader_input(
            "u_genera_mid_cov_w", float(getattr(cfg, "cloud_genera_mid_cov_weight", 0.60))
        )
        clouds.set_shader_input(
            "u_genera_high_detail", float(getattr(cfg, "cloud_genera_high_detail_scale", 0.45))
        )
        clouds.set_shader_input(
            "u_genera_mid_detail", float(getattr(cfg, "cloud_genera_mid_detail_scale", 0.85))
        )
        self._cloud_np = clouds

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
        assert self._dome_np is not None  # invariant: guarded in late_update before call
        dome = self._dome_np
        dome.set_pos(cx, cy, cz)  # NEVER parented under the camera: world-oriented

        sun = st.sun_dir
        moon = st.moon_dir
        dome.set_shader_input("u_sun_dir", LVecBase3f(float(sun.x), float(sun.y), float(sun.z)))
        dome.set_shader_input("u_sun_color", LVecBase3f(*st.sun_color))
        dome.set_shader_input("u_sun_intensity", float(st.sun_intensity))
        dome.set_shader_input("u_moon_dir", LVecBase3f(float(moon.x), float(moon.y), float(moon.z)))
        dome.set_shader_input("u_moon_phase", float(st.moon_phase))
        dome.set_shader_input("u_zenith_color", LVecBase3f(*st.zenith_color))
        dome.set_shader_input("u_horizon_color", LVecBase3f(*st.horizon_color))
        dome.set_shader_input("u_star_visibility", float(st.star_visibility))
        dome.set_shader_input("u_time", float(self._time_s))
        dome.set_shader_input("u_fog_color", LVecBase3f(*st.fog_color))
        # Legacy horizon fog band only on the CPU backend; the froxel fog
        # owns atmosphere depth under external (GPU volumetric) lighting.
        dome.set_shader_input("u_fog_blend", 0.0 if self.external_lighting else self._fog_blend(st))

        # Physical-atmosphere per-frame inputs.
        dome.set_shader_input("u_daylight", float(st.daylight))
        gray = min(1.0, 1.6 * float(st.cloud_coverage) * float(st.cloud_density))
        dome.set_shader_input("u_weather_gray", gray)
        illum = 0.5 * (1.0 - math.cos(2.0 * math.pi * float(st.moon_phase)))
        dome.set_shader_input("u_moon_glow", float(illum))

        # Froxel-fog composite: bind the pipeline's integrated texture once
        # it exists (the pipeline is created after the sky GameObject).
        pipeline = getattr(self.base, "lighting_pipeline", None)
        if self.external_lighting and pipeline is not None:
            # Auto-exposure: the dome uses the COMPRESSED adaptation
            # (pipeline.exposure_sky) — terrain brightens fully in the dark,
            # the night sky deepens only slightly (stars keep their contrast).
            dome.set_shader_input(
                "u_exposure",
                float(getattr(pipeline, "exposure_sky", getattr(pipeline, "exposure", 0.9))),
            )
            if getattr(pipeline, "fog_enabled", False):
                if not self._fog_tex_bound:
                    dome.set_shader_input("u_fog_integrated", pipeline.fog_integrated_tex)
                    dome.set_shader_input("u_fog_enabled", 1.0)
                    self._fog_tex_bound = True
                win = self.base.win
                dome.set_shader_input(
                    "u_viewport", LVecBase2f(float(win.get_x_size()), float(win.get_y_size()))
                )

        # Slow whole-sky star rotation: one revolution per game day.
        rot = 0.0
        if self.clock is not None:
            rot = (float(self.clock.game_time_of_day) / _GAME_SECONDS_PER_DAY) * 2.0 * math.pi
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
        assert self._dome_np is not None  # invariant: guarded in late_update before call
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
        s = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            dtype=np.float64,
        )
        # Travel direction: random vector orthogonalised against the start dir.
        az2 = float(rng.random()) * 2.0 * math.pi
        raw = np.array([math.cos(az2), math.sin(az2), -0.6 * float(rng.random())], dtype=np.float64)
        trav = raw - s * float(np.dot(raw, s))
        norm = float(np.linalg.norm(trav))
        if norm < 1e-6:
            return
        trav /= norm
        dome.set_shader_input("u_ss_start", LVecBase3f(float(s[0]), float(s[1]), float(s[2])))
        dome.set_shader_input(
            "u_ss_travel", LVecBase3f(float(trav[0]), float(trav[1]), float(trav[2]))
        )
        dome.set_shader_input("u_ss_active", 1.0)
        dome.set_shader_input("u_ss_progress", 0.0)
        self._ss_progress = 0.0

    def _update_clouds(self, st: Any, cx: float, cy: float, cz: float) -> None:
        """Follow the camera and push the volumetric-cloud per-frame uniforms."""
        clouds = self._cloud_np
        if clouds is None:
            return
        # Camera-follow (translation only — the dome must stay world-oriented so
        # the slab intersection uses true world directions).
        clouds.set_pos(cx, cy, cz)
        clouds.set_shader_input("u_cam_pos", LVecBase3f(cx, cy, cz))

        sun = st.sun_dir
        moon = st.moon_dir
        clouds.set_shader_input("u_sun_dir", LVecBase3f(float(sun.x), float(sun.y), float(sun.z)))
        clouds.set_shader_input(
            "u_moon_dir", LVecBase3f(float(moon.x), float(moon.y), float(moon.z))
        )
        clouds.set_shader_input("u_sun_radiance", LVecBase3f(*st.sun_radiance))
        clouds.set_shader_input("u_moon_radiance", LVecBase3f(*st.moon_radiance))
        clouds.set_shader_input("u_sky_ambient", LVecBase3f(*st.sky_ambient))
        clouds.set_shader_input("u_coverage", _clamp01(float(st.cloud_coverage)))
        clouds.set_shader_input("u_cloud_density", _clamp01(0.75 + 0.25 * float(st.cloud_density)))
        clouds.set_shader_input("u_wind", LVecBase2f(self._wind_x_m, self._wind_y_m))
        clouds.set_shader_input("u_time", float(self._time_s))

        # Legacy (post-off) path tonemaps inside the cloud shader — keep its
        # exposure synced to the dome's compressed auto-exposure.
        pipeline = getattr(self.base, "lighting_pipeline", None)
        if pipeline is not None:
            clouds.set_shader_input(
                "u_exposure",
                float(getattr(pipeline, "exposure_sky", getattr(pipeline, "exposure", 0.9))),
            )

    def _update_fog_and_light(self, st: Any) -> None:
        """Exponential fog + clear colour + global terrain light scale."""
        if self.external_lighting:
            # GPU pipeline owns terrain light + fog; keep a plain horizon
            # clear colour so un-domed pixels match the sky.
            hr, hg, hb = st.horizon_color
            self.base.set_background_color(hr, hg, hb, 1.0)
            return
        fr, fg, fb = st.fog_color
        if self._fog is not None:
            self._fog.set_exp_density(float(st.fog_density))
            self._fog.set_color(LVecBase4f(fr, fg, fb, 1.0))

        # Clear colour behind everything: horizon blended toward fog.
        blend = self._fog_blend(st)
        hr, hg, hb = st.horizon_color
        self.base.set_background_color(
            hr + (fr - hr) * blend, hg + (fg - hg) * blend, hb + (fb - hb) * blend, 1.0
        )

        # Lighting integration: baked vertex sunlight × global day/night scale.
        if self.terrain_root is not None:
            sr, sg, sb = st.terrain_light_scale
            self.terrain_root.set_color_scale(float(sr), float(sg), float(sb), 1.0)

    @staticmethod
    def _fog_blend(st: Any) -> float:
        """
        Map ``fog_density`` (1/m, ~0.0008 clear … 0.025 heavy) to a 0-1 factor
        for blending the horizon band / clear colour toward the fog colour.
        """
        return _clamp01((float(st.fog_density) - 0.0008) / (0.020 - 0.0008))
