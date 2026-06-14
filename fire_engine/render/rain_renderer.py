"""
world/rain_renderer.py — RainRendererComponent: volumetric rain (M6).

The render half of the M6 rain overhaul.  Replaces the camera-following scrolled
cylinders that lived in ``sky_renderer.py`` (and rained everywhere, even under a
roof) with two **gated** modes selected by ``config.gfx_rain_mode``:

* ``"particles"`` (medium+) — ``config.gfx_rain_particles`` GPU-instanced
  falling streaks on a camera-anchored wrapping lattice (the ``mote_renderer``
  pattern: instance placement / fall phase / sway from ``gl_InstanceID`` in the
  vertex shader, zero CPU per-particle state).
* ``"cylinders"`` (low) — the cheap nested camera-following cylinders, kept for
  weak GPUs but now gated by the same two tests in the fragment shader.
* ``"off"`` — nothing drawn.

Both rendered modes apply two per-element gates at the element's world XY:

1. **Rain-cover heightmap cull (THE M6 FIX).**  The component owns a headless
   :class:`~fire_engine.world.terrain.RainCoverField` — a top-down heightmap of the
   highest solid voxel per 1 m column around the player — and uploads it to
   ``u_rain_height_tex`` with committed-origin discipline (origin refreshed only
   in the same frame as the texel upload, mirroring ``wind``/``weather``).  A
   streak whose world Z is below the cover height there is under a roof/overhang
   and is discarded.  Toggled by ``config.gfx_rain_occlusion``.

2. **Storm-footprint precip gate.**  Both shaders sample the inherited
   weather-map precip channel (``u_weather_map`` B) at the element XY, so rain
   only exists inside storm cells (fading with precip).  When the weather map is
   off (``u_weather_map_enabled == 0``) they fall back to the scalar
   ``SkyState.rain_intensity`` bound here as ``u_rain_intensity``.

Rebuild discipline
------------------
The component subscribes to ``ChunkLoadedEvent`` / ``TerrainEditedEvent`` and
marks the affected chunk **columns** dirty; each ``late_update`` it recenters
the cover window when the player crosses a cell threshold (full rebuild) and
otherwise refolds up to ``config.rain_cover_budget_columns`` dirty columns,
amortising a cold rebuild over frames.  The heightmap is re-uploaded whenever it
changed (and always the frame it recenters).

Like the wind/grass components this is **GPU lighting backend only** (it needs
the live ``GpuLightingPipeline`` so the inherited wind/fog/camera uniforms exist
on ``terrain_root``).  On the CPU backend — or with ``gfx_rain_mode == "off"`` —
it disables itself with a log line.  Every feature is individually killable.

Example (wired by main.py)
--------------------------
    rain_go = instantiate()
    rain_go.add_component(
        RainRendererComponent,
        base=app, sky_system=sky_system, chunk_provider=chunk_manager,
        lighting_pipeline=pipeline, bus=bus)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    BoundingBox,
    ColorBlendAttrib,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LPoint3,
    LVecBase2f,
    LVecBase3f,
    NodePath,
    SamplerState,
    Shader,
    Texture,
    TransparencyAttrib,
)

from fire_engine.core import (
    ChunkLoadedEvent,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.core.rng import for_domain
from fire_engine.world.terrain import RainCoverField
from fire_engine.render.component import Component
from fire_engine.render import rain_shaders

__all__ = ["RainRendererComponent"]

_log = get_logger("world.rain")

# Streak geometry tuning (visual, not world config).
_RAIN_BOX_M:      float = 36.0    # camera-anchored lattice box edge (m)
_RAIN_SIZE_M:     float = 0.035   # streak half-width (m)
_RAIN_LENGTH_M:   float = 0.7     # streak half-length (m)
_RAIN_FALL_MPS:   float = 18.0    # base fall speed (m/s)
_RAIN_TINT:       tuple[float, float, float] = (0.62, 0.70, 0.85)  # cool gray-blue

# Cylinder mode geometry (mirrors the old sky_renderer values).
_CYL_LAYERS: tuple[tuple[float, float], ...] = ((4.0, 1.6), (7.0, 1.15), (11.0, 0.85))
_CYL_HEIGHT_M:    float = 14.0
_CYL_SEGMENTS:    int   = 32
_CYL_TEX_U_M:     float = 3.0
_CYL_TEX_V_M:     float = 12.0
_CYL_BASE_SCROLL: float = 1.4
_CYL_MAX_TILT_DEG: float = 14.0
_RAIN_HIDE_THRESHOLD: float = 0.05


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _build_streak_quad() -> Geom:
    """Shared unit streak quad (xy ∈ {-1,+1}, z=0, UV 0–1) drawn N times."""
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("rain_quad", fmt, Geom.UH_static)
    vdata.set_num_rows(4)
    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")
    corners = ((-1.0, -1.0, 0.0, 0.0), (1.0, -1.0, 1.0, 0.0),
               (1.0, 1.0, 1.0, 1.0), (-1.0, 1.0, 0.0, 1.0))
    for x, y, u, v in corners:
        vw.add_data3(x, y, 0.0)
        tw.add_data2(u, v)
    tris = GeomTriangles(Geom.UH_static)
    tris.add_vertices(0, 1, 2)
    tris.add_vertices(0, 2, 3)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    return geom


def _build_cylinder(radius_m: float, height_m: float, segments: int) -> GeomNode:
    """One open vertical cylinder for a cylinder-mode rain layer.

    UVs tile the rain-streak texture ~world-scaled; V is mirrored (bottom = high
    v) so a DECREASING per-frame scroll translates the pattern downward — the
    same convention the old sky_renderer cylinder used.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    x = (radius_m * np.cos(theta)).astype(np.float32)
    y = (radius_m * np.sin(theta)).astype(np.float32)
    u_tiles = (2.0 * np.pi * radius_m) / _CYL_TEX_U_M
    v_tiles = height_m / _CYL_TEX_V_M
    u = np.linspace(0.0, u_tiles, segments + 1).astype(np.float32)

    n = segments + 1
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("rain_cyl", fmt, Geom.UH_static)
    vdata.set_num_rows(2 * n)
    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")
    for k in range(n):
        vw.add_data3(float(x[k]), float(y[k]), -0.5 * height_m)
        tw.add_data2(float(u[k]), float(v_tiles))
    for k in range(n):
        vw.add_data3(float(x[k]), float(y[k]), 0.5 * height_m)
        tw.add_data2(float(u[k]), 0.0)
    tris = GeomTriangles(Geom.UH_static)
    for j in range(segments):
        b0, b1 = j, j + 1
        t0, t1 = j + n, j + 1 + n
        tris.add_vertices(b0, b1, t1)
        tris.add_vertices(b0, t1, t0)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("rain_cyl_layer")
    node.add_geom(geom)
    return node


def _rain_streak_texture():
    """The ``rain_streak`` procedural texture, repeat-wrapped + linear-filtered."""
    from fire_engine.procedural import get as get_procedural
    from fire_engine.render.texture_bridge import to_panda_texture
    rgba = get_procedural("rain_streak")
    tex = to_panda_texture(rgba)
    tex.set_wrap_u(Texture.WM_repeat)
    tex.set_wrap_v(Texture.WM_repeat)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex


class RainRendererComponent(Component):
    """
    Render component for M6 volumetric rain (gated by cover + storm footprint).

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``render``, ``terrain_root``, ``camera_go``
        and ``_config``.
    sky_system : fire_engine.world.sky.SkySystem
        Read-only; ``.state.rain_intensity`` is the scalar fallback when the
        weather map is off.
    chunk_provider : object
        Anything with a ``.chunks`` dict (``ChunkManager``) — the loaded chunks
        the cover heightmap folds.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` disables.
    bus : EventBus | None
        Subscribes to ``ChunkLoadedEvent`` / ``TerrainEditedEvent`` to mark the
        cover heightmap dirty (state-change events only — never per-frame).

    Units: meters, seconds.  World-space Z-up.
    """

    def __init__(self, base: Any = None, sky_system: Any = None,
                 chunk_provider: Any = None, lighting_pipeline: Any = None,
                 bus: Any = None) -> None:
        super().__init__()
        self.base = base
        self.sky_system = sky_system
        self.chunk_provider = chunk_provider
        self.lighting_pipeline = lighting_pipeline
        self.bus = bus

        self._mode: str = "off"
        self._occlusion: bool = True
        self._time_s: float = 0.0

        # Cover heightmap (headless) + its GPU texture.
        self._cover: RainCoverField | None = None
        self._cover_tex: Texture | None = None
        self._dirty_columns: set[tuple[int, int]] = set()
        self._cover_committed: bool = False     # has a recenter committed once?
        self._recenter_threshold_m: float = 0.0

        # Render nodes.
        self._particle_node: NodePath | None = None
        self._cyl_root: NodePath | None = None
        self._cyl_layers: list = []             # (NodePath, scroll_mult)
        self._cyl_scroll: list[float] = []
        self._cyl_visible: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the cover heightmap + the selected rain mode's nodes (once)."""
        if self.base is None:
            _log.warning("RainRendererComponent: missing base — disabled")
            self.enabled = False
            return
        cfg = self.base._config
        self._mode = str(getattr(cfg, "gfx_rain_mode", "particles")).lower()
        self._occlusion = bool(getattr(cfg, "gfx_rain_occlusion", True))

        if self._mode == "off":
            _log.info("Rain disabled (gfx_rain_mode = \"off\")")
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning("RainRendererComponent: GPU lighting pipeline required "
                         "(lighting_backend = \"gpu\") — disabled")
            self.enabled = False
            return

        # Cover heightmap + its texture.  Recenter threshold = a quarter span so
        # the player can roam ~64 m before a full rebuild (cheap; budgeted refolds
        # handle edits between recenters).
        self._cover = RainCoverField(cfg)
        self._recenter_threshold_m = 0.25 * self._cover.span_m
        tex = Texture("rain_cover")
        # Single-channel float32 heightmap (world Z meters).  Nearest-filtered:
        # the cull wants the exact column height, not a blend across the roof edge.
        tex.setup_2d_texture(self._cover.cells, self._cover.cells,
                             Texture.T_float, Texture.F_r32)
        tex.set_minfilter(SamplerState.FT_nearest)
        tex.set_magfilter(SamplerState.FT_nearest)
        tex.set_wrap_u(SamplerState.WM_clamp)
        tex.set_wrap_v(SamplerState.WM_clamp)
        tex.set_keep_ram_image(False)
        self._cover_tex = tex

        if self._mode == "particles":
            self._build_particles(cfg)
        elif self._mode == "cylinders":
            self._build_cylinders(cfg)
        else:
            _log.warning("RainRendererComponent: unknown gfx_rain_mode %r — "
                         "disabled", self._mode)
            self.enabled = False
            return

        if not self.enabled:           # a build path may have disabled us
            return

        # Bind every shader input the rain node reads ONCE here so the very first
        # rendered frame (before any late_update) has them all present: the cover
        # contract (texture + placeholder origin) and the scalar intensity.  The
        # texels stay at their clear value until the first late_update upload —
        # an all-OPEN_SKY_Z map culls nothing, so rain shows immediately.
        self._cover_tex.set_ram_image(
            np.ascontiguousarray(self._cover.height, dtype=np.float32).tobytes())
        self._bind_cover_uniforms()
        self._push_intensity()

        if self.bus is not None:
            self.bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)
            self.bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)

        _log.info("Rain online: mode=%s, occlusion=%s, cover %dx%d @ %.1f m cells",
                  self._mode, "on" if self._occlusion else "off",
                  self._cover.cells, self._cover.cells, self._cover.cell_m)

    def late_update(self, dt: float) -> None:
        """Advance the clock, refresh the cover heightmap, push per-frame state."""
        if self._cover is None or self._cover_tex is None:
            return
        self._time_s += dt
        if self._particle_node is not None:
            self._particle_node.set_shader_input("u_time_s", self._time_s)

        cam = self._camera_pos()
        self._refresh_cover(cam)
        self._push_intensity()
        if self._mode == "cylinders":
            self._update_cylinders(cam, dt)

    def on_destroy(self) -> None:
        """Unsubscribe and detach all rain nodes."""
        if self.bus is not None:
            self.bus.unsubscribe(ChunkLoadedEvent, self._on_chunk_loaded)
            self.bus.unsubscribe(TerrainEditedEvent, self._on_terrain_edited)
        if self._particle_node is not None:
            self._particle_node.remove_node()
            self._particle_node = None
        if self._cyl_root is not None:
            self._cyl_root.remove_node()
            self._cyl_root = None
        self._cyl_layers.clear()
        self._cover_tex = None
        self._cover = None

    # ------------------------------------------------------------------
    # Cover heightmap upload (committed-origin discipline)
    # ------------------------------------------------------------------

    def _refresh_cover(self, cam: tuple[float, float, float]) -> None:
        """Recenter on a threshold crossing, else refold a budget of columns."""
        cover = self._cover
        chunks = getattr(self.chunk_provider, "chunks", {}) \
            if self.chunk_provider is not None else {}

        ox, oy = cover.origin_m
        cx_center = ox + 0.5 * cover.span_m
        cy_center = oy + 0.5 * cover.span_m
        recenter = (not self._cover_committed
                    or abs(cam[0] - cx_center) > self._recenter_threshold_m
                    or abs(cam[1] - cy_center) > self._recenter_threshold_m)

        changed = False
        if recenter:
            cover.recenter((cam[0], cam[1]))
            cover.rebuild_all(chunks)
            self._dirty_columns.clear()
            self._cover_committed = True
            changed = True
        elif self._dirty_columns:
            budget = int(getattr(self.base._config, "rain_cover_budget_columns", 4))
            take = [self._dirty_columns.pop()
                    for _ in range(min(budget, len(self._dirty_columns)))]
            cover.rebuild_columns(chunks, take)
            changed = True

        if changed:
            # Committed-origin: upload texels + refresh origin in the SAME frame.
            self._cover_tex.set_ram_image(
                np.ascontiguousarray(cover.height, dtype=np.float32).tobytes())
            self._bind_cover_uniforms()

    def _bind_cover_uniforms(self) -> None:
        """Bind the cover texture + origin/cell/cells on the active rain node(s)."""
        ox, oy = self._cover.origin_m
        for node in self._rain_nodes():
            node.set_shader_input("u_rain_height_tex", self._cover_tex)
            node.set_shader_input("u_rain_height_origin",
                                  LVecBase2f(float(ox), float(oy)))
            node.set_shader_input("u_rain_height_cell_m", float(self._cover.cell_m))
            node.set_shader_input("u_rain_height_cells", float(self._cover.cells))

    def _push_intensity(self) -> None:
        """Refresh the scalar rain-intensity fallback (used when wmap is off)."""
        st = getattr(self.sky_system, "state", None)
        ri = float(getattr(st, "rain_intensity", 0.0)) if st is not None else 0.0
        for node in self._rain_nodes():
            node.set_shader_input("u_rain_intensity", ri)

    def _rain_nodes(self) -> list[NodePath]:
        if self._particle_node is not None:
            return [self._particle_node]
        return [layer for layer, _ in self._cyl_layers]

    # ------------------------------------------------------------------
    # Particle mode
    # ------------------------------------------------------------------

    def _build_particles(self, cfg: Any) -> None:
        count = int(getattr(cfg, "gfx_rain_particles", 0))
        if count <= 0:
            _log.info("RainRendererComponent: gfx_rain_particles <= 0 — nothing "
                      "to draw")
            self.enabled = False
            return
        shader = Shader.make(Shader.SL_GLSL,
                             vertex=rain_shaders.RAIN_PARTICLE_VERTEX,
                             fragment=rain_shaders.RAIN_PARTICLE_FRAGMENT)
        geom_node = GeomNode("rain_particles")
        geom_node.add_geom(_build_streak_quad())
        # Parent under terrain_root so wind/fog/camera + the weather-map contract
        # (bound on render, inherited by terrain_root) all arrive automatically.
        node = self.base.terrain_root.attach_new_node(geom_node)
        node.set_shader(shader)            # node-level shader + instance count
        node.set_instance_count(count)
        node.set_shader_input("u_hash_seed", _rain_hash_seed())
        node.set_shader_input("u_rain_box_m", _RAIN_BOX_M)
        node.set_shader_input("u_rain_size_m", _RAIN_SIZE_M)
        node.set_shader_input("u_rain_length_m", _RAIN_LENGTH_M)
        node.set_shader_input("u_rain_fall_mps", _RAIN_FALL_MPS)
        node.set_shader_input("u_rain_intensity", 0.0)
        node.set_shader_input("u_rain_occlusion", 1.0 if self._occlusion else 0.0)
        node.set_shader_input("u_rain_tint", LVecBase3f(*_RAIN_TINT))
        # u_time_s is the shared animation clock.  Grass binds it on ITS own node
        # (grass_root), so it is NOT inherited on terrain_root — bind + refresh
        # our own copy each frame (the mote_renderer split; the camera u_cam_pos
        # IS inherited from terrain_root, so it needs no rebind).
        node.set_shader_input("u_time_s", 0.0)

        node.set_transparency(TransparencyAttrib.M_none)
        node.set_attrib(ColorBlendAttrib.make(
            ColorBlendAttrib.M_add,
            ColorBlendAttrib.O_incoming_alpha,
            ColorBlendAttrib.O_one))
        node.set_depth_write(False)
        node.set_bin("fixed", 0)
        node.set_two_sided(True)

        big = 1.0e9
        geom_node.set_bounds(BoundingBox(LPoint3(-big, -big, -big),
                                         LPoint3(big, big, big)))
        geom_node.set_final(True)
        self._particle_node = node

    # ------------------------------------------------------------------
    # Cylinder mode
    # ------------------------------------------------------------------

    def _build_cylinders(self, cfg: Any) -> None:
        rain_tex = _rain_streak_texture()
        shader = Shader.make(Shader.SL_GLSL,
                             vertex=rain_shaders.RAIN_CYLINDER_VERTEX,
                             fragment=rain_shaders.RAIN_CYLINDER_FRAGMENT)
        # Parent under terrain_root for the inherited weather-map contract.
        root = self.base.terrain_root.attach_new_node("rain_cyl_root")
        for radius_m, scroll_mult in _CYL_LAYERS:
            node = _build_cylinder(radius_m, _CYL_HEIGHT_M, _CYL_SEGMENTS)
            layer = root.attach_new_node(node)
            layer.set_shader(shader)
            layer.set_shader_input("u_rain_tex", rain_tex)
            layer.set_shader_input("u_rain_alpha", 0.0)
            layer.set_shader_input("u_rain_intensity", 0.0)
            layer.set_shader_input("u_rain_occlusion",
                                   1.0 if self._occlusion else 0.0)
            layer.set_shader_input("u_uv_scroll", LVecBase2f(0.0, 0.0))
            layer.set_two_sided(True)
            layer.set_light_off()
            layer.set_depth_write(False)
            layer.set_transparency(TransparencyAttrib.M_none)
            layer.set_attrib(ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_one,
                ColorBlendAttrib.O_one))
            self._cyl_layers.append((layer, scroll_mult))
            self._cyl_scroll.append(0.0)
        root.hide()
        self._cyl_root = root
        self._cyl_visible = False

    def _update_cylinders(self, cam: tuple[float, float, float], dt: float) -> None:
        st = getattr(self.sky_system, "state", None)
        ri = float(getattr(st, "rain_intensity", 0.0)) if st is not None else 0.0
        if ri < _RAIN_HIDE_THRESHOLD:
            if self._cyl_visible:
                self._cyl_root.hide()
                self._cyl_visible = False
            return
        if not self._cyl_visible:
            self._cyl_root.show()
            self._cyl_visible = True

        self._cyl_root.set_pos(cam[0], cam[1], cam[2])
        # Slight wind tilt from the SkyState wind direction.
        wx, wy = getattr(st, "wind_dir", (0.0, 1.0))
        heading = math.degrees(math.atan2(-float(wx), float(wy)))
        tilt = min(_CYL_MAX_TILT_DEG, float(getattr(st, "wind_speed", 0.0)) * 1.1)
        self._cyl_root.set_hpr(heading, tilt, 0.0)

        rate = _CYL_BASE_SCROLL * (0.5 + 1.5 * ri)
        alpha = _clamp01(0.12 + 0.38 * ri)
        for i, (layer, mult) in enumerate(self._cyl_layers):
            self._cyl_scroll[i] = (self._cyl_scroll[i] - rate * mult * dt) % 1.0
            layer.set_shader_input("u_uv_scroll",
                                   LVecBase2f(0.0, self._cyl_scroll[i]))
            layer.set_shader_input("u_rain_alpha", alpha)

    # ------------------------------------------------------------------
    # Helpers + event handlers
    # ------------------------------------------------------------------

    def _camera_pos(self) -> tuple[float, float, float]:
        go = getattr(self.base, "camera_go", None)
        if go is not None:
            p = go.transform.position
            return float(p.x), float(p.y), float(p.z)
        cp = self.base.camera.get_pos(self.base.render)
        return float(cp.x), float(cp.y), float(cp.z)

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        """A chunk streamed in → its column's cover may have changed."""
        self._dirty_columns.add((int(event.coord[0]), int(event.coord[1])))

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        """A brush edit → refold every touched chunk column's cover."""
        coords = event.chunk_coords
        if isinstance(coords, tuple) and len(coords) == 3 and \
                all(isinstance(c, int) for c in coords):
            coords = (coords,)
        for c in coords:
            self._dirty_columns.add((int(c[0]), int(c[1])))


# ---------------------------------------------------------------------------
# Hash seed (Hard Rule 2 — all randomness via for_domain)
# ---------------------------------------------------------------------------

def _rain_hash_seed() -> int:
    """Deterministic rain-streak instance-chain seed via
    ``for_domain("rain", "particles")``.  Bounded to ``[0, 2**31)`` (Panda3D
    passes shader-input ints as signed)."""
    return int(for_domain("rain", "particles").integers(0, 2 ** 31))
