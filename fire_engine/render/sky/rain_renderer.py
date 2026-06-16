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

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    LVecBase2f,
    NodePath,
    SamplerState,
    Texture,
)

from fire_engine.core import (
    ChunkLoadedEvent,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.render.component import Component
from fire_engine.render.sky._impl.cover_events import edited_chunk_columns
from fire_engine.render.sky._impl.rain_build import (
    build_cylinders,
    build_particles,
    update_cylinders,
)
from fire_engine.world.terrain import RainCoverField

__all__ = ["RainRendererComponent"]

_log = get_logger("world.rain")


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

    Docs: docs/systems/render.sky.md
    """

    # Class-level annotations for attributes read/written by _impl functions.
    base: Any
    sky_system: Any
    chunk_provider: Any
    lighting_pipeline: Any
    bus: Any
    _mode: str
    _occlusion: bool
    _time_s: float
    _cover: RainCoverField | None
    _cover_tex: Texture | None
    _dirty_columns: set[tuple[int, int]]
    _cover_committed: bool
    _recenter_threshold_m: float
    _particle_node: NodePath | None
    _cyl_root: NodePath | None
    _cyl_layers: list[tuple[NodePath, float]]
    _cyl_scroll: list[float]
    _cyl_visible: bool

    def __init__(
        self,
        base: Any = None,
        sky_system: Any = None,
        chunk_provider: Any = None,
        lighting_pipeline: Any = None,
        bus: Any = None,
    ) -> None:
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
        self._cover_committed: bool = False  # has a recenter committed once?
        self._recenter_threshold_m: float = 0.0

        # Render nodes.
        self._particle_node: NodePath | None = None
        self._cyl_root: NodePath | None = None
        self._cyl_layers: list[tuple[NodePath, float]] = []  # (NodePath, scroll_mult)
        self._cyl_scroll: list[float] = []
        self._cyl_visible: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the cover heightmap + the selected rain mode's nodes (once).

        Docs: docs/systems/render.sky.md
        """
        if self.base is None:
            _log.warning("RainRendererComponent: missing base — disabled")
            self.enabled = False
            return
        cfg = self.base._config
        self._mode = str(getattr(cfg, "gfx_rain_mode", "particles")).lower()
        self._occlusion = bool(getattr(cfg, "gfx_rain_occlusion", True))

        if self._mode == "off":
            _log.info('Rain disabled (gfx_rain_mode = "off")')
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "RainRendererComponent: GPU lighting pipeline required "
                '(lighting_backend = "gpu") — disabled'
            )
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
        tex.setup_2d_texture(self._cover.cells, self._cover.cells, Texture.T_float, Texture.F_r32)
        tex.set_minfilter(SamplerState.FT_nearest)
        tex.set_magfilter(SamplerState.FT_nearest)
        tex.set_wrap_u(SamplerState.WM_clamp)
        tex.set_wrap_v(SamplerState.WM_clamp)
        tex.set_keep_ram_image(False)
        self._cover_tex = tex

        if self._mode == "particles":
            build_particles(self, cfg)
        elif self._mode == "cylinders":
            build_cylinders(self, cfg)
        else:
            _log.warning("RainRendererComponent: unknown gfx_rain_mode %r — disabled", self._mode)
            self.enabled = False
            return

        if not self.enabled:  # a build path may have disabled us
            return

        # Bind every shader input the rain node reads ONCE here so the very first
        # rendered frame (before any late_update) has them all present: the cover
        # contract (texture + placeholder origin) and the scalar intensity.  The
        # texels stay at their clear value until the first late_update upload —
        # an all-OPEN_SKY_Z map culls nothing, so rain shows immediately.
        self._cover_tex.set_ram_image(
            np.ascontiguousarray(self._cover.height, dtype=np.float32).tobytes()
        )
        self._bind_cover_uniforms()
        self._push_intensity()

        if self.bus is not None:
            self.bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)
            self.bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)

        _log.info(
            "Rain online: mode=%s, occlusion=%s, cover %dx%d @ %.1f m cells",
            self._mode,
            "on" if self._occlusion else "off",
            self._cover.cells,
            self._cover.cells,
            self._cover.cell_m,
        )

    def late_update(self, dt: float) -> None:
        """Advance the clock, refresh the cover heightmap, push per-frame state.

        Docs: docs/systems/render.sky.md
        """
        if self._cover is None or self._cover_tex is None:
            return
        self._time_s += dt
        if self._particle_node is not None:
            self._particle_node.set_shader_input("u_time_s", self._time_s)

        cam = self._camera_pos()
        self._refresh_cover(cam)
        self._push_intensity()
        if self._mode == "cylinders":
            update_cylinders(self, cam, dt)

    def on_destroy(self) -> None:
        """Unsubscribe and detach all rain nodes.

        Docs: docs/systems/render.sky.md
        """
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
        assert self._cover is not None  # invariant: guarded in late_update before call
        assert self._cover_tex is not None  # invariant: guarded in late_update before call
        cover = self._cover
        chunks = (
            getattr(self.chunk_provider, "chunks", {}) if self.chunk_provider is not None else {}
        )

        ox, oy = cover.origin_m
        cx_center = ox + 0.5 * cover.span_m
        cy_center = oy + 0.5 * cover.span_m
        recenter = (
            not self._cover_committed
            or abs(cam[0] - cx_center) > self._recenter_threshold_m
            or abs(cam[1] - cy_center) > self._recenter_threshold_m
        )

        changed = False
        if recenter:
            cover.recenter((cam[0], cam[1]))
            cover.rebuild_all(chunks)
            self._dirty_columns.clear()
            self._cover_committed = True
            changed = True
        elif self._dirty_columns:
            budget = int(getattr(self.base._config, "rain_cover_budget_columns", 4))
            take = [self._dirty_columns.pop() for _ in range(min(budget, len(self._dirty_columns)))]
            cover.rebuild_columns(chunks, take)
            changed = True

        if changed:
            # Committed-origin: upload texels + refresh origin in the SAME frame.
            self._cover_tex.set_ram_image(
                np.ascontiguousarray(cover.height, dtype=np.float32).tobytes()
            )
            self._bind_cover_uniforms()

    def _bind_cover_uniforms(self) -> None:
        """Bind the cover texture + origin/cell/cells on the active rain node(s)."""
        assert self._cover is not None  # invariant: always set before bind is called
        assert self._cover_tex is not None  # invariant: always set before bind is called
        ox, oy = self._cover.origin_m
        for node in self._rain_nodes():
            node.set_shader_input("u_rain_height_tex", self._cover_tex)
            node.set_shader_input("u_rain_height_origin", LVecBase2f(float(ox), float(oy)))
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
        self._dirty_columns.update(edited_chunk_columns(event))
