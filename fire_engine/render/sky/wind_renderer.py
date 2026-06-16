"""
world/wind_renderer.py — wind-field GPU upload + uniform binding (render component).

``WindSystemComponent`` is the world-side half of the wind system: it owns the
per-frame orchestration of the headless :class:`~fire_engine.world.wind.WindField`
(``wind/`` is panda3d-free per the hard rule), packs the published snapshot into
a small 2-D float16 texture, and binds the **wind uniform contract** on
``App.terrain_root`` so grass — and later flags/cloth/motes/leaves — sample one
spatially-varying, time-evolving wind velocity field instead of four flat scalar
uniforms.

The contract (bound here, inherited by every node under ``terrain_root``):

    sampler2D u_wind_tex      RGBA16F — R=vx, G=vy, B=turb, A=horizontal speed
                              (m/s; FT_linear, WM_clamp)
    vec2  u_wind_origin       world XY (m) of texel (0,0)'s corner — refreshed
                              ONLY together with a texture upload
    float u_wind_cell_m       cell edge in meters
    float u_wind_cells        cells per axis
    float u_wind_enabled      0.0 (boot default, set in main.py) / 1.0 once the
                              first upload has landed
    float u_time_s            already bound by the grass component (shared clock)

Decode in any shader::

    vec2 uv = (world_xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
    vec4 w  = texture(u_wind_tex, uv);     // R=vx G=vy B=turb A=speed

Like the grass component this is **GPU lighting backend only**: it needs the
``GpuLightingPipeline`` to be live (so the inherited cascade/fog uniforms exist
on ``terrain_root`` and the wind texture rides alongside them).  On the CPU
backend — or if no :class:`WindField` was constructed — it disables itself with
a log line and leaves ``u_wind_enabled = 0.0`` in place, so grass falls back to
its scalar SkyState sway path (``grass.vert`` ``else`` branch).

Committed-origin discipline
---------------------------
``u_wind_origin`` is refreshed **only in the same frame as a texture upload**,
never on a bare recenter — exactly the discipline
``lighting/gpu.py::_commit_assembly_result`` follows for the radiance-cascade
window origins.  If the origin moved but the texels did not (or vice-versa) the
shader would decode the wind UV against a mismatched origin for a frame and the
field would visibly jump.  Since this component packs + uploads + (re)binds the
origin every late_update, they can never disagree.

Texture format
--------------
The texture is ``Texture.T_half_float`` + ``Texture.F_rgba16`` (true half-float,
2 bytes × 4 channels), and :func:`~fire_engine.world.wind.pack_wind_field` produces
exactly that layout: little-endian float16, row-major ``(y, x)``, BGRA.  We are
the FIRST CPU fp16 uploader in the engine — the lighting pipeline's ``rgba16``
radiance textures are GPU-written, never ``set_ram_image``-d from Python — so
the layout is pinned here and in ``pack_wind_field`` together (a wind test
asserts the byte length and channel order).  Note the component type is
``T_half_float`` and **not** ``T_float``: with ``F_rgba16`` Panda3D's ``T_float``
expects a 4-byte (fp32) buffer per channel, whereas ``T_half_float`` expects the
2-byte fp16 buffer ``pack_wind_field`` already produces — so no repack is needed
and ``wind/field.py`` stays untouched.

Filtering
---------
``FT_linear`` min/mag (a deliberate deviation from grass's *nearest*-filtered
field textures): wind is a smooth physical field, and linear filtering is what
makes a gust **glide** across a grass field rather than snapping cell-to-cell at
the 4 m grid boundaries.  ``WM_clamp`` on u+v so blades outside the 256 m window
read the nearest edge velocity (matching ``WindField.sample``'s edge clamp).

Example (wired by main.py)
--------------------------
    wind_go = instantiate()
    wind_go.add_component(
        WindSystemComponent,
        base=app, clock=clock, wind_field=wind_field, worker=venturi_worker,
        sky_system=sky_system, chunk_provider=chunk_manager,
        lighting_pipeline=pipeline, bus=bus)
"""

from __future__ import annotations

import contextlib
from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    LVecBase2f,
    SamplerState,
    Texture,
)

from fire_engine.core import (
    ChunkLoadedEvent,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.render.component import Component
from fire_engine.world.wind import pack_wind_field

__all__ = ["WindSystemComponent"]

_log = get_logger("world.wind")

# Monotonic absolute game time = day * this + game_time_of_day.  Mirrors
# clock.py::_GAME_SECONDS_PER_DAY (module-private there).  Used only to SEED
# the wind clock at start() so a loaded save resumes at a deterministic gust
# phase; per-frame the wind clock accumulates real dt (see late_update).
_GAME_SECONDS_PER_DAY: float = 24.0 * 3600.0


class WindSystemComponent(Component):
    """
    Render component that uploads the wind field and binds its uniforms.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``terrain_root``, ``camera_go`` and
        ``lighting_pipeline``.
    clock : fire_engine.core.Clock
        The shared game clock — used only to SEED the wind clock at start()
        (converted through ``clock.game_time_scale``) so a loaded save resumes
        at a deterministic gust phase.  Per frame the wind clock accumulates
        **real** ``dt`` × ``config.wind_time_scale``: gust travel is an
        aesthetic real-time effect, deliberately independent of the game
        timescale (60× today, 30× later, 1800× on the F7 dev toggle — none of
        which should change how fast gusts sweep the grass).
    wind_field : fire_engine.world.wind.WindField | None
        The headless field.  ``None`` disables the component (grass keeps its
        scalar sway fallback).
    worker : object | None
        The venturi worker (``fire_engine.world.wind.VenturiWorker``), or ``None``
        (identity venturi).  If this component is given the worker it OWNS it
        and stops it in :meth:`on_destroy`.
    sky_system : fire_engine.world.sky.SkySystem
        Read-only weather source; its ``state`` (duck-typed wind/rain/cloud) is
        passed straight into ``WindField.update``.
    chunk_provider : object
        Anything with a ``.chunks`` dict (``ChunkManager``); forwarded to
        ``WindField.update`` for the venturi solver when terrain is dirty.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` (CPU backend)
        disables the component.
    bus : EventBus | None
        Subscribes to ``TerrainEditedEvent`` / ``ChunkLoadedEvent`` to flag the
        venturi field dirty (state-change events only — never per-frame
        plumbing).

    Units: meters, seconds, radians.  World-space Z-up.
    """

    def __init__(
        self,
        base: Any = None,
        clock: Any = None,
        wind_field: Any = None,
        worker: Any = None,
        sky_system: Any = None,
        chunk_provider: Any = None,
        lighting_pipeline: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.clock = clock
        self.wind_field = wind_field
        self.worker = worker
        self.sky_system = sky_system
        self.chunk_provider = chunk_provider
        self.lighting_pipeline = lighting_pipeline
        self.bus = bus

        self._tex: Texture | None = None
        self._cells: int = 0
        self._chunks_dirty: bool = True  # first update feeds chunks (venturi init)
        self._uploaded_once: bool = False
        self._wind_time: float = 0.0  # seeded in start(), then += real dt

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Allocate the wind texture and bind the uniform contract (once)."""
        if self.base is None or self.wind_field is None or self.clock is None:
            _log.warning(
                "WindSystemComponent: missing base/wind_field/clock — "
                "disabled (grass uses scalar sway fallback)"
            )
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "WindSystemComponent: GPU lighting pipeline required "
                '(lighting_backend = "gpu") — disabled (grass uses '
                "scalar sway fallback)"
            )
            self.enabled = False
            return

        cfg = self.base._config
        self._cells = int(cfg.wind_cells)

        # 2-D float16 RGBA texture: pack_wind_field writes exactly this layout
        # (fp16, row-major (y,x), BGRA).  We are the first CPU fp16 uploader —
        # the lighting pipeline's rgba16 textures are GPU-written only.
        #
        # Component type is T_half_float, NOT T_float: with F_rgba16, Panda3D's
        # T_float component width is 4 bytes (it expects an fp32 buffer, 64*64*
        # 4*4 = 64 KB) while T_half_float is 2 bytes (the fp16 buffer
        # pack_wind_field emits, 64*64*4*2 = 32 KB).  Using T_half_float lets us
        # set_ram_image the packed fp16 bytes directly with no repack, keeping
        # wind/field.py untouched.  (F_rgba16 is still the storage format — half
        # precision is plenty for a smooth ±~15 m/s velocity field.)
        tex = Texture("wind_field")
        tex.setup_2d_texture(self._cells, self._cells, Texture.T_half_float, Texture.F_rgba16)
        # FT_linear (NOT nearest like grass's field textures): wind is a smooth
        # physical field — linear filtering makes a gust glide across the grass
        # instead of snapping cell-to-cell at the 4 m grid boundaries.
        tex.set_minfilter(SamplerState.FT_linear)
        tex.set_magfilter(SamplerState.FT_linear)
        # Clamp: blades outside the 256 m window read the edge velocity (matches
        # WindField.sample's out-of-region edge clamp).
        tex.set_wrap_u(SamplerState.WM_clamp)
        tex.set_wrap_v(SamplerState.WM_clamp)
        tex.set_clear_color((0.0, 0.0, 0.0, 0.0))
        tex.set_keep_ram_image(False)
        self._tex = tex

        # Bind on terrain_root: the texture + static grid metadata + a
        # placeholder origin.  u_wind_enabled stays at its main.py boot default
        # (0.0) until the first real upload lands in late_update.
        root = self.base.terrain_root
        root.set_shader_input("u_wind_tex", tex)
        root.set_shader_input("u_wind_cell_m", float(cfg.wind_cell_m))
        root.set_shader_input("u_wind_cells", float(self._cells))
        root.set_shader_input("u_wind_origin", LVecBase2f(0.0, 0.0))

        if self.bus is not None:
            self.bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)
            self.bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)

        # Seed the wind clock from the game clock, converted to real-time
        # seconds through the CURRENT timescale, then scaled by the wind rate.
        # This makes a loaded save resume at a deterministic gust phase while
        # per-frame advancement (below) stays real-time and timescale-free.
        game_s = float(self.clock.game_day) * _GAME_SECONDS_PER_DAY + float(
            self.clock.game_time_of_day
        )
        scale = max(float(self.clock.game_time_scale), 1e-6)
        self._wind_time = game_s / scale * float(cfg.wind_time_scale)

        _log.info(
            "Wind system online: %dx%d field, %.1f m cells (%.0f m "
            "region), wind clock %.2f s/s (real-time, timescale-free)",
            self._cells,
            self._cells,
            float(cfg.wind_cell_m),
            self._cells * float(cfg.wind_cell_m),
            float(cfg.wind_time_scale),
        )

    def late_update(self, dt: float) -> None:
        """Update the field, upload it, and refresh the origin (same frame)."""
        if self._tex is None or self.wind_field is None:
            return

        base = self.base
        # Advance the wind clock by REAL frame time × the configured rate.
        # Deliberately NOT the game clock: gust travel/oscillation are an
        # aesthetic real-time effect, and at game-time pacing a 60× timescale
        # (or the F7 1800× toggle) would sweep gust crests across the grass
        # 60×/1800× too fast.  Monotonic by construction (dt ≥ 0).
        self._wind_time += float(dt) * float(base._config.wind_time_scale)

        sky_state = getattr(self.sky_system, "state", None) if self.sky_system is not None else None

        cam = base.camera_go.transform.position
        cam_pos = (float(cam.x), float(cam.y), float(cam.z))

        # Feed chunks to the venturi solver only when terrain changed (or on the
        # very first update so the worker can initialise) — the heavy solve runs
        # off-thread inside WindField/the worker; passing None most frames keeps
        # the per-frame main-thread cost to just the gust eval + this upload.
        chunks = None
        if self._chunks_dirty and self.chunk_provider is not None:
            chunks = getattr(self.chunk_provider, "chunks", None)
            self._chunks_dirty = False

        self.wind_field.update(dt, self._wind_time, sky_state, cam_pos, chunks=chunks)

        # Pack (64x64 fp16 ~= 32 KB — cheap on the main thread) and upload.
        snap = self.wind_field.snapshot
        self._tex.set_ram_image(pack_wind_field(snap))

        # Committed-origin discipline: refresh u_wind_origin ONLY here, in the
        # same frame as the upload, so texels and origin never disagree.
        root = base.terrain_root
        ox, oy = snap.origin_m
        root.set_shader_input("u_wind_origin", LVecBase2f(float(ox), float(oy)))

        if not self._uploaded_once:
            root.set_shader_input("u_wind_enabled", 1.0)
            self._uploaded_once = True

    def on_destroy(self) -> None:
        """Unsubscribe and stop the venturi worker (this component owns it)."""
        if self.bus is not None:
            self.bus.unsubscribe(TerrainEditedEvent, self._on_terrain_edited)
            self.bus.unsubscribe(ChunkLoadedEvent, self._on_chunk_loaded)
        if self.worker is not None:
            with contextlib.suppress(Exception):
                self.worker.stop(join=True)
            self.worker = None
        self._tex = None

    # ------------------------------------------------------------------
    # Event handlers (flag dirty only — work happens in late_update)
    # ------------------------------------------------------------------

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        """Terrain changed → re-submit a venturi job next update."""
        self._chunks_dirty = True

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        """A new chunk streamed in → re-fold occupancy next update."""
        self._chunks_dirty = True
