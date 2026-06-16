"""
world/lightning_renderer.py — LightningRendererComponent: procedural bolts (M7).

The render half of the M7 procedural-lightning feature.  Subscribes to
:class:`~fire_engine.core.event_bus.LightningStrikeEvent` (published by the
headless :class:`~fire_engine.world.weather.WeatherSystem` schedule) and, for each
strike:

1. Regrows the bolt geometry from the event ``seed`` with
   :func:`~fire_engine.world.weather.generate_bolt` (deterministic — the event carries
   only the seed, so every machine draws the same channel).
2. Resolves the strike's ground/roof Z from a rain-cover heightmap (so a bolt
   hits a roof, not the floor under it), falling back to the event's
   ``ground_pos`` Z when no cover is known there.
3. Uploads the bolt segments to one of a small **pool of two** dynamic-geometry
   nodes (so a quick double-strike doesn't clobber the first bolt mid-flash) and
   plays a two-phase envelope over the bolt's short lifetime: a flickering
   **leader** that reveals the channel top-down, then a bright HDR **return
   stroke** + afterglow + one or two seeded **restrikes**.
4. Adds a transient :class:`~fire_engine.lighting.PointLight` (``ttl_s`` ≈ the
   flash length) at the strike so the scene lights up, and pulses a
   ``u_lightning_flash`` uniform on the sky dome + cloud shaders (NEVER an
   exposure change — that would fight auto-adaptation).
5. Re-publishes a :class:`~fire_engine.core.event_bus.ThunderEvent` (distance to
   camera, delay = distance / 343) for the delayed audio crack.

The whole component is gated behind ``config.gfx_lightning_bolts`` (off ⇒ it
disables itself; the headless strike SCHEDULE + ThunderEvents are unaffected —
they live in the weather system).  Like the rain/wind/grass components it is
**GPU lighting backend only** (it needs the live ``GpuLightingPipeline`` for the
flash light + the inherited ``u_cam_pos`` on ``terrain_root``); on the CPU
backend it disables itself with a log line.

Example (wired by main.py)
--------------------------
    light_go = instantiate()
    light_go.add_component(
        LightningRendererComponent,
        base=app, sky_system=sky_system, chunk_provider=chunk_manager,
        lighting_pipeline=pipeline, bus=bus)

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

import math
from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    BoundingBox,
    ColorBlendAttrib,
    GeomNode,
    GeomVertexFormat,
    LPoint3,
    LVecBase3f,
    NodePath,
    Shader,
    TransparencyAttrib,
)

from fire_engine.core import (
    ChunkLoadedEvent,
    LightningStrikeEvent,
    TerrainEditedEvent,
    ThunderEvent,
    get_logger,
)
from fire_engine.render.component import Component
from fire_engine.render.sky import lightning_shaders
from fire_engine.render.sky._impl.cover_events import edited_chunk_columns
from fire_engine.render.sky._impl.lightning_bolt import (
    _WIDTH_SCALE_M,
    add_flash_light,
    advance_bolt,
    bolt_sky_flash,
    cover_z,
    refresh_cover,
    upload_bolt,
)
from fire_engine.world.terrain import RainCoverField
from fire_engine.world.weather import generate_bolt

__all__ = ["LightningRendererComponent"]

_log = get_logger("world.lightning")

#: Speed of sound (m/s) — thunder delay = distance / this.
_SPEED_OF_SOUND_MS: float = 343.0

#: How far the player can roam before the cover heightmap recenters (m).
_POOL_SIZE: int = 2

#: Ribbon look (bolt geometry colours — distinct from the flash point-light colour).
_CORE_COLOR: tuple[float, float, float] = (0.92, 0.95, 1.0)
_GLOW_COLOR: tuple[float, float, float] = (0.45, 0.60, 1.0)


def _bolt_vertex_format() -> GeomVertexFormat:
    """
    Custom vertex format: position + the segment's other endpoint + ribbon tuple.

    Columns (all float32):
        vertex   (3) — THIS segment endpoint, world XYZ  (p3d_Vertex)
        a_other  (3) — the segment's OTHER endpoint, world XYZ
        a_ribbon (4) — (side -1/+1, alongT 0..1, width, brightness)
    """
    from panda3d.core import (
        Geom,
        GeomVertexArrayFormat,
        InternalName,
    )

    arr = GeomVertexArrayFormat()
    arr.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    arr.add_column(InternalName.make("a_other"), 3, Geom.NT_float32, Geom.C_vector)
    arr.add_column(InternalName.make("a_ribbon"), 4, Geom.NT_float32, Geom.C_other)
    fmt = GeomVertexFormat()
    fmt.add_array(arr)
    return GeomVertexFormat.register_format(fmt)


class _Bolt:
    """One pooled dynamic-geometry bolt node + its live animation state."""

    def __init__(self, node: NodePath) -> None:
        self.node = node
        self.node.hide()
        self.active: bool = False
        self.age_s: float = 0.0
        self.life_s: float = 0.0
        self.intensity: float = 0.0
        self.channel_len: float = 1.0  # max alongT span (for reveal speed)
        self.restrikes: list[float] = []  # seeded restrike pulse times (s)


class LightningRendererComponent(Component):
    """
    Render component for M7 procedural lightning bolts.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``render``, ``terrain_root``, ``camera_go``
        and ``_config``.
    sky_system : fire_engine.world.sky.SkySystem
        Read-only; its ``.weather`` is the strike source (events arrive on the
        bus) and ``.state`` is unused here.  Kept for symmetry / future taps.
    chunk_provider : object | None
        Anything with a ``.chunks`` dict (``ChunkManager``) — the loaded chunks
        the rain-cover heightmap folds so bolts hit roofs, not the floor.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` disables the
        component.  Its ``.lights`` (a ``LightSet``) receives the transient
        flash light.
    bus : EventBus | None
        Subscribes to ``LightningStrikeEvent`` (drives bolts) and
        ``ChunkLoadedEvent`` / ``TerrainEditedEvent`` (cover heightmap dirty).
        Publishes ``ThunderEvent``.

    Units: meters, seconds.  World-space Z-up.

    Docs: docs/systems/render.sky.md
    """

    # Class-level annotations for attributes read/written by _impl functions.
    base: Any
    sky_system: Any
    chunk_provider: Any
    lighting_pipeline: Any
    bus: Any
    _root: NodePath | None
    _shader: Shader | None
    _fmt: GeomVertexFormat | None
    _pool: list[_Bolt]
    _next_bolt: int
    _cover: RainCoverField | None
    _cover_committed: bool
    _dirty_columns: set[tuple[int, int]]
    _recenter_threshold_m: float
    _sky_flash: float

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

        self._root: NodePath | None = None
        self._shader: Shader | None = None
        self._fmt: GeomVertexFormat | None = None
        self._pool: list[_Bolt] = []
        self._next_bolt: int = 0

        # Cover heightmap (headless) for roof-aware strike Z.  Mirrors
        # RainRendererComponent's budgeted-refold discipline: chunk/edit events
        # mark columns dirty, and late_update refolds a small per-frame budget of
        # them (never a full O(all-chunks) rebuild_all on every chunk load — that
        # was the per-frame stall flagged in the profiler session).
        self._cover: RainCoverField | None = None
        self._cover_committed: bool = False
        self._dirty_columns: set[tuple[int, int]] = set()
        self._recenter_threshold_m: float = 0.0

        # Sky/cloud flash pulse value bound on render each frame.
        self._sky_flash: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the bolt pool + shader and subscribe to strikes (once).

        Docs: docs/systems/render.sky.md
        """
        if self.base is None:
            _log.warning("LightningRendererComponent: missing base — disabled")
            self.enabled = False
            return
        cfg = self.base._config
        if not bool(getattr(cfg, "gfx_lightning_bolts", False)):
            _log.info(
                "Lightning bolts disabled (gfx_lightning_bolts=false) — "
                "the strike schedule + thunder still run headless"
            )
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "LightningRendererComponent: GPU lighting pipeline "
                'required (lighting_backend="gpu") — disabled'
            )
            self.enabled = False
            return

        # Bolt nodes parent under terrain_root so u_cam_pos is inherited.
        self._root = self.base.terrain_root.attach_new_node("lightning_root")
        self._shader = Shader.make(
            Shader.SL_GLSL,
            vertex=lightning_shaders.LIGHTNING_VERTEX,
            fragment=lightning_shaders.LIGHTNING_FRAGMENT,
        )
        self._fmt = _bolt_vertex_format()

        for i in range(_POOL_SIZE):
            geom_node = GeomNode(f"bolt_{i}")
            node = self._root.attach_new_node(geom_node)
            node.set_shader(self._shader)
            node.set_shader_input("u_reveal", 0.0)
            node.set_shader_input("u_flash", 0.0)
            node.set_shader_input("u_width_scale", _WIDTH_SCALE_M)
            node.set_shader_input("u_core_color", LVecBase3f(*_CORE_COLOR))
            node.set_shader_input("u_glow_color", LVecBase3f(*_GLOW_COLOR))
            # Additive HDR emissive, depth-test on / write off (order-independent).
            node.set_transparency(TransparencyAttrib.M_none)
            node.set_attrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.M_add,
                    ColorBlendAttrib.O_incoming_alpha,
                    ColorBlendAttrib.O_one,
                )
            )
            node.set_depth_write(False)
            node.set_bin("fixed", 0)
            node.set_two_sided(True)
            big = 1.0e9
            geom_node.set_bounds(BoundingBox(LPoint3(-big, -big, -big), LPoint3(big, big, big)))
            geom_node.set_final(True)
            self._pool.append(_Bolt(node))

        # Rain-cover heightmap so strikes hit roofs.  Reuse the M6 field.
        self._cover = RainCoverField(cfg)
        self._recenter_threshold_m = 0.25 * self._cover.span_m

        # Boot-default the sky/cloud flash pulse so the uniform is always defined.
        self.base.render.set_shader_input("u_lightning_flash", 0.0)

        if self.bus is not None:
            self.bus.subscribe(LightningStrikeEvent, self._on_strike)
            self.bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)
            self.bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)

        _log.info(
            "Lightning online: %d-bolt pool, cover %dx%d @ %.1f m cells",
            _POOL_SIZE,
            self._cover.cells,
            self._cover.cells,
            self._cover.cell_m,
        )

    def late_update(self, dt: float) -> None:
        """Advance every active bolt's envelope and the sky flash pulse.

        Docs: docs/systems/render.sky.md
        """
        if self._root is None:
            return
        refresh_cover(self)

        sky_flash = 0.0
        for bolt in self._pool:
            if not bolt.active:
                continue
            advance_bolt(bolt, dt)
            sky_flash = max(sky_flash, bolt_sky_flash(bolt))

        # Bind the combined sky/cloud flash pulse (additive whitening) on render.
        if sky_flash != self._sky_flash:
            self.base.render.set_shader_input("u_lightning_flash", float(sky_flash))
            self._sky_flash = sky_flash

    def on_destroy(self) -> None:
        """Unsubscribe and detach the bolt nodes.

        Docs: docs/systems/render.sky.md
        """
        if self.bus is not None:
            self.bus.unsubscribe(LightningStrikeEvent, self._on_strike)
            self.bus.unsubscribe(ChunkLoadedEvent, self._on_chunk_loaded)
            self.bus.unsubscribe(TerrainEditedEvent, self._on_terrain_edited)
        if self._root is not None:
            self._root.remove_node()
            self._root = None
        self._pool.clear()
        self._cover = None

    # ------------------------------------------------------------------
    # Strike handling
    # ------------------------------------------------------------------

    def _on_strike(self, event: LightningStrikeEvent) -> None:
        """Generate + ignite a bolt for a strike, add a flash light, send thunder."""
        if self._root is None or self.enabled is False:
            return
        cfg = self.base._config

        # Roof-aware ground Z: prefer the cover heightmap under the strike XY.
        gx, gy, gz_cfg = event.ground_pos
        gz = cover_z(self, gx, gy)
        ground_z = gz if gz is not None else float(gz_cfg)
        start = (float(event.pos[0]), float(event.pos[1]), float(event.pos[2]))

        bolt_geom = generate_bolt(int(event.seed), start, ground_z, cfg)
        if len(bolt_geom) == 0:
            return

        bolt = self._pool[self._next_bolt % _POOL_SIZE]
        self._next_bolt += 1
        upload_bolt(self, bolt, bolt_geom, float(event.intensity), int(event.seed))

        # Transient scene flash light at the strike point.
        add_flash_light(self, (gx, gy, ground_z), float(event.intensity))

        # Thunder: distance from the camera → delayed audio crack.
        if self.bus is not None:
            cam = self._camera_pos()
            dist = math.dist(cam, (gx, gy, ground_z))
            self.bus.publish_deferred(
                ThunderEvent(
                    pos=(float(event.pos[0]), float(event.pos[1]), float(event.pos[2])),
                    distance_m=float(dist),
                    delay_s=float(dist) / _SPEED_OF_SOUND_MS,
                    time_abs=float(event.time_abs),
                    intensity=float(event.intensity),
                )
            )

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
        """A chunk streamed in → mark its column dirty (refolded on a budget)."""
        self._dirty_columns.add((int(event.coord[0]), int(event.coord[1])))

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        """A brush edit → mark every touched chunk column dirty."""
        self._dirty_columns.update(edited_chunk_columns(event))
