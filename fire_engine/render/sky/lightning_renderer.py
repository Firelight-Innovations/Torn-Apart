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
"""

from __future__ import annotations

import math
from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    BoundingBox,
    ColorBlendAttrib,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    InternalName,
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
from fire_engine.world.terrain import RainCoverField
from fire_engine.world.weather import generate_bolt

__all__ = ["LightningRendererComponent"]

_log = get_logger("world.lightning")

#: Speed of sound (m/s) — thunder delay = distance / this.
_SPEED_OF_SOUND_MS: float = 343.0

#: Bolt envelope phase durations (seconds, real time).
_LEADER_S: float = 0.16  # flickering leader reveals the channel top-down
_RETURN_S: float = 0.10  # bright return stroke
_AFTERGLOW_S: float = 0.45  # fading afterglow after the return stroke
_RESTRIKE_GAP_S: float = 0.09  # spacing of seeded restrike pulses

#: HDR brightness of each phase (multiplies the per-segment brightness).
_LEADER_FLASH: float = 1.2
_RETURN_FLASH: float = 6.0
_RESTRIKE_FLASH: float = 3.0

#: Sky/cloud flash-pulse peak (bound as u_lightning_flash; additive, small).
_SKY_FLASH_PEAK: float = 0.9

#: Transient scene-light tuning.
_LIGHT_COLOR: tuple[float, float, float] = (0.80, 0.86, 1.0)  # cool white-blue
_LIGHT_INTENSITY: float = 40.0
_LIGHT_RADIUS_M: float = 260.0
_LIGHT_TTL_S: float = 0.30

#: Ribbon look.
_WIDTH_SCALE_M: float = 0.35  # base ribbon half-width (m) before per-seg width
_CORE_COLOR: tuple[float, float, float] = (0.92, 0.95, 1.0)
_GLOW_COLOR: tuple[float, float, float] = (0.45, 0.60, 1.0)

#: How far the player can roam before the cover heightmap recenters (m).
_POOL_SIZE: int = 2


def _bolt_vertex_format() -> GeomVertexFormat:
    """
    Custom vertex format: position + the segment's other endpoint + ribbon tuple.

    Columns (all float32):
        vertex   (3) — THIS segment endpoint, world XYZ  (p3d_Vertex)
        a_other  (3) — the segment's OTHER endpoint, world XYZ
        a_ribbon (4) — (side -1/+1, alongT 0..1, width, brightness)
    """
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
    """

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

        # Cover heightmap (headless) for roof-aware strike Z.
        self._cover: RainCoverField | None = None
        self._cover_committed: bool = False
        self._recenter_threshold_m: float = 0.0

        # Sky/cloud flash pulse value bound on render each frame.
        self._sky_flash: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the bolt pool + shader and subscribe to strikes (once)."""
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
        """Advance every active bolt's envelope and the sky flash pulse."""
        if self._root is None:
            return
        self._refresh_cover()

        sky_flash = 0.0
        for bolt in self._pool:
            if not bolt.active:
                continue
            self._advance_bolt(bolt, dt)
            sky_flash = max(sky_flash, self._bolt_sky_flash(bolt))

        # Bind the combined sky/cloud flash pulse (additive whitening) on render.
        if sky_flash != self._sky_flash:
            self.base.render.set_shader_input("u_lightning_flash", float(sky_flash))
            self._sky_flash = sky_flash

    def on_destroy(self) -> None:
        """Unsubscribe and detach the bolt nodes."""
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
        gz = self._cover_z(gx, gy)
        ground_z = gz if gz is not None else float(gz_cfg)
        start = (float(event.pos[0]), float(event.pos[1]), float(event.pos[2]))

        bolt_geom = generate_bolt(int(event.seed), start, ground_z, cfg)
        if len(bolt_geom) == 0:
            return

        bolt = self._pool[self._next_bolt % _POOL_SIZE]
        self._next_bolt += 1
        self._upload_bolt(bolt, bolt_geom, float(event.intensity), int(event.seed))

        # Transient scene flash light at the strike point.
        self._add_flash_light((gx, gy, ground_z), float(event.intensity))

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

    def _upload_bolt(self, bolt: _Bolt, geom: Any, intensity: float, seed: int) -> None:
        """Build the ribbon quad soup for a bolt geometry and ignite the node."""
        n = len(geom)
        vdata = GeomVertexData("bolt", self._fmt, Geom.UH_dynamic)
        vdata.set_num_rows(n * 4)
        vw = GeomVertexWriter(vdata, "vertex")
        ow = GeomVertexWriter(vdata, "a_other")
        rw = GeomVertexWriter(vdata, "a_ribbon")

        a = geom.a
        b = geom.b
        width = geom.width
        bright = geom.brightness

        # alongT for each segment = its start-point fraction down the channel
        # (top = 0, ground = 1), driving the top-down reveal.  Use the segment
        # start Z relative to the overall bolt Z span.
        z_top = float(max(a[:, 2].max(), b[:, 2].max()))
        z_bot = float(min(a[:, 2].min(), b[:, 2].min()))
        z_span = max(z_top - z_bot, 1e-3)
        along = (z_top - a[:, 2]) / z_span  # (N,) 0 at top → 1 at bottom

        tris = GeomTriangles(Geom.UH_dynamic)
        for i in range(n):
            ax, ay, az = float(a[i, 0]), float(a[i, 1]), float(a[i, 2])
            bx, by, bz = float(b[i, 0]), float(b[i, 1]), float(b[i, 2])
            w = float(width[i])
            br = float(bright[i])
            t0 = float(along[i])
            # alongT for the b-end uses b's own depth so the ribbon reveals
            # smoothly along its length.
            t1 = float((z_top - b[i, 2]) / z_span)
            # 4 verts: (a,side-1)(a,side+1)(b,side+1)(b,side-1).
            for px, py, pz, ox, oy, oz, side, t in (
                (ax, ay, az, bx, by, bz, -1.0, t0),
                (ax, ay, az, bx, by, bz, +1.0, t0),
                (bx, by, bz, ax, ay, az, +1.0, t1),
                (bx, by, bz, ax, ay, az, -1.0, t1),
            ):
                vw.add_data3(px, py, pz)
                ow.add_data3(ox, oy, oz)
                rw.add_data4(side, t, w, br)
            base = i * 4
            tris.add_vertices(base + 0, base + 1, base + 2)
            tris.add_vertices(base + 0, base + 2, base + 3)

        geom_obj = Geom(vdata)
        geom_obj.add_primitive(tris)
        gn = bolt.node.node()
        gn.remove_all_geoms()
        gn.add_geom(geom_obj)
        big = 1.0e9
        gn.set_bounds(BoundingBox(LPoint3(-big, -big, -big), LPoint3(big, big, big)))
        gn.set_final(True)

        bolt.active = True
        bolt.age_s = 0.0
        bolt.intensity = float(intensity)
        bolt.life_s = _LEADER_S + _RETURN_S + _AFTERGLOW_S
        bolt.channel_len = 1.0
        # One or two seeded restrikes during the afterglow.
        from fire_engine.core.rng import for_domain

        rng = for_domain("weather", "bolt", int(seed), "restrike")
        n_re = int(rng.integers(1, 3))  # 1 or 2
        t_re = _LEADER_S + _RETURN_S
        bolt.restrikes = []
        for _ in range(n_re):
            t_re += _RESTRIKE_GAP_S * float(rng.uniform(1.0, 2.2))
            if t_re < bolt.life_s:
                bolt.restrikes.append(t_re)
        bolt.node.show()
        bolt.node.set_shader_input("u_width_scale", _WIDTH_SCALE_M * (0.7 + 0.6 * intensity))

    # ------------------------------------------------------------------
    # Envelope animation
    # ------------------------------------------------------------------

    def _advance_bolt(self, bolt: _Bolt, dt: float) -> None:
        """Step one bolt's reveal + flash envelope; retire it at end of life."""
        bolt.age_s += dt
        if bolt.age_s >= bolt.life_s:
            bolt.active = False
            bolt.node.hide()
            bolt.node.set_shader_input("u_flash", 0.0)
            return

        reveal, flash = self._envelope(bolt)
        bolt.node.set_shader_input("u_reveal", float(reveal))
        bolt.node.set_shader_input("u_flash", float(flash * (0.5 + bolt.intensity)))

    def _envelope(self, bolt: _Bolt) -> tuple[float, float]:
        """(reveal 0..1, flash HDR) for a bolt at its current age."""
        t = bolt.age_s
        if t < _LEADER_S:
            # Leader: reveal the channel top-down, flickering.
            reveal = t / _LEADER_S
            flicker = 0.6 + 0.4 * abs(math.sin(t * 90.0))
            return reveal, _LEADER_FLASH * flicker
        reveal = 1.0
        tr = t - _LEADER_S
        if tr < _RETURN_S:
            # Return stroke: full channel, bright.
            return reveal, _RETURN_FLASH
        # Afterglow: exponential decay, with seeded restrike spikes.
        glow_t = tr - _RETURN_S
        flash = _RETURN_FLASH * math.exp(-glow_t * 6.0) * 0.5
        for rt in bolt.restrikes:
            d = abs(t - rt)
            if d < 0.04:
                flash = max(flash, _RESTRIKE_FLASH * (1.0 - d / 0.04))
        return reveal, flash

    def _bolt_sky_flash(self, bolt: _Bolt) -> float:
        """The sky/cloud flash-pulse contribution of one bolt this frame."""
        _, flash = self._envelope(bolt)
        # Normalise the bolt flash (peak ~_RETURN_FLASH) to the sky pulse range,
        # scaled by the strike intensity.
        return min(
            _SKY_FLASH_PEAK, _SKY_FLASH_PEAK * (flash / _RETURN_FLASH) * (0.5 + bolt.intensity)
        )

    # ------------------------------------------------------------------
    # Scene flash light
    # ------------------------------------------------------------------

    def _add_flash_light(self, pos: tuple[float, float, float], intensity: float) -> None:
        """Register a short-lived PointLight at the strike (fades via ttl_s)."""
        lights = getattr(self.lighting_pipeline, "lights", None)
        if lights is None:
            return
        from fire_engine.lighting.lights import PointLight

        # Lift the light a little above the strike point so it isn't buried.
        lit_pos = (float(pos[0]), float(pos[1]), float(pos[2]) + 6.0)
        lights.add(
            PointLight(
                position=lit_pos,
                color=_LIGHT_COLOR,
                intensity=_LIGHT_INTENSITY * (0.6 + 0.6 * intensity),
                radius=_LIGHT_RADIUS_M,
                ttl_s=_LIGHT_TTL_S,
            )
        )

    # ------------------------------------------------------------------
    # Cover heightmap (roof-aware strike Z)
    # ------------------------------------------------------------------

    def _refresh_cover(self) -> None:
        """Recenter + rebuild the cover heightmap when the player roams far."""
        cover = self._cover
        if cover is None:
            return
        cam = self._camera_pos()
        ox, oy = cover.origin_m
        cx_center = ox + 0.5 * cover.span_m
        cy_center = oy + 0.5 * cover.span_m
        if (
            not self._cover_committed
            or abs(cam[0] - cx_center) > self._recenter_threshold_m
            or abs(cam[1] - cy_center) > self._recenter_threshold_m
        ):
            chunks = (
                getattr(self.chunk_provider, "chunks", {})
                if self.chunk_provider is not None
                else {}
            )
            cover.recenter((cam[0], cam[1]))
            cover.rebuild_all(chunks)
            self._cover_committed = True

    def _cover_z(self, x: float, y: float) -> float | None:
        """World Z (m) of the cover at world XY, or None if outside / unknown."""
        from fire_engine.world.terrain.rain_cover import OPEN_SKY_Z

        cover = self._cover
        if cover is None or not self._cover_committed:
            return None
        ox, oy = cover.origin_m
        col = math.floor((x - ox) / cover.cell_m)
        row = math.floor((y - oy) / cover.cell_m)
        if 0 <= col < cover.cells and 0 <= row < cover.cells:
            z = float(cover.height[row, col])
            if z > OPEN_SKY_Z * 0.5:  # a real solid voxel (not the sentinel)
                return z
        return None

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
        """A chunk streamed in → force a cover rebuild next frame (roofs change)."""
        self._cover_committed = False

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        """A brush edit → force a cover rebuild next frame."""
        self._cover_committed = False
