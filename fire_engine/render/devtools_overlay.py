"""
world/devtools_overlay.py — Panda3D renderer for the in-game developer overlay.

This is the *only* panda3d-touching half of the dev tools (CLAUDE.md hard rule
1).  It is a thin presentation layer over the headless
:class:`~fire_engine.devtools.manager.DevToolsManager`:

  - turns each tool's :class:`~fire_engine.devtools.fields.Panel` into DirectGUI
    widgets (rebuilt only when a tool's ``revision`` changes; values refreshed
    every frame),
  - applies edits the user types straight back through each Field's ``set``,
  - converts a mouse click into a world-space ray and asks the manager to pick
    the object under the cursor (ray/AABB, headless),
  - draws a bright wireframe box around the selected object, and
  - spawns simple primitive props (rgbCube) the owner can select / move / edit.

It owns no editor *logic* — swapping this file for a Dear ImGui backend later
would not touch ``fire_engine/devtools/`` at all.

Toggle the overlay with **F1** (also frees the mouse cursor so you can click
panels and objects; closing it re-captures the mouse for free-look).  While the
overlay is open and the cursor is free, **left-click selects** an object; while
flying (cursor captured) left-click keeps its normal in-game meaning.

Coordinate conversions (math3d ↔ Panda3D) happen only here and in app.py /
camera.py, per the world-layer boundary.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any, ClassVar

from direct.gui import DirectGuiGlobals as DGG

# Panda3D imports are allowed in render/ per ARCHITECTURE §3.
from direct.gui.DirectGui import (
    DirectButton,
    DirectEntry,
    DirectFrame,
    DirectLabel,
)
from panda3d.core import (
    LineSegs,
    LQuaternionf,
    NodePath,
    Point3,
    TextNode,
)

from fire_engine.core.math3d import Vec3
from fire_engine.devtools import (
    ActionsTool,
    Button,
    CallbackTool,
    ClockTool,
    DevToolsManager,
    DragState,
    Field,
    FieldKind,
    Gizmo,
    GizmoMode,
    Handle,
    InspectorTool,
    Panel,
    PerformanceTool,
    Section,
    is_chunk,
    update_drag,
)
from fire_engine.lighting.lights import AreaLight
from fire_engine.render.registry import instantiate
from fire_engine.world.terrain import raycast_voxel

if TYPE_CHECKING:
    from fire_engine.render.app import App
    from fire_engine.render.gameobject import GameObject


# --- Layout constants (aspect2d units) -------------------------------------
_TEXT_SCALE = 0.040
_ROW_H = 0.052
_PANEL_W = 0.64  # left column panel width
_INSPECTOR_W = 0.74  # right column panel width
_MARGIN_X = 0.04
_TOP_Z = -0.06
_LABEL_COL = 0.30  # x offset where the value/control begins within a panel
_ENTRY_SCALE = 0.038

_TERRAIN_RAY_MAX_M = 200.0  # how far a dev click probes for a terrain chunk

_OUTLINE_COLOR = (0.40, 1.0, 0.35, 1.0)
_PANEL_BG = (0.05, 0.06, 0.08, 0.74)
_TITLE_FG = (0.55, 0.85, 1.0, 1.0)
_SECTION_FG = (1.0, 0.82, 0.4, 1.0)
_VALUE_FG = (0.92, 0.95, 1.0, 1.0)


def _fmt(value: object) -> str:
    """Compact display string for a scalar field value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


class DevOverlay:
    """
    In-game developer overlay renderer (Panda3D DirectGUI).

    Parameters
    ----------
    app : App
        The application shell; used for the camera, mouse-capture toggle, and
        chunk-manager stats.
    manager : DevToolsManager | None
        The headless tool hub.  If ``None`` a fresh one is created.  The overlay
        registers the built-in tools (Performance, Time, Inspector, World) on it.

    Attributes
    ----------
    manager : DevToolsManager — the headless hub (tools, selection, picking).
    actions : ActionsTool — the "World" action panel; add buttons via
        ``overlay.actions.add_action("label", fn)``.

    Example
    -------
        overlay = DevOverlay(app)
        app.accept("f1", overlay.toggle)
        overlay.actions.add_action("Fire Explosion", explode_at_camera)
    """

    def __init__(self, app: App, manager: DevToolsManager | None = None) -> None:
        self._app = app
        self._base = app
        self.manager = manager if manager is not None else DevToolsManager()

        # Built-in tools ----------------------------------------------------
        self.manager.register_tool(PerformanceTool(self._perf_providers()))
        self.manager.register_tool(ClockTool(app._clock))
        # Environment panel — only when a sky/weather system is wired in
        # (concurrent feature). Editable day/night + weather, exactly where the
        # owner asked these controls to live. Bound defensively so it degrades
        # gracefully if the sky API shifts.
        sky = getattr(app, "sky_system", None)
        if sky is not None:
            _sky_ref = sky
            _clock_ref = app._clock

            def _build_env() -> tuple[list[Section], list[Button]]:
                return self._build_environment(_sky_ref, _clock_ref)

            self.manager.register_tool(
                CallbackTool(
                    "environment",
                    "Environment",
                    _build_env,
                )
            )

        # --- M8: Environment summon panel (weather control) + debug keys. ----
        # Reads sky_system.weather (the spatial WeatherSystem). Built defensively
        # so a concurrent weather-API shift degrades to blanks, never a crash.
        self._weather = getattr(sky, "weather", None) if sky is not None else None
        self._rain_cover_np: NodePath | None = None  # toggle overlay quad
        if self._weather is not None and hasattr(self._weather, "summon_cell"):
            self.manager.register_tool(
                CallbackTool("env_summon", "Weather Control", self._build_weather_control)
            )
            # Debug keys (gated on a weather system being present):
            #   K — stamp a synthetic cell at the camera,
            #   L — fire a LightningStrikeEvent at the crosshair,
            #   J — toggle the rain-cover overlay quad.
            try:
                app.accept("k", self._summon_cell_at_camera)
                app.accept("l", self._fire_lightning_at_crosshair)
                app.accept("j", self._toggle_rain_cover_overlay)
            except Exception:
                pass
        self.manager.register_tool(InspectorTool(self.manager.selection))
        # Transform gizmo panel (Move / Rotate / Scale / Off) — switches the
        # active manipulator for the selected object.
        self.manager.register_tool(CallbackTool("gizmo", "Gizmo", self._build_gizmo_panel))

        def _spawn_cube_action() -> None:
            self.spawn_cube()

        self.actions = ActionsTool(
            "World",
            {
                "Spawn Cube": _spawn_cube_action,
                "Toggle Emissive": self.toggle_emissive,
            },
        )
        self.manager.register_tool(self.actions)

        # Widget bookkeeping ------------------------------------------------
        self._widgets: list[Any] = []  # DirectGui items to destroy on rebuild
        self._updaters: list[Any] = []  # per-frame value refreshers
        self._last_sig: tuple[Any, ...] | None = None  # (tool_id, revision) signature
        self._outline_np: NodePath | None = None

        # Transform gizmo state -------------------------------------------
        self._gizmo_mode: GizmoMode | None = GizmoMode.TRANSLATE
        self._gizmo_drag: DragState | None = None  # active drag
        self._gizmo_go: GameObject | None = None
        self._gizmo_np: NodePath | None = None

        # Spawned prop visuals: GameObject -> NodePath
        self._spawned: dict[GameObject, NodePath] = {}
        self._spawn_count = 0
        # Emissive props: GameObject -> (light_id, AreaLight) registered on
        # the GPU lighting pipeline (the cube becomes an emissive box light).
        self._emissive: dict[GameObject, tuple[int, AreaLight]] = {}

        # Weather-cycle state for the Environment panel (None = natural schedule).
        self._weather_types: list[Any] = []
        self._wx = 0
        try:
            from fire_engine.world.sky import WeatherType

            self._weather_types = [*list(WeatherType), None]
            self._wx = len(self._weather_types) - 1
        except Exception:
            pass

        # Default selection so the inspector shows something on first open.
        self.manager.selection.set(app.camera_go)

        # Drive the overlay after the main frame task (camera already synced).
        app.task_mgr.add(self._task, "DevOverlay", sort=50)

    # ------------------------------------------------------------------
    # Performance providers (panda3d-backed; kept out of the headless tool)
    # ------------------------------------------------------------------

    def _perf_providers(self) -> dict[str, Any]:  # Callable[[], object] values
        app = self._app

        def fps() -> str:
            from panda3d.core import ClockObject

            return f"{ClockObject.get_global_clock().get_average_frame_rate():.1f}"

        def frame_ms() -> str:
            from panda3d.core import ClockObject

            return f"{ClockObject.get_global_clock().get_dt() * 1000.0:.1f} ms"

        def chunks() -> int:
            cm = getattr(app, "chunk_manager", None)
            return len(cm.chunks) if cm is not None else 0

        def objects() -> int:
            from fire_engine.render.registry import _STATE

            return len(_STATE.objects)

        def selected() -> str:
            go = self.manager.selection.current
            if go is None:
                return "(none)"
            name = getattr(go, "name", None)
            if name is not None:
                return str(name)
            coord = getattr(go, "coord", None)  # a picked terrain chunk
            return f"Chunk {tuple(coord)}" if coord is not None else repr(go)

        return {
            "FPS": fps,
            "frame": frame_ms,
            "chunks loaded": chunks,
            "game objects": objects,
            "selected": selected,
        }

    # ------------------------------------------------------------------
    # Environment panel (day/night + weather) — built defensively
    # ------------------------------------------------------------------

    def _build_environment(self, sky: Any, clock: Any) -> tuple[list[Section], list[Button]]:
        """
        Build the Environment panel: editable time-of-day / time-scale plus a
        live read-out of the current weather and sky parameters, with a single
        compact "Cycle Weather" button.

        All engine access is guarded (``getattr`` / ``try``) so a change in the
        concurrent sky API degrades to blanks rather than crashing the overlay.
        """

        def get_tod_hours() -> float:
            return float(getattr(clock, "game_time_of_day", 0.0)) / 3600.0

        def set_tod_hours(h: Any) -> None:
            with contextlib.suppress(Exception):
                clock.game_time_of_day = (float(h) % 24.0) * 3600.0

        def set_scale(v: Any) -> None:
            with contextlib.suppress(Exception):
                clock.game_time_scale = float(v)

        def state_attr(name: str) -> Any:
            st = getattr(sky, "state", None)
            return getattr(st, name, None) if st is not None else None

        weather = getattr(sky, "weather", None)

        def weather_name() -> str:
            cur = getattr(weather, "current", None)
            return str(getattr(cur, "value", "?")) if cur is not None else "?"

        sections = [
            Section(
                "Time",
                [
                    Field(
                        "time of day",
                        FieldKind.FLOAT,
                        get_tod_hours,
                        set_tod_hours,
                        step=1.0,
                        units="h",
                    ),
                    Field(
                        "time scale",
                        FieldKind.FLOAT,
                        lambda: float(getattr(clock, "game_time_scale", 60.0)),
                        set_scale,
                        step=60.0,
                    ),
                    Field("day", FieldKind.LABEL, lambda: getattr(clock, "game_day", 0)),
                ],
            ),
            Section(
                "Sky",
                [
                    Field("weather", FieldKind.LABEL, weather_name),
                    Field(
                        "cloud cover",
                        FieldKind.LABEL,
                        lambda: _fmt(state_attr("cloud_coverage")),
                    ),
                    Field("fog /m", FieldKind.LABEL, lambda: _fmt(state_attr("fog_density"))),
                    Field("rain", FieldKind.LABEL, lambda: _fmt(state_attr("rain_intensity"))),
                ],
            ),
        ]
        buttons: list[Button] = []
        if weather is not None and hasattr(weather, "force_weather"):
            _weather_ref = weather

            def _do_cycle() -> None:
                self._cycle_weather(_weather_ref)

            buttons.append(Button("Cycle Weather", _do_cycle))
        return sections, buttons

    def _cycle_weather(self, weather: Any) -> None:
        """Advance the forced-weather override one step (last step = natural)."""
        if not self._weather_types:
            return
        self._wx = (self._wx + 1) % len(self._weather_types)
        with contextlib.suppress(Exception):
            weather.force_weather(self._weather_types[self._wx])

    # ------------------------------------------------------------------
    # M8 — Weather Control panel (spatial summon API + nearest-cell readout)
    # ------------------------------------------------------------------

    def _camera_xy(self) -> tuple[float, float]:
        """Player/camera world XY (meters) — the summon + readout reference."""
        p = self._app.camera_go.transform.position
        return (float(p.x), float(p.y))

    def _time_abs(self) -> float:
        """Absolute game seconds from the clock (day·86400 + time-of-day)."""
        clk = self._app._clock
        day = int(getattr(clk, "game_day", 0))
        tod = float(getattr(clk, "game_time_of_day", 0.0))
        return day * 86400.0 + tod

    def _wx_local_class(self) -> str:
        """Read-out: local weather class from the WeatherSystem (guarded)."""
        try:
            return str(getattr(self._weather.current, "value", "?"))  # type: ignore[union-attr]
        except Exception:
            return "?"

    def _wx_local_sample(self) -> Any:
        """Return the local WeatherSample at the camera position (guarded)."""
        try:
            return self._weather.sample_local(  # type: ignore[union-attr]
                self._camera_xy(), self._time_abs()
            )
        except Exception:
            return None

    def _wx_nearest(self) -> tuple[Any, float, float, float] | None:
        """(cell, dist_m, bearing_deg, eta_s) for the nearest active cell."""
        try:
            w = self._weather
            cells = list(w.cells)  # type: ignore[union-attr]
            if not cells:
                return None
            t = self._time_abs()
            px, py = self._camera_xy()
            import numpy as _np

            cell = cells[0]  # already nearest-first
            c = cell.center(t, w.synoptic)  # type: ignore[union-attr]
            dx, dy = float(c[0]) - px, float(c[1]) - py
            dist = float(_np.hypot(dx, dy))
            bearing = (math.degrees(math.atan2(dx, dy))) % 360.0  # 0=+Y(N)
            eta = float(w.cell_eta_s(cell, t, (px, py)))  # type: ignore[union-attr]
            return cell, dist, bearing, eta
        except Exception:
            return None

    def _build_weather_control(self) -> tuple[list[Section], list[Button]]:
        """
        Build the "Weather Control" panel: summon buttons + a live read-out of
        the local weather class and the nearest cell's kind / distance / bearing
        / ETA.  Every engine access is guarded so a weather-API shift degrades to
        blanks rather than crashing the overlay.
        """
        sections = [
            Section(
                "Local",
                [
                    Field("class", FieldKind.LABEL, self._wx_local_class),
                    Field(
                        "humidity",
                        FieldKind.LABEL,
                        lambda: (
                            _fmt(getattr(self._wx_local_sample(), "humidity", None))
                            if self._wx_local_sample()
                            else "?"
                        ),
                    ),
                    Field(
                        "wetness",
                        FieldKind.LABEL,
                        lambda: (
                            _fmt(getattr(self._wx_local_sample(), "wetness", None))
                            if self._wx_local_sample()
                            else "?"
                        ),
                    ),
                ],
            ),
            Section(
                "Nearest cell",
                [
                    Field(
                        "kind",
                        FieldKind.LABEL,
                        lambda: (
                            str(getattr(n[0].kind, "value", "?"))
                            if (n := self._wx_nearest())
                            else "(none)"
                        ),
                    ),
                    Field(
                        "distance",
                        FieldKind.LABEL,
                        lambda: f"{n[1]:.0f} m" if (n := self._wx_nearest()) else "-",
                    ),
                    Field(
                        "bearing",
                        FieldKind.LABEL,
                        lambda: f"{n[2]:.0f} deg" if (n := self._wx_nearest()) else "-",
                    ),
                    Field("ETA", FieldKind.LABEL, self._wx_near_eta),
                ],
            ),
        ]
        buttons = [
            Button("Summon Rainstorm", lambda: self._summon("summon_rainstorm")),
            Button("Summon Thunderstorm", lambda: self._summon("summon_thunderstorm")),
            Button("Summon Fog Bank", lambda: self._summon("summon_fog_bank")),
            Button("Clear Skies", self._clear_skies),
        ]
        return sections, buttons

    def _wx_near_eta(self) -> str:
        """ETA string for the nearest weather cell (guarded)."""
        n = self._wx_nearest()
        if not n:
            return "-"
        eta = n[3]
        if not math.isfinite(eta):
            return "receding"
        return f"{eta / 60.0:.1f} min"

    def _summon(self, method_name: str) -> None:
        """Call a WeatherSystem summon wrapper aimed at the camera."""
        w = self._weather
        if w is None:
            return
        with contextlib.suppress(Exception):
            getattr(w, method_name)(time_abs=self._time_abs(), player_pos=self._camera_xy())

    def _clear_skies(self) -> None:
        """Clear summoned cells + suppress the current natural weather."""
        with contextlib.suppress(Exception):
            self._weather.clear_all()  # type: ignore[union-attr]

    def _summon_cell_at_camera(self) -> None:
        """Debug key (K): stamp a synthetic thunderstorm right at the camera."""
        w = self._weather
        if w is None:
            return
        try:
            from fire_engine.world.weather import CellKind

            w.summon_cell(
                CellKind.THUNDERSTORM,
                time_abs=self._time_abs(),
                player_pos=self._camera_xy(),
                upwind_m=0.0,
            )
        except Exception:
            pass

    def _fire_lightning_at_crosshair(self) -> None:
        """
        Debug key (L): publish a :class:`LightningStrikeEvent` at the crosshair.

        Resolves the world point under the camera crosshair (terrain raycast,
        falling back to a point 60 m ahead) and publishes the event on the bus
        per the M7 contract.  The import resolves at boot once M7's event is
        merged into ``core/event_bus`` — this file is excluded from the headless
        suite, so it never needs the event to exist at test time.
        """
        bus = getattr(self._app, "_event_bus", None) or getattr(self._app, "event_bus", None)
        if bus is None:
            return
        try:
            from fire_engine.core.event_bus import LightningStrikeEvent
        except Exception:
            return

        cam_tf = self._app.camera_go.transform
        ground = cam_tf.position + cam_tf.forward * 60.0
        ray = self._cursor_ray()
        if ray is not None:
            hit_pt = self._raycast_ground(*ray)
            if hit_pt is not None:
                ground = hit_pt
        pos = (float(ground.x), float(ground.y), float(cam_tf.position.z))
        ground_pos = (float(ground.x), float(ground.y), float(ground.z))
        try:
            ev = LightningStrikeEvent(
                pos=pos,
                ground_pos=ground_pos,
                seed=int(self._time_abs()) & 0x7FFFFFFF,
                time_abs=self._time_abs(),
                cell_id=-1,
                intensity=1.0,
            )
            # Publish immediately if available, else defer.
            publish = getattr(bus, "publish", None) or getattr(bus, "publish_deferred", None)
            if publish is not None:
                publish(ev)
        except Exception:
            pass

    def _raycast_ground(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        """World point where a ray hits terrain, or ``None`` (voxel raycast)."""
        cm = getattr(self._app, "chunk_manager", None)
        if cm is None:
            return None
        hit = raycast_voxel(origin, direction, cm.get_or_create, max_distance_m=_TERRAIN_RAY_MAX_M)
        if hit is None:
            return None
        result: Vec3 | None = getattr(hit, "world_point", None) or getattr(hit, "point", None)
        return result

    def _toggle_rain_cover_overlay(self) -> None:
        """
        Debug key (J): toggle a translucent quad visualising the rain-cover
        window (``RainCoverField`` — where rain is blocked by roofs/overhangs).

        Draws a flat card spanning the cover field's footprint at the field
        origin; a second press removes it.  Best-effort: no-op if the rain
        component / cover field is not wired in.
        """
        if self._rain_cover_np is not None:
            self._rain_cover_np.remove_node()
            self._rain_cover_np = None
            return
        cover = self._rain_cover_field()
        if cover is None:
            return
        try:
            from panda3d.core import CardMaker

            ox, oy = cover.origin_m
            span = float(cover.cells) * float(cover.cell_m)
            cm = CardMaker("rain_cover_overlay")
            cm.set_frame(0.0, span, 0.0, span)
            node = self._base.render.attach_new_node(cm.generate())
            node.set_pos(float(ox), float(oy), 0.05)  # just above ground
            node.set_p(-90)  # lay flat (XY plane)
            node.set_transparency(True)
            node.set_color(0.2, 0.55, 1.0, 0.28)
            node.set_light_off()
            node.set_two_sided(True)
            self._rain_cover_np = node
        except Exception:
            self._rain_cover_np = None

    def _rain_cover_field(self) -> Any:
        """Locate the headless ``RainCoverField`` owned by the rain component."""
        rain_go = getattr(self._app, "rain_go", None)
        if rain_go is None:
            return None
        try:
            from fire_engine.render.rain_renderer import RainRendererComponent

            comp = rain_go.get_component(RainRendererComponent)
            return getattr(comp, "_cover", None) if comp is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Enable / toggle
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Flip overlay visibility (bound to F1)."""
        self.set_enabled(not self.manager.enabled)

    def set_enabled(self, value: bool) -> None:
        """
        Show or hide the overlay.

        Opening the overlay frees the mouse cursor (so panels/objects are
        clickable); closing it re-captures the cursor for free-look.

        Parameters
        ----------
        value : bool
        """
        self.manager.enabled = value
        # Menu open → cursor free; menu closed → cursor captured for flying.
        self._app.input_state.mouse_captured = not value
        self._app._set_mouse_capture(not value)
        if not value:
            self._clear_widgets()
            self._last_sig = None

    # ------------------------------------------------------------------
    # World-click handling (called by main.py's mouse1 binding)
    # ------------------------------------------------------------------

    def handle_world_click(self) -> bool:
        """
        Try to consume a left-click as a dev selection.

        Returns
        -------
        bool
            ``True`` if the overlay handled the click (so the caller should not
            also treat it as an in-game action).  ``True`` whenever the overlay
            is open with a free cursor — either an object/chunk was picked, the
            world was clicked empty (deselect), or a UI panel absorbed the click.
            ``False`` when the overlay is closed or the cursor is captured (the
            click belongs to gameplay, e.g. the demo explosion).

        Picking order: a gizmo handle on the selected object wins (begins a drag);
        then a registered dev object (spawned cube, etc.); if the ray hits no
        object, it falls back to a voxel raycast against terrain so the owner can
        click a chunk and inspect its properties; a full miss (empty sky)
        deselects.
        """
        if not self.manager.enabled or self._app.input_state.mouse_captured:
            return False

        mw = self._base.mouseWatcherNode
        # Click landed on a DirectGui region (a panel/button) — let the widget
        # handle it; don't also pick the world behind it.
        if mw.get_over_region() is not None:
            return True

        ray = self._cursor_ray()
        if ray is None:
            return True
        origin, direction = ray

        # 0. Grab a transform-gizmo handle on the current selection (start a drag).
        if self._begin_gizmo(origin, direction):
            return True

        # 1. Registered dev object under the cursor (spawned cube, prop, …).
        hit_go = self.manager.pick(origin, direction)
        if hit_go is not None:
            self.manager.selection.set(hit_go)
            return True

        # 2. No object — probe the voxel terrain so a chunk can be inspected.
        self.manager.selection.set(self._pick_chunk(origin, direction))
        return True

    def _cursor_ray(self) -> tuple[Vec3, Vec3] | None:
        """
        World-space ray ``(origin, direction)`` through the mouse cursor, or
        ``None`` when the window has no mouse.  ``direction`` is the near→far
        vector (not normalised) — fine for both AABB/voxel picking and the gizmo
        math, which only need a consistent ray.
        """
        mw = self._base.mouseWatcherNode
        if not mw.has_mouse():
            return None
        m = mw.get_mouse()
        near = Point3()
        far = Point3()
        self._base.camLens.extrude(m, near, far)
        near_w = self._base.render.get_relative_point(self._base.camera, near)
        far_w = self._base.render.get_relative_point(self._base.camera, far)
        origin = Vec3(near_w.x, near_w.y, near_w.z)
        direction = Vec3(far_w.x - near_w.x, far_w.y - near_w.y, far_w.z - near_w.z)
        return origin, direction

    def _pick_chunk(self, origin: Vec3, direction: Vec3) -> Any:
        """
        Voxel-raycast the terrain under a click and return the hit ``Chunk``.

        Returns the loaded/generated chunk containing the first solid voxel the
        ray enters, or ``None`` on a miss (or when no terrain is wired in).  The
        chunk is the same object the streaming manager caches, so re-clicking the
        same chunk is selection-stable (no inspector rebuild).
        """
        cm = getattr(self._app, "chunk_manager", None)
        if cm is None:
            return None
        hit = raycast_voxel(origin, direction, cm.get_or_create, max_distance_m=_TERRAIN_RAY_MAX_M)
        if hit is None:
            return None
        return cm.get_or_create(hit.chunk_coord)

    # ------------------------------------------------------------------
    # Transform gizmo (Unity-style move / rotate / scale manipulator)
    # ------------------------------------------------------------------

    def _build_gizmo_panel(self) -> tuple[list[Section], list[Button]]:
        """Build the Gizmo panel: a current-mode read-out + tool buttons."""

        def mode_label() -> str:
            return self._gizmo_mode.value if self._gizmo_mode is not None else "off"

        sections = [Section("", [Field("tool", FieldKind.LABEL, mode_label)])]
        buttons = [
            Button("Move", lambda: self._set_gizmo_mode(GizmoMode.TRANSLATE)),
            Button("Rotate", lambda: self._set_gizmo_mode(GizmoMode.ROTATE)),
            Button("Scale", lambda: self._set_gizmo_mode(GizmoMode.SCALE)),
            Button("Off", lambda: self._set_gizmo_mode(None)),
        ]
        return sections, buttons

    def _set_gizmo_mode(self, mode: GizmoMode | None) -> None:
        """Switch the active gizmo tool (``None`` hides the gizmo)."""
        self._gizmo_mode = mode
        self._gizmo_drag = None  # cancel any in-flight drag on a mode switch

    def _gizmo_target(self) -> GameObject | None:
        """
        The object the gizmo currently manipulates, or ``None``.

        Only a registered, pickable GameObject qualifies — that excludes the
        camera (no AABB; ``FlyController`` overwrites its rotation anyway) and
        picked terrain chunks (not GameObjects), and requires an active mode.
        """
        if self._gizmo_mode is None:
            return None
        go = self.manager.selection.current
        if go is None or is_chunk(go):
            return None
        if self.manager.find_selectable(go) is None:
            return None
        return go

    def _gizmo_pivot_size(self, go: Any) -> tuple[Vec3, float]:
        """Gizmo pivot (object origin) + a camera-distance-scaled world size."""
        pivot = go.transform.local_position
        cam = self._app.camera_go.transform.position
        dist = (pivot - cam).length
        return pivot, max(dist * 0.14, 0.3)

    def _begin_gizmo(self, origin: Vec3, direction: Vec3) -> bool:
        """
        If a gizmo handle is under the cursor, start dragging it.

        Returns ``True`` (click consumed) when a drag began, so the click does
        not also re-select or deselect.
        """
        go = self._gizmo_target()
        if go is None or self._gizmo_mode is None:
            return False
        pivot, size = self._gizmo_pivot_size(go)
        giz = Gizmo(pivot, size, self._gizmo_mode)
        handle = giz.pick(origin, direction)
        if handle is None:
            return False
        tf = go.transform
        self._gizmo_drag = giz.begin(
            handle,
            origin,
            direction,
            tf.local_position,
            tf.local_rotation,
            tf.local_scale,
        )
        self._gizmo_go = go
        return True

    def end_gizmo_drag(self) -> None:
        """Release an in-progress gizmo drag (bound to ``mouse1-up`` in main)."""
        self._gizmo_drag = None

    def _update_gizmo(self) -> None:
        """Per-frame: apply an active drag and redraw the gizmo (or clear it)."""
        if self._gizmo_np is not None:
            self._gizmo_np.remove_node()
            self._gizmo_np = None

        go = self._gizmo_target() if self.manager.enabled else None
        if go is None:
            self._gizmo_drag = None
            return

        ray = self._cursor_ray()
        hovered = None
        if self._gizmo_drag is not None:
            # Resolve the live drag and write the new pose back to the object.
            if ray is not None:
                pos, rot, scl = update_drag(self._gizmo_drag, ray[0], ray[1])
                tf = go.transform
                tf.local_position = pos
                tf.local_rotation = rot
                tf.local_scale = scl
            hovered = self._gizmo_drag.handle
        elif ray is not None and self._base.mouseWatcherNode.get_over_region() is None:
            # Hover highlight when not dragging and not over a panel.
            pivot, size = self._gizmo_pivot_size(go)
            if self._gizmo_mode is not None:
                hovered = Gizmo(pivot, size, self._gizmo_mode).pick(ray[0], ray[1])

        pivot, size = self._gizmo_pivot_size(go)
        self._draw_gizmo(pivot, size, self._gizmo_mode, hovered)

    # -- gizmo geometry -------------------------------------------------

    _AXIS_DIR: ClassVar[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    _AXIS_COL: ClassVar[
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ]
    ] = ((1.0, 0.35, 0.35, 1.0), (0.4, 1.0, 0.4, 1.0), (0.45, 0.6, 1.0, 1.0))
    _HL_COL: ClassVar[tuple[float, float, float, float]] = (1.0, 1.0, 0.3, 1.0)
    _OTHER_AXES: ClassVar[dict[int, tuple[int, int]]] = {0: (1, 2), 1: (2, 0), 2: (0, 1)}

    def _gizmo_axis_col(
        self, i: int, htype: Any, hovered: Handle | None
    ) -> tuple[float, float, float, float]:
        """Return the per-axis colour (highlighted when the handle is hovered)."""
        from fire_engine.devtools import HandleType

        hot = (
            hovered is not None
            and hovered.type == htype
            and (htype == HandleType.UNIFORM or hovered.axis == i)
        )
        return self._HL_COL if hot else self._AXIS_COL[i]

    def _draw_gizmo_axes(
        self,
        ls: Any,
        px: float,
        py: float,
        pz: float,
        size: float,
        hovered: Handle | None,
    ) -> None:
        """Draw the three axis arrows with cross-tip markers (translate + scale)."""
        from fire_engine.devtools import HandleType

        for i, a in enumerate(self._AXIS_DIR):
            ls.set_color(*self._gizmo_axis_col(i, HandleType.AXIS, hovered))
            ex, ey, ez = px + a[0] * size, py + a[1] * size, pz + a[2] * size
            ls.move_to(px, py, pz)
            ls.draw_to(ex, ey, ez)
            j, k = self._OTHER_AXES[i]
            t = size * 0.12
            jd, kd = self._AXIS_DIR[j], self._AXIS_DIR[k]
            for sgn in (t, -t):
                ls.move_to(ex, ey, ez)
                ls.draw_to(ex + jd[0] * sgn, ey + jd[1] * sgn, ez + jd[2] * sgn)
                ls.move_to(ex, ey, ez)
                ls.draw_to(ex + kd[0] * sgn, ey + kd[1] * sgn, ez + kd[2] * sgn)

    def _draw_gizmo_rings(
        self,
        ls: Any,
        px: float,
        py: float,
        pz: float,
        size: float,
        hovered: Handle | None,
    ) -> None:
        """Draw three rotation rings (one per axis)."""
        from fire_engine.devtools import HandleType

        seg = 48
        for i in range(3):
            ls.set_color(*self._gizmo_axis_col(i, HandleType.RING, hovered))
            j, k = self._OTHER_AXES[i]
            jd, kd = self._AXIS_DIR[j], self._AXIS_DIR[k]
            for n in range(seg + 1):
                ang = 2.0 * math.pi * n / seg
                cj, ck = math.cos(ang) * size, math.sin(ang) * size
                x = px + jd[0] * cj + kd[0] * ck
                y = py + jd[1] * cj + kd[1] * ck
                z = pz + jd[2] * cj + kd[2] * ck
                (ls.move_to if n == 0 else ls.draw_to)(x, y, z)

    def _draw_gizmo(
        self,
        pivot: Vec3,
        size: float,
        mode: GizmoMode | None,
        hovered: Handle | None,
    ) -> None:
        from fire_engine.devtools import HandleType

        ls = LineSegs("gizmo")
        ls.set_thickness(2.5)
        px, py, pz = pivot.x, pivot.y, pivot.z

        if mode in (GizmoMode.TRANSLATE, GizmoMode.SCALE):
            self._draw_gizmo_axes(ls, px, py, pz, size, hovered)

        if mode == GizmoMode.TRANSLATE:
            lo, hi = size * 0.2, size * 0.45
            for i in range(3):
                ls.set_color(*self._gizmo_axis_col(i, HandleType.PLANE, hovered))
                j, k = self._OTHER_AXES[i]
                jd, kd = self._AXIS_DIR[j], self._AXIS_DIR[k]
                corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi), (lo, lo)]
                for n, (cj, ck) in enumerate(corners):
                    x = px + jd[0] * cj + kd[0] * ck
                    y = py + jd[1] * cj + kd[1] * ck
                    z = pz + jd[2] * cj + kd[2] * ck
                    (ls.move_to if n == 0 else ls.draw_to)(x, y, z)

        if mode == GizmoMode.SCALE:
            uni_hot = hovered is not None and hovered.type == HandleType.UNIFORM
            ls.set_color(*(self._HL_COL if uni_hot else (0.9, 0.9, 0.9, 1.0)))
            c = size * 0.1
            box = [(-c, -c), (c, -c), (c, c), (-c, c), (-c, -c)]
            for n, (dx, dz) in enumerate(box):
                (ls.move_to if n == 0 else ls.draw_to)(px + dx, py, pz + dz)

        if mode == GizmoMode.ROTATE:
            self._draw_gizmo_rings(ls, px, py, pz, size, hovered)

        node = self._base.render.attach_new_node(ls.create())
        node.set_light_off()
        node.set_depth_test(False)  # always visible through geometry
        node.set_depth_write(False)
        node.set_bin("fixed", 110)  # above the selection outline (bin 100)
        self._gizmo_np = node

    # ------------------------------------------------------------------
    # Spawning dev props
    # ------------------------------------------------------------------

    def spawn_cube(self) -> GameObject:
        """
        Spawn a 1 m cube 5 m in front of the camera, select it, and make it
        pickable.  The cube has no components — it's a transform you can move and
        edit live in the Inspector (proof the edit round-trip works end to end).

        Returns
        -------
        GameObject — the spawned object.
        """
        cam_tf = self._app.camera_go.transform
        pos = cam_tf.position + cam_tf.forward * 5.0
        go = instantiate(position=pos)
        self._spawn_count += 1
        go.name = f"Cube{self._spawn_count}"
        go.tag = "devspawn"

        model = self._base.loader.load_model("models/misc/rgbCube")
        if model is None or model.is_empty():
            # Fallback: a plain box model name some Panda3D builds ship instead.
            model = self._base.loader.load_model("box")
        if model is not None and not model.is_empty():
            model.set_scale(0.5)  # rgbCube spans -1..1 → a 1 m cube (half-extent 0.5)
            model.reparent_to(self._base.render)
            model.set_light_off()
            self._spawned[go] = model

        self.manager.add_selectable(go, Vec3(0.5, 0.5, 0.5))
        self.manager.selection.set(go)
        return go

    def toggle_emissive(self) -> None:
        """
        Toggle the SELECTED spawned prop between inert and emissive.

        Emissive props register an :class:`~fire_engine.lighting.lights.AreaLight`
        matching their world bounds on the GPU lighting pipeline — the cube
        becomes a glowing box light feeding the GI gather and the froxel
        fog (the emission-map path for dynamic objects).  The visual gets a
        bright warm colour-scale so the prop itself reads as glowing.
        No-op without the GPU lighting backend or with nothing selected.
        """
        go = self.manager.selection.current
        pipeline = getattr(self._app, "lighting_pipeline", None)
        if go is None or go not in self._spawned or pipeline is None:
            return
        np_ = self._spawned[go]
        if go in self._emissive:
            light_id, _ = self._emissive.pop(go)
            pipeline.lights.remove(light_id)
            np_.clear_color_scale()
            return
        bounds = np_.get_tight_bounds()
        p = go.transform.position
        center = (p.x, p.y, p.z)
        half = (0.5, 0.5, 0.5)
        if bounds is not None:
            mn, mx = bounds
            center = ((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.5)
            half = (
                max((mx.x - mn.x) * 0.5, 0.05),
                max((mx.y - mn.y) * 0.5, 0.05),
                max((mx.z - mn.z) * 0.5, 0.05),
            )
        light = AreaLight(
            center=center, half_extents=half, color=(1.0, 0.78, 0.45), intensity=10.0, radius=14.0
        )
        self._emissive[go] = (pipeline.lights.add(light), light)
        np_.set_color_scale(2.2, 1.8, 1.1, 1.0)  # the prop visibly glows

    def _sync_spawned(self) -> None:
        """
        Mirror each spawned GameObject's transform onto its NodePath, then
        push the props' world AABBs to the lighting pipeline as dynamic
        occluders (shadow casting / god-ray cutting) and keep any emissive
        prop's AreaLight glued to its box.  ``OccluderSet.set_boxes`` is
        change-detected internally, so static props cost nothing.
        """
        pipeline = getattr(self._app, "lighting_pipeline", None)
        boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        lights_dirty = False
        for go, np_ in self._spawned.items():
            p = go.transform.position
            q = go.transform.rotation
            s = go.transform.local_scale
            np_.set_pos(p.x, p.y, p.z)
            np_.set_quat(LQuaternionf(q.w, q.x, q.y, q.z))
            np_.set_scale(0.5 * s.x, 0.5 * s.y, 0.5 * s.z)
            if pipeline is None:
                continue
            bounds = np_.get_tight_bounds()  # world AABB (includes rotation)
            if bounds is None:
                continue
            mn, mx = bounds
            boxes.append(((mn.x, mn.y, mn.z), (mx.x, mx.y, mx.z)))
            em = self._emissive.get(go)
            if em is not None:
                light = em[1]
                center = ((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.5)
                if any(abs(a - b) > 0.01 for a, b in zip(center, light.center, strict=True)):
                    light.center = center
                    light.half_extents = (
                        max((mx.x - mn.x) * 0.5, 0.05),
                        max((mx.y - mn.y) * 0.5, 0.05),
                        max((mx.z - mn.z) * 0.5, 0.05),
                    )
                    lights_dirty = True
        if pipeline is not None:
            pipeline.occluders.set_boxes(boxes)
            if lights_dirty:
                pipeline.lights.notify_changed()

    # ------------------------------------------------------------------
    # Per-frame task
    # ------------------------------------------------------------------

    def _task(self, task: Any) -> Any:
        self._sync_spawned()

        if not self.manager.enabled:
            if self._widgets:
                self._clear_widgets()
            self._update_outline()
            self._update_gizmo()
            return task.cont

        # Rebuild widgets only when some tool's structure revision changed.
        sig = tuple((t.tool_id, t.revision) for t in self.manager.tools)
        if sig != self._last_sig:
            self._rebuild()
            self._last_sig = sig

        # Refresh live values every frame.
        for upd in self._updaters:
            upd()

        self._update_outline()
        self._update_gizmo()
        return task.cont

    # ------------------------------------------------------------------
    # Outline of the selected object
    # ------------------------------------------------------------------

    def _update_outline(self) -> None:
        if self._outline_np is not None:
            self._outline_np.remove_node()
            self._outline_np = None

        go = self.manager.selection.current
        if not self.manager.enabled or go is None:
            return
        box = self._selection_aabb(go)
        if box is None:
            return  # selection has no drawable box (e.g. the camera)
        self._draw_box(*box)

    def _selection_aabb(self, go: Any) -> tuple[Any, Any] | None:
        """
        World-space AABB ``(min, max)`` to outline for the current selection.

        A picked terrain chunk outlines its full 16 m cube (origin → origin +
        size); a registered object uses its :class:`Selectable` box; anything
        else (e.g. the camera, which has no box) returns ``None``.
        """
        from fire_engine.devtools import is_chunk

        if is_chunk(go):
            o = go.world_origin
            m = go.chunk_meters
            return (o.x, o.y, o.z), (o.x + m, o.y + m, o.z + m)
        sel = self.manager.find_selectable(go)
        if sel is None:
            return None
        return sel.world_aabb()

    def _draw_box(self, bmin: Any, bmax: Any) -> None:
        x0, y0, z0 = float(bmin[0]), float(bmin[1]), float(bmin[2])
        x1, y1, z1 = float(bmax[0]), float(bmax[1]), float(bmax[2])
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # verticals
        ]
        ls = LineSegs("selection_outline")
        ls.set_color(*_OUTLINE_COLOR)
        ls.set_thickness(2.5)
        for a, b in edges:
            ls.move_to(*corners[a])
            ls.draw_to(*corners[b])
        node = self._base.render.attach_new_node(ls.create())
        # Draw on top so the box is always visible through geometry.
        node.set_light_off()
        node.set_depth_test(False)
        node.set_depth_write(False)
        node.set_bin("fixed", 100)
        self._outline_np = node

    # ------------------------------------------------------------------
    # Widget build / teardown
    # ------------------------------------------------------------------

    def _clear_widgets(self) -> None:
        for w in self._widgets:
            with contextlib.suppress(Exception):
                w.destroy()
        self._widgets.clear()
        self._updaters.clear()

    def _rebuild(self) -> None:
        self._clear_widgets()
        left_z = _TOP_Z
        right_z = _TOP_Z
        for panel in self.manager.panels():
            if panel.tool_id == "inspector":
                parent = self._base.a2dTopRight
                x = -_INSPECTOR_W - _MARGIN_X
                width = _INSPECTOR_W
                right_z = self._build_panel(panel, parent, x, right_z, width)
                right_z -= _ROW_H * 0.6
            else:
                parent = self._base.a2dTopLeft
                x = _MARGIN_X
                width = _PANEL_W
                left_z = self._build_panel(panel, parent, x, left_z, width)
                left_z -= _ROW_H * 0.6

    def _build_panel(self, panel: Panel, parent: Any, x: float, z: float, width: float) -> float:
        """Render one panel starting at ``z``; return the z below the panel."""
        top = z
        rows: list[tuple[str, Any]] = []  # deferred widget creation so the bg frame sits behind

        # Title
        rows.append(("title", panel.title))
        for section in panel.sections:
            if section.title:
                rows.append(("section", section.title))
            rows.extend(("field", fld) for fld in section.fields)
        if panel.buttons:
            rows.append(("buttons", panel.buttons))

        # Background frame (sized to the row count) — created first so it's behind.
        n_rows = len(rows)
        height = n_rows * _ROW_H + 0.04
        bg = DirectFrame(
            parent=parent,
            frameColor=_PANEL_BG,
            frameSize=(x - 0.02, x + width, top - height, top + 0.02),
            state="normal",  # eats clicks so picking ignores the panel area
        )
        self._widgets.append(bg)

        cz = top - _ROW_H + 0.01
        for kind, payload in rows:
            if kind == "title":
                self._mk_label(parent, x, cz, payload, _TITLE_FG, _TEXT_SCALE * 1.05)
            elif kind == "section":
                self._mk_label(parent, x + 0.01, cz, payload, _SECTION_FG, _TEXT_SCALE * 0.95)
            elif kind == "field":
                self._mk_field(parent, x, cz, payload, width)
            elif kind == "buttons":
                self._mk_buttons(parent, x, cz, payload)
            cz -= _ROW_H
        return top - height

    # ------------------------------------------------------------------
    # Row widgets
    # ------------------------------------------------------------------

    def _mk_label(
        self,
        parent: Any,
        x: float,
        z: float,
        text: str,
        fg: Any,
        scale: float,
    ) -> DirectLabel:
        lbl = DirectLabel(
            parent=parent,
            text=str(text),
            text_fg=fg,
            text_scale=scale,
            text_align=TextNode.ALeft,
            relief=None,
            pos=(x + 0.02, 0, z),
        )
        self._widgets.append(lbl)
        return lbl

    def _mk_field(self, parent: Any, x: float, z: float, fld: Field, width: float) -> None:
        # Field name on the left.
        self._mk_label(parent, x + 0.01, z, fld.label, _VALUE_FG, _TEXT_SCALE * 0.9)
        vx = x + _LABEL_COL

        if fld.kind == FieldKind.LABEL or fld.read_only:
            val_lbl = self._mk_label(
                parent, vx, z, _fmt(fld.get()), (0.7, 0.9, 0.7, 1.0), _TEXT_SCALE * 0.9
            )
            self._updaters.append(lambda lbl=val_lbl, f=fld: lbl.__setitem__("text", _fmt(f.get())))
            return

        if fld.kind == FieldKind.BOOL:
            btn = DirectButton(
                parent=parent,
                text=self._checkbox(fld.get()),
                text_scale=_TEXT_SCALE,
                text_align=TextNode.ALeft,
                relief=None,
                text_fg=(0.8, 1.0, 0.8, 1.0),
                pos=(vx, 0, z),
                command=lambda f=fld: f.set(not f.get()),
            )
            self._widgets.append(btn)
            self._updaters.append(
                lambda b=btn, f=fld: b.__setitem__("text", self._checkbox(f.get()))
            )
            return

        if fld.kind == FieldKind.VEC3:
            self._mk_field_vec3(parent, vx, z, fld)
            return

        # FLOAT / INT / STRING — single entry
        self._mk_field_scalar(parent, vx, z, fld)

    def _mk_field_vec3(self, parent: Any, vx: float, z: float, fld: Field) -> None:
        """Build a three-component VEC3 entry row and wire up submit + refresh."""
        entries: list[DirectEntry] = []
        for i in range(3):
            e = self._mk_entry(parent, vx + i * 0.15, z, width=4)
            entries.append(e)

        def submit(_: Any = None, f: Field = fld, es: list[DirectEntry] = entries) -> None:
            if f.set is None:
                return
            try:
                vals = tuple(float(e.get()) for e in es)
            except ValueError:
                return
            f.set(vals)

        for e in entries:
            # Commit on Enter AND on click-off (focus out), so an edit is
            # never silently discarded by leaving the box.
            e["command"] = submit
            e["focusOutCommand"] = submit
        self._updaters.append(lambda es=entries, f=fld: self._refresh_vec3(es, f))

    def _mk_field_scalar(self, parent: Any, vx: float, z: float, fld: Field) -> None:
        """Build a single-value entry row (FLOAT / INT / STRING) and wire it up."""
        entry = self._mk_entry(parent, vx, z, width=8)

        def submit_scalar(_: Any = None, f: Field = fld, e: DirectEntry = entry) -> None:
            if f.set is None:
                return
            txt = e.get()
            try:
                if f.kind == FieldKind.INT:
                    f.set(int(float(txt)))
                elif f.kind == FieldKind.FLOAT:
                    f.set(float(txt))
                else:
                    f.set(txt)
            except ValueError:
                return

        # Commit on Enter AND on click-off (focus out).
        entry["command"] = submit_scalar
        entry["focusOutCommand"] = submit_scalar
        self._updaters.append(lambda e=entry, f=fld: self._refresh_scalar(e, f))

    def _mk_entry(self, parent: Any, x: float, z: float, width: int) -> DirectEntry:
        e = DirectEntry(
            parent=parent,
            scale=_ENTRY_SCALE,
            pos=(x, 0, z),
            width=width,
            numLines=1,
            initialText="",
            text_align=TextNode.ALeft,
            frameColor=(0.15, 0.17, 0.2, 0.9),
            text_fg=(1, 1, 1, 1),
        )
        self._widgets.append(e)
        return e

    def _mk_buttons(self, parent: Any, x: float, z: float, buttons: list[Button]) -> None:
        bx = x + 0.02
        for b in buttons:
            btn = DirectButton(
                parent=parent,
                text=b.label,
                text_scale=_TEXT_SCALE * 0.9,
                text_align=TextNode.ALeft,
                pos=(bx, 0, z),
                frameColor=(0.2, 0.35, 0.5, 0.95),
                text_fg=(1, 1, 1, 1),
                # FLAT + a thin border: the DirectGui default is a 0.1-unit
                # raised bevel, which on an unscaled button dwarfs the 0.036-tall
                # text and swallows the label. A flat fill sized snugly to the
                # text reads as a clean button.
                relief=DGG.FLAT,
                borderWidth=(0.01, 0.01),
                command=b.on_click,
                pad=(0.02, 0.01),
            )
            self._widgets.append(btn)
            bx += 0.02 + len(b.label) * _TEXT_SCALE * 0.62

    # ------------------------------------------------------------------
    # Live-value refresh helpers (skip widgets the user is editing)
    # ------------------------------------------------------------------

    @staticmethod
    def _checkbox(value: bool) -> str:
        return "[x]" if value else "[ ]"

    @staticmethod
    def _is_focused(entry: DirectEntry) -> bool:
        # The PGEntry method is get_focus() — the old is_focused() never existed,
        # so this used to always raise → always return False → the per-frame
        # refresh stomped whatever the user was typing.  With real focus state,
        # a focused entry is left untouched until Enter / click-off commits it.
        try:
            return bool(entry.guiItem.get_focus())
        except Exception:
            return False

    def _refresh_scalar(self, entry: DirectEntry, fld: Field) -> None:
        if self._is_focused(entry):
            return
        entry.set(_fmt(fld.get()))

    def _refresh_vec3(self, entries: list[DirectEntry], fld: Field) -> None:
        vals = fld.get()
        for e, v in zip(entries, vals, strict=False):
            if self._is_focused(e):
                continue
            e.set(_fmt(float(v)))
