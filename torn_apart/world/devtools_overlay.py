"""
world/devtools_overlay.py — Panda3D renderer for the in-game developer overlay.

This is the *only* panda3d-touching half of the dev tools (CLAUDE.md hard rule
1).  It is a thin presentation layer over the headless
:class:`~torn_apart.devtools.manager.DevToolsManager`:

  - turns each tool's :class:`~torn_apart.devtools.fields.Panel` into DirectGUI
    widgets (rebuilt only when a tool's ``revision`` changes; values refreshed
    every frame),
  - applies edits the user types straight back through each Field's ``set``,
  - converts a mouse click into a world-space ray and asks the manager to pick
    the object under the cursor (ray/AABB, headless),
  - draws a bright wireframe box around the selected object, and
  - spawns simple primitive props (rgbCube) the owner can select / move / edit.

It owns no editor *logic* — swapping this file for a Dear ImGui backend later
would not touch ``torn_apart/devtools/`` at all.

Toggle the overlay with **F1** (also frees the mouse cursor so you can click
panels and objects; closing it re-captures the mouse for free-look).  While the
overlay is open and the cursor is free, **left-click selects** an object; while
flying (cursor captured) left-click keeps its normal in-game meaning.

Coordinate conversions (math3d ↔ Panda3D) happen only here and in app.py /
camera.py, per the world-layer boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

# Panda3D imports are allowed in world/ per ARCHITECTURE §3.
from direct.gui.DirectGui import (  # type: ignore[import]
    DirectFrame,
    DirectLabel,
    DirectButton,
    DirectEntry,
)
from direct.gui import DirectGuiGlobals as DGG  # type: ignore[import]
from panda3d.core import (  # type: ignore[import]
    LineSegs,
    Point3,
    LQuaternionf,
    TextNode,
    NodePath,
)

from torn_apart.core.math3d import Vec3
from torn_apart.terrain import raycast_voxel
from torn_apart.world.registry import instantiate
from torn_apart.devtools import (
    DevToolsManager,
    PerformanceTool,
    InspectorTool,
    ActionsTool,
    ClockTool,
    CallbackTool,
    FieldKind,
    Field,
    Section,
    Button,
    Gizmo,
    GizmoMode,
    update_drag,
    is_chunk,
)

if TYPE_CHECKING:
    from torn_apart.world.app import App
    from torn_apart.world.gameobject import GameObject


# --- Layout constants (aspect2d units) -------------------------------------
_TEXT_SCALE = 0.040
_ROW_H = 0.052
_PANEL_W = 0.64          # left column panel width
_INSPECTOR_W = 0.74      # right column panel width
_MARGIN_X = 0.04
_TOP_Z = -0.06
_LABEL_COL = 0.30        # x offset where the value/control begins within a panel
_ENTRY_SCALE = 0.038

_TERRAIN_RAY_MAX_M = 200.0   # how far a dev click probes for a terrain chunk

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

    def __init__(self, app: "App", manager: "Optional[DevToolsManager]" = None) -> None:
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
            self.manager.register_tool(
                CallbackTool("environment", "Environment",
                             lambda s=sky: self._build_environment(s, app._clock))
            )
        self.manager.register_tool(InspectorTool(self.manager.selection))
        # Transform gizmo panel (Move / Rotate / Scale / Off) — switches the
        # active manipulator for the selected object.
        self.manager.register_tool(
            CallbackTool("gizmo", "Gizmo", self._build_gizmo_panel)
        )
        self.actions = ActionsTool("World", {
            "Spawn Cube": self.spawn_cube,
            "Toggle Emissive": self.toggle_emissive,
        })
        self.manager.register_tool(self.actions)

        # Widget bookkeeping ------------------------------------------------
        self._widgets: list[object] = []           # DirectGui items to destroy on rebuild
        self._updaters: list = []                  # per-frame value refreshers
        self._last_sig: tuple | None = None        # (tool_id, revision) signature
        self._outline_np: Optional[NodePath] = None

        # Transform gizmo state -------------------------------------------
        self._gizmo_mode: "Optional[GizmoMode]" = GizmoMode.TRANSLATE
        self._gizmo_drag = None                     # DragState | None (active drag)
        self._gizmo_go: "Optional[GameObject]" = None
        self._gizmo_np: Optional[NodePath] = None

        # Spawned prop visuals: GameObject -> NodePath
        self._spawned: "dict[GameObject, NodePath]" = {}
        self._spawn_count = 0
        # Emissive props: GameObject -> (light_id, AreaLight) registered on
        # the GPU lighting pipeline (the cube becomes an emissive box light).
        self._emissive: "dict[GameObject, tuple[int, object]]" = {}

        # Weather-cycle state for the Environment panel (None = natural schedule).
        self._weather_types: list = []
        self._wx = 0
        try:
            from torn_apart.sky import WeatherType
            self._weather_types = list(WeatherType) + [None]
            self._wx = len(self._weather_types) - 1
        except Exception:  # noqa: BLE001 — sky feature may be absent
            pass

        # Default selection so the inspector shows something on first open.
        self.manager.selection.set(app.camera_go)

        # Drive the overlay after the main frame task (camera already synced).
        app.task_mgr.add(self._task, "DevOverlay", sort=50)

    # ------------------------------------------------------------------
    # Performance providers (panda3d-backed; kept out of the headless tool)
    # ------------------------------------------------------------------

    def _perf_providers(self) -> dict:
        base = self._base
        app = self._app

        def fps() -> str:
            return f"{globalClock.get_average_frame_rate():.1f}"  # noqa: F821

        def frame_ms() -> str:
            return f"{globalClock.get_dt() * 1000.0:.1f} ms"  # noqa: F821

        def chunks() -> object:
            cm = getattr(app, "chunk_manager", None)
            return len(cm.chunks) if cm is not None else 0

        def objects() -> object:
            from torn_apart.world.registry import _STATE
            return len(_STATE.objects)

        def selected() -> str:
            go = self.manager.selection.current
            if go is None:
                return "(none)"
            name = getattr(go, "name", None)
            if name is not None:
                return name
            coord = getattr(go, "coord", None)   # a picked terrain chunk
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

    def _build_environment(self, sky, clock):
        """
        Build the Environment panel: editable time-of-day / time-scale plus a
        live read-out of the current weather and sky parameters, with a single
        compact "Cycle Weather" button.

        All engine access is guarded (``getattr`` / ``try``) so a change in the
        concurrent sky API degrades to blanks rather than crashing the overlay.
        """
        def get_tod_hours() -> float:
            return float(getattr(clock, "game_time_of_day", 0.0)) / 3600.0

        def set_tod_hours(h) -> None:
            try:
                clock.game_time_of_day = (float(h) % 24.0) * 3600.0
            except Exception:  # noqa: BLE001
                pass

        def set_scale(v) -> None:
            try:
                clock.game_time_scale = float(v)
            except Exception:  # noqa: BLE001
                pass

        def state_attr(name):
            st = getattr(sky, "state", None)
            return getattr(st, name, None) if st is not None else None

        weather = getattr(sky, "weather", None)

        def weather_name() -> str:
            cur = getattr(weather, "current", None)
            return getattr(cur, "value", "?") if cur is not None else "?"

        sections = [
            Section("Time", [
                Field("time of day", FieldKind.FLOAT, get_tod_hours, set_tod_hours,
                      step=1.0, units="h"),
                Field("time scale", FieldKind.FLOAT,
                      lambda: float(getattr(clock, "game_time_scale", 60.0)),
                      set_scale, step=60.0),
                Field("day", FieldKind.LABEL, lambda: getattr(clock, "game_day", 0)),
            ]),
            Section("Sky", [
                Field("weather", FieldKind.LABEL, weather_name),
                Field("cloud cover", FieldKind.LABEL, lambda: _fmt(state_attr("cloud_coverage"))),
                Field("fog /m", FieldKind.LABEL, lambda: _fmt(state_attr("fog_density"))),
                Field("rain", FieldKind.LABEL, lambda: _fmt(state_attr("rain_intensity"))),
            ]),
        ]
        buttons = []
        if weather is not None and hasattr(weather, "force_weather"):
            buttons.append(Button("Cycle Weather", lambda w=weather: self._cycle_weather(w)))
        return sections, buttons

    def _cycle_weather(self, weather) -> None:
        """Advance the forced-weather override one step (last step = natural)."""
        if not self._weather_types:
            return
        self._wx = (self._wx + 1) % len(self._weather_types)
        try:
            weather.force_weather(self._weather_types[self._wx])
        except Exception:  # noqa: BLE001
            pass

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

    def _cursor_ray(self):
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

    def _pick_chunk(self, origin: Vec3, direction: Vec3):
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
        hit = raycast_voxel(
            origin, direction, cm.get_or_create, max_distance_m=_TERRAIN_RAY_MAX_M
        )
        if hit is None:
            return None
        return cm.get_or_create(hit.chunk_coord)

    # ------------------------------------------------------------------
    # Transform gizmo (Unity-style move / rotate / scale manipulator)
    # ------------------------------------------------------------------

    def _build_gizmo_panel(self):
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

    def _set_gizmo_mode(self, mode: "Optional[GizmoMode]") -> None:
        """Switch the active gizmo tool (``None`` hides the gizmo)."""
        self._gizmo_mode = mode
        self._gizmo_drag = None   # cancel any in-flight drag on a mode switch

    def _gizmo_target(self) -> "Optional[GameObject]":
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

    def _gizmo_pivot_size(self, go) -> "tuple[Vec3, float]":
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
        if go is None:
            return False
        pivot, size = self._gizmo_pivot_size(go)
        giz = Gizmo(pivot, size, self._gizmo_mode)
        handle = giz.pick(origin, direction)
        if handle is None:
            return False
        tf = go.transform
        self._gizmo_drag = giz.begin(
            handle, origin, direction,
            tf.local_position, tf.local_rotation, tf.local_scale,
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
            hovered = Gizmo(pivot, size, self._gizmo_mode).pick(ray[0], ray[1])

        pivot, size = self._gizmo_pivot_size(go)
        self._draw_gizmo(pivot, size, self._gizmo_mode, hovered)

    # -- gizmo geometry -------------------------------------------------

    _AXIS_DIR = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    _AXIS_COL = ((1.0, 0.35, 0.35, 1.0), (0.4, 1.0, 0.4, 1.0), (0.45, 0.6, 1.0, 1.0))
    _HL_COL = (1.0, 1.0, 0.3, 1.0)
    _OTHER_AXES = {0: (1, 2), 1: (2, 0), 2: (0, 1)}

    def _draw_gizmo(self, pivot, size, mode, hovered) -> None:
        from torn_apart.devtools import HandleType

        ls = LineSegs("gizmo")
        ls.set_thickness(2.5)
        px, py, pz = pivot.x, pivot.y, pivot.z

        def is_hot(htype, axis) -> bool:
            return (hovered is not None and hovered.type == htype
                    and (htype == HandleType.UNIFORM or hovered.axis == axis))

        def axis_col(i, htype):
            return self._HL_COL if is_hot(htype, i) else self._AXIS_COL[i]

        if mode in (GizmoMode.TRANSLATE, GizmoMode.SCALE):
            for i, a in enumerate(self._AXIS_DIR):
                ls.set_color(*axis_col(i, HandleType.AXIS))
                ex, ey, ez = px + a[0] * size, py + a[1] * size, pz + a[2] * size
                ls.move_to(px, py, pz)
                ls.draw_to(ex, ey, ez)
                # Tip marker: a small cross (translate arrowhead / scale knob).
                j, k = self._OTHER_AXES[i]
                t = size * 0.12
                jd = self._AXIS_DIR[j]
                kd = self._AXIS_DIR[k]
                for sgn in (t, -t):
                    ls.move_to(ex, ey, ez)
                    ls.draw_to(ex + jd[0] * sgn, ey + jd[1] * sgn, ez + jd[2] * sgn)
                    ls.move_to(ex, ey, ez)
                    ls.draw_to(ex + kd[0] * sgn, ey + kd[1] * sgn, ez + kd[2] * sgn)

        if mode == GizmoMode.TRANSLATE:
            lo, hi = size * 0.2, size * 0.45
            for i in range(3):
                ls.set_color(*(self._HL_COL if is_hot(HandleType.PLANE, i)
                               else self._AXIS_COL[i]))
                j, k = self._OTHER_AXES[i]
                jd, kd = self._AXIS_DIR[j], self._AXIS_DIR[k]
                corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi), (lo, lo)]
                for n, (cj, ck) in enumerate(corners):
                    x = px + jd[0] * cj + kd[0] * ck
                    y = py + jd[1] * cj + kd[1] * ck
                    z = pz + jd[2] * cj + kd[2] * ck
                    (ls.move_to if n == 0 else ls.draw_to)(x, y, z)

        if mode == GizmoMode.SCALE:
            ls.set_color(*(self._HL_COL if (hovered is not None
                           and hovered.type == HandleType.UNIFORM)
                           else (0.9, 0.9, 0.9, 1.0)))
            c = size * 0.1
            box = [(-c, -c), (c, -c), (c, c), (-c, c), (-c, -c)]
            for n, (dx, dz) in enumerate(box):
                (ls.move_to if n == 0 else ls.draw_to)(px + dx, py, pz + dz)

        if mode == GizmoMode.ROTATE:
            import math as _math
            seg = 48
            for i in range(3):
                ls.set_color(*(self._HL_COL if is_hot(HandleType.RING, i)
                               else self._AXIS_COL[i]))
                j, k = self._OTHER_AXES[i]
                jd, kd = self._AXIS_DIR[j], self._AXIS_DIR[k]
                for n in range(seg + 1):
                    ang = 2.0 * _math.pi * n / seg
                    cj, ck = _math.cos(ang) * size, _math.sin(ang) * size
                    x = px + jd[0] * cj + kd[0] * ck
                    y = py + jd[1] * cj + kd[1] * ck
                    z = pz + jd[2] * cj + kd[2] * ck
                    (ls.move_to if n == 0 else ls.draw_to)(x, y, z)

        node = self._base.render.attach_new_node(ls.create())
        node.set_light_off()
        node.set_depth_test(False)   # always visible through geometry
        node.set_depth_write(False)
        node.set_bin("fixed", 110)   # above the selection outline (bin 100)
        self._gizmo_np = node

    # ------------------------------------------------------------------
    # Spawning dev props
    # ------------------------------------------------------------------

    def spawn_cube(self) -> "GameObject":
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

        Emissive props register an :class:`~torn_apart.lighting.lights.AreaLight`
        matching their world bounds on the GPU lighting pipeline — the cube
        becomes a glowing box light feeding the GI flood fill and the froxel
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
        from torn_apart.lighting.lights import AreaLight
        bounds = np_.get_tight_bounds()
        p = go.transform.position
        center = (p.x, p.y, p.z)
        half = (0.5, 0.5, 0.5)
        if bounds is not None:
            mn, mx = bounds
            center = ((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5,
                      (mn.z + mx.z) * 0.5)
            half = (max((mx.x - mn.x) * 0.5, 0.05),
                    max((mx.y - mn.y) * 0.5, 0.05),
                    max((mx.z - mn.z) * 0.5, 0.05))
        light = AreaLight(center=center, half_extents=half,
                          color=(1.0, 0.78, 0.45), intensity=10.0,
                          radius=14.0)
        self._emissive[go] = (pipeline.lights.add(light), light)
        np_.set_color_scale(2.2, 1.8, 1.1, 1.0)   # the prop visibly glows

    def _sync_spawned(self) -> None:
        """
        Mirror each spawned GameObject's transform onto its NodePath, then
        push the props' world AABBs to the lighting pipeline as dynamic
        occluders (shadow casting / god-ray cutting) and keep any emissive
        prop's AreaLight glued to its box.  ``OccluderSet.set_boxes`` is
        change-detected internally, so static props cost nothing.
        """
        pipeline = getattr(self._app, "lighting_pipeline", None)
        boxes: list = []
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
            bounds = np_.get_tight_bounds()   # world AABB (includes rotation)
            if bounds is None:
                continue
            mn, mx = bounds
            boxes.append(((mn.x, mn.y, mn.z), (mx.x, mx.y, mx.z)))
            em = self._emissive.get(go)
            if em is not None:
                light = em[1]
                center = ((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5,
                          (mn.z + mx.z) * 0.5)
                if any(abs(a - b) > 0.01
                       for a, b in zip(center, light.center)):
                    light.center = center
                    light.half_extents = (
                        max((mx.x - mn.x) * 0.5, 0.05),
                        max((mx.y - mn.y) * 0.5, 0.05),
                        max((mx.z - mn.z) * 0.5, 0.05))
                    lights_dirty = True
        if pipeline is not None:
            pipeline.occluders.set_boxes(boxes)
            if lights_dirty:
                pipeline.lights.notify_changed()

    # ------------------------------------------------------------------
    # Per-frame task
    # ------------------------------------------------------------------

    def _task(self, task):
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

    def _selection_aabb(self, go):
        """
        World-space AABB ``(min, max)`` to outline for the current selection.

        A picked terrain chunk outlines its full 16 m cube (origin → origin +
        size); a registered object uses its :class:`Selectable` box; anything
        else (e.g. the camera, which has no box) returns ``None``.
        """
        from torn_apart.devtools import is_chunk

        if is_chunk(go):
            o = go.world_origin
            m = go.chunk_meters
            return (o.x, o.y, o.z), (o.x + m, o.y + m, o.z + m)
        sel = self.manager.find_selectable(go)
        if sel is None:
            return None
        return sel.world_aabb()

    def _draw_box(self, bmin, bmax) -> None:
        x0, y0, z0 = float(bmin[0]), float(bmin[1]), float(bmin[2])
        x1, y1, z1 = float(bmax[0]), float(bmax[1]), float(bmax[2])
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),   # top
            (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
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
            try:
                w.destroy()
            except Exception:  # noqa: BLE001 — defensive teardown
                pass
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

    def _build_panel(self, panel, parent, x: float, z: float, width: float) -> float:
        """Render one panel starting at ``z``; return the z below the panel."""
        top = z
        rows: list = []   # deferred widget creation so the bg frame sits behind

        # Title
        rows.append(("title", panel.title))
        for section in panel.sections:
            if section.title:
                rows.append(("section", section.title))
            for fld in section.fields:
                rows.append(("field", fld))
        if panel.buttons:
            rows.append(("buttons", panel.buttons))

        # Background frame (sized to the row count) — created first so it's behind.
        n_rows = sum(1 if kind != "buttons" else 1 for kind, _ in rows)
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

    def _mk_label(self, parent, x: float, z: float, text: str, fg, scale: float):
        lbl = DirectLabel(
            parent=parent, text=str(text), text_fg=fg, text_scale=scale,
            text_align=TextNode.ALeft, relief=None, pos=(x + 0.02, 0, z),
        )
        self._widgets.append(lbl)
        return lbl

    def _mk_field(self, parent, x: float, z: float, fld, width: float) -> None:
        # Field name on the left.
        self._mk_label(parent, x + 0.01, z, fld.label, _VALUE_FG, _TEXT_SCALE * 0.9)
        vx = x + _LABEL_COL

        if fld.kind == FieldKind.LABEL or fld.read_only:
            val_lbl = self._mk_label(parent, vx, z, _fmt(fld.get()), (0.7, 0.9, 0.7, 1.0),
                                     _TEXT_SCALE * 0.9)
            self._updaters.append(lambda l=val_lbl, f=fld: l.__setitem__("text", _fmt(f.get())))
            return

        if fld.kind == FieldKind.BOOL:
            btn = DirectButton(
                parent=parent, text=self._checkbox(fld.get()), text_scale=_TEXT_SCALE,
                text_align=TextNode.ALeft, relief=None, text_fg=(0.8, 1.0, 0.8, 1.0),
                pos=(vx, 0, z), command=lambda f=fld: f.set(not f.get()),
            )
            self._widgets.append(btn)
            self._updaters.append(
                lambda b=btn, f=fld: b.__setitem__("text", self._checkbox(f.get()))
            )
            return

        if fld.kind == FieldKind.VEC3:
            entries = []
            for i in range(3):
                e = self._mk_entry(parent, vx + i * 0.15, z, width=4)
                entries.append(e)

            def submit(_=None, f=fld, es=entries):
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
            self._updaters.append(
                lambda es=entries, f=fld: self._refresh_vec3(es, f)
            )
            return

        # FLOAT / INT / STRING — single entry
        entry = self._mk_entry(parent, vx, z, width=8)

        def submit_scalar(_=None, f=fld, e=entry):
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

    def _mk_entry(self, parent, x: float, z: float, width: int) -> DirectEntry:
        e = DirectEntry(
            parent=parent, scale=_ENTRY_SCALE, pos=(x, 0, z), width=width,
            numLines=1, initialText="", text_align=TextNode.ALeft,
            frameColor=(0.15, 0.17, 0.2, 0.9), text_fg=(1, 1, 1, 1),
        )
        self._widgets.append(e)
        return e

    def _mk_buttons(self, parent, x: float, z: float, buttons) -> None:
        bx = x + 0.02
        for b in buttons:
            btn = DirectButton(
                parent=parent, text=b.label, text_scale=_TEXT_SCALE * 0.9,
                text_align=TextNode.ALeft, pos=(bx, 0, z),
                frameColor=(0.2, 0.35, 0.5, 0.95), text_fg=(1, 1, 1, 1),
                # FLAT + a thin border: the DirectGui default is a 0.1-unit
                # raised bevel, which on an unscaled button dwarfs the 0.036-tall
                # text and swallows the label. A flat fill sized snugly to the
                # text reads as a clean button.
                relief=DGG.FLAT, borderWidth=(0.01, 0.01),
                command=b.on_click, pad=(0.02, 0.01),
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
        except Exception:  # noqa: BLE001
            return False

    def _refresh_scalar(self, entry: DirectEntry, fld) -> None:
        if self._is_focused(entry):
            return
        entry.set(_fmt(fld.get()))

    def _refresh_vec3(self, entries, fld) -> None:
        vals = fld.get()
        for e, v in zip(entries, vals):
            if self._is_focused(e):
                continue
            e.set(_fmt(float(v)))
