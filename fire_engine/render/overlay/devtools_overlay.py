"""
render/overlay/devtools_overlay.py — Panda3D renderer for the in-game developer overlay.

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

Docs: docs/systems/render.overlay.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

# Panda3D imports are allowed in render/ per ARCHITECTURE §3.
from panda3d.core import NodePath, Point3

from fire_engine.core.math3d import Vec3
from fire_engine.devtools import (
    ActionsTool,
    CallbackTool,
    ClockTool,
    DevToolsManager,
    DragState,
    GizmoMode,
    InspectorTool,
    PerformanceTool,
)
from fire_engine.lighting.lights import AreaLight
from fire_engine.render.overlay._overlay_environment import build_environment as _build_environment
from fire_engine.render.overlay._overlay_gizmo import (
    _AXIS_COL,
    _AXIS_DIR,
    _HL_COL,
    _OTHER_AXES,
)
from fire_engine.render.overlay._overlay_gizmo import (
    begin_gizmo as _begin_gizmo,
)
from fire_engine.render.overlay._overlay_gizmo import (
    build_gizmo_panel as _build_gizmo_panel,
)
from fire_engine.render.overlay._overlay_gizmo import (
    update_gizmo as _update_gizmo,
)
from fire_engine.render.overlay._overlay_outline import update_outline as _update_outline
from fire_engine.render.overlay._overlay_panels import (
    clear_widgets as _clear_widgets,
)
from fire_engine.render.overlay._overlay_panels import (
    rebuild as _rebuild,
)
from fire_engine.render.overlay._overlay_spawn import (
    spawn_cube as _spawn_cube,
)
from fire_engine.render.overlay._overlay_spawn import (
    sync_spawned as _sync_spawned,
)
from fire_engine.render.overlay._overlay_spawn import (
    toggle_emissive as _toggle_emissive,
)
from fire_engine.render.overlay._overlay_weather import (
    build_weather_control as _build_weather_control,
)
from fire_engine.render.overlay._overlay_weather import (
    fire_lightning_at_crosshair as _fire_lightning_at_crosshair,
)
from fire_engine.render.overlay._overlay_weather import (
    summon_cell_at_camera as _summon_cell_at_camera,
)
from fire_engine.render.overlay._overlay_weather import (
    toggle_rain_cover_overlay as _toggle_rain_cover_overlay,
)
from fire_engine.world.terrain import raycast_voxel

if TYPE_CHECKING:
    from fire_engine.render.app import App
    from fire_engine.render.gameobject import GameObject

_TERRAIN_RAY_MAX_M = 200.0


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

    Docs: docs/systems/render.overlay.md
    """

    # --- Class-level annotations consumed by helper modules ---------------
    _app: Any
    _base: Any
    _widgets: list[Any]
    _updaters: list[Any]
    _last_sig: tuple[Any, ...] | None
    _outline_np: NodePath | None
    _gizmo_mode: GizmoMode | None
    _gizmo_drag: DragState | None
    _gizmo_go: GameObject | None
    _gizmo_np: NodePath | None
    _spawned: dict[Any, Any]
    _spawn_count: int
    _emissive: dict[Any, tuple[int, AreaLight]]
    _weather: Any
    _rain_cover_np: NodePath | None
    _weather_types: list[Any]
    _wx: int

    # Gizmo constant tables re-exported from _overlay_gizmo
    _AXIS_DIR: ClassVar[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = _AXIS_DIR
    _AXIS_COL: ClassVar[
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ]
    ] = _AXIS_COL
    _HL_COL: ClassVar[tuple[float, float, float, float]] = _HL_COL
    _OTHER_AXES: ClassVar[dict[int, tuple[int, int]]] = _OTHER_AXES

    def __init__(self, app: App, manager: DevToolsManager | None = None) -> None:
        self._app = app
        self._base = app
        self.manager = manager if manager is not None else DevToolsManager()

        self.manager.register_tool(PerformanceTool(self._perf_providers()))
        self.manager.register_tool(ClockTool(app._clock))

        sky = getattr(app, "sky_system", None)
        if sky is not None:
            _sky_ref, _clock_ref = sky, app._clock
            self.manager.register_tool(
                CallbackTool(
                    "environment",
                    "Environment",
                    lambda: _build_environment(self, _sky_ref, _clock_ref),
                )
            )

        self._weather = getattr(sky, "weather", None) if sky is not None else None
        self._rain_cover_np = None
        if self._weather is not None and hasattr(self._weather, "summon_cell"):
            self.manager.register_tool(
                CallbackTool(
                    "env_summon",
                    "Weather Control",
                    lambda: _build_weather_control(self),
                )
            )
            try:
                app.accept("k", lambda: _summon_cell_at_camera(self))
                app.accept("l", lambda: _fire_lightning_at_crosshair(self))
                app.accept("j", lambda: _toggle_rain_cover_overlay(self))
            except Exception:
                pass

        self.manager.register_tool(InspectorTool(self.manager.selection))
        self.manager.register_tool(CallbackTool("gizmo", "Gizmo", lambda: _build_gizmo_panel(self)))
        self.actions = ActionsTool(
            "World",
            {
                "Spawn Cube": lambda: _spawn_cube(self),
                "Toggle Emissive": lambda: _toggle_emissive(self),
            },
        )
        self.manager.register_tool(self.actions)

        self._widgets = []
        self._updaters = []
        self._last_sig = None
        self._outline_np = None
        self._gizmo_mode = GizmoMode.TRANSLATE
        self._gizmo_drag = None
        self._gizmo_go = None
        self._gizmo_np = None
        self._spawned = {}
        self._spawn_count = 0
        self._emissive = {}
        self._weather_types = []
        self._wx = 0
        try:
            from fire_engine.world.sky import WeatherType

            self._weather_types = [*list(WeatherType), None]
            self._wx = len(self._weather_types) - 1
        except Exception:
            pass

        self.manager.selection.set(app.camera_go)
        app.task_mgr.add(self._task, "DevOverlay", sort=50)

    # ------------------------------------------------------------------
    # Performance providers (panda3d-backed; kept out of the headless tool)
    # ------------------------------------------------------------------

    def _perf_providers(self) -> dict[str, Any]:
        """Return a dict of provider callables for the PerformanceTool."""
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
            coord = getattr(go, "coord", None)
            return f"Chunk {tuple(coord)}" if coord is not None else repr(go)

        return {
            "FPS": fps,
            "frame": frame_ms,
            "chunks loaded": chunks,
            "game objects": objects,
            "selected": selected,
        }

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
        self._app.input_state.mouse_captured = not value
        self._app._set_mouse_capture(not value)
        if not value:
            _clear_widgets(self)
            self._last_sig = None

    # ------------------------------------------------------------------
    # World-click handling
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
        if mw.get_over_region() is not None:
            return True
        ray = self._cursor_ray()
        if ray is None:
            return True
        origin, direction = ray
        if _begin_gizmo(self, origin, direction):
            return True
        hit_go = self.manager.pick(origin, direction)
        if hit_go is not None:
            self.manager.selection.set(hit_go)
            return True
        self.manager.selection.set(self._pick_chunk(origin, direction))
        return True

    def _cursor_ray(self) -> tuple[Vec3, Vec3] | None:
        """
        World-space ray ``(origin, direction)`` through the mouse cursor, or
        ``None`` when the window has no mouse.
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
        Voxel-raycast the terrain under a click and return the hit ``Chunk``, or
        ``None`` on a miss.
        """
        cm = getattr(self._app, "chunk_manager", None)
        if cm is None:
            return None
        hit = raycast_voxel(origin, direction, cm.get_or_create, max_distance_m=_TERRAIN_RAY_MAX_M)
        if hit is None:
            return None
        return cm.get_or_create(hit.chunk_coord)

    # ------------------------------------------------------------------
    # Transform gizmo public API
    # ------------------------------------------------------------------

    def end_gizmo_drag(self) -> None:
        """Release an in-progress gizmo drag (bound to ``mouse1-up`` in main)."""
        self._gizmo_drag = None

    # ------------------------------------------------------------------
    # Spawning dev props (public API)
    # ------------------------------------------------------------------

    def spawn_cube(self) -> Any:
        """
        Spawn a 1 m cube 5 m in front of the camera, select it, and make it
        pickable.

        Returns
        -------
        GameObject — the spawned object.
        """
        return _spawn_cube(self)

    def toggle_emissive(self) -> None:
        """
        Toggle the SELECTED spawned prop between inert and emissive.

        Emissive props register an :class:`~fire_engine.lighting.lights.AreaLight`
        matching their world bounds on the GPU lighting pipeline.
        No-op without the GPU lighting backend or with nothing selected.
        """
        _toggle_emissive(self)

    # ------------------------------------------------------------------
    # Per-frame task
    # ------------------------------------------------------------------

    def _task(self, task: Any) -> Any:
        _sync_spawned(self)
        if not self.manager.enabled:
            if self._widgets:
                _clear_widgets(self)
            _update_outline(self)
            _update_gizmo(self)
            return task.cont

        sig = tuple((t.tool_id, t.revision) for t in self.manager.tools)
        if sig != self._last_sig:
            _rebuild(self)
            self._last_sig = sig

        for upd in self._updaters:
            upd()

        _update_outline(self)
        _update_gizmo(self)
        return task.cont
