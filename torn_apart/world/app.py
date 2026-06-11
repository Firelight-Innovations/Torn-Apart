"""
world/app.py — Panda3D ShowBase wrapper: the main application entry point.

App is the thin shell between Panda3D and the Torn Apart engine.  It owns:
  - The Panda3D window (1280×720, vsync, FPS meter).
  - The per-frame loop that drives Clock → ComponentRegistry → integration hooks → EventBus.
  - Input state collection (keyboard + mouse) exposed to FlyController without panda3d imports.
  - Transform → Panda3D NodePath sync (math3d types → Panda3D types happen HERE only).

Frame loop order (matches ARCHITECTURE.md §4a.1):
  1. Collect input → populate InputState
  2. clock.update(dt)
  3. registry.run_frame(clock)  (awake / start / update / fixed / late)
  4. # integration hook: chunk streaming  (filled by Phase 3)
  5. # integration hook: lighting dirty work  (filled by Phase 4)
  6. event_bus.drain()
  7. Panda3D renders the frame

App never imports game logic; game logic imports from world/ and below.
Panda3D imports are ALLOWED here per ARCHITECTURE.md §3.

Injected Dependencies
---------------------
  config    : Config    — loaded from config.toml at boot
  clock     : Clock     — frame dt + fixed-step accumulator
  event_bus : EventBus  — deferred event queue

Terrain-render injection (set by main.py AFTER construction)
------------------------------------------------------------
The orchestrator (main.py) wires terrain rendering by setting these optional
attributes on the App instance after ``App(...)`` returns (App is allowed to
import terrain/lighting per ARCHITECTURE §4a.2; we use injection so the engine
shell never *requires* terrain to exist — headless tooling can run App bare):

  app.chunk_manager : ChunkManager | None
      Drained each frame: ``stream_frame`` is called, then ``pending_meshes`` are
      converted to Geoms and ``unloaded_this_frame`` Geoms are removed.
  app.light_sampler : Callable | None
      Forwarded to ``stream_frame`` so remeshed chunks bake fresh sunlight.
  app.terrain_root  : NodePath
      Created in ``__init__``; parent of every chunk NodePath.  The procedural
      ground texture is applied here ONCE and lighting is turned off (vertex
      colours already carry baked sunlight).

Call ``app.setup_terrain_rendering(ground_texture)`` once after injecting
``chunk_manager`` to apply the texture and configure the render state.

Example
-------
    from torn_apart.core import load_config, Clock, EventBus
    from torn_apart.world.app import App

    cfg   = load_config()
    bus   = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    app   = App(cfg, clock, bus)
    app.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Panda3D imports allowed in world/ per ARCHITECTURE §3
from direct.showbase.ShowBase import ShowBase  # type: ignore[import]
from panda3d.core import (  # type: ignore[import]
    WindowProperties,
    LPoint3f,
    LQuaternionf,
    AntialiasAttrib,
    NodePath,
)

from torn_apart.world.registry import ComponentRegistry, instantiate
from torn_apart.world.camera import CameraComponent

if TYPE_CHECKING:
    from torn_apart.core.config    import Config
    from torn_apart.core.clock     import Clock
    from torn_apart.core.event_bus import EventBus


# ---------------------------------------------------------------------------
# InputState — panda3d-free snapshot that FlyController reads
# ---------------------------------------------------------------------------

@dataclass
class InputState:
    """
    Snapshot of the current input state passed to FlyController each frame.

    App populates this from Panda3D's key/mouse state before calling
    registry.run_frame.  FlyController reads it without importing panda3d.

    Attributes
    ----------
    move_forward  : bool — W key held
    move_backward : bool — S key held
    move_left     : bool — A key held
    move_right    : bool — D key held
    move_up       : bool — Space key held (or E)
    move_down     : bool — Ctrl key held (or Q)
    sprint        : bool — Shift held (5× speed multiplier)
    mouse_dx      : float — raw mouse delta X since last frame (pixels)
    mouse_dy      : float — raw mouse delta Y since last frame (pixels)
    mouse_captured: bool  — True when the cursor is locked to the window
    escape_pressed: bool  — True on the frame ESC was pressed (toggle mouse capture)
    """
    move_forward:   bool  = False
    move_backward:  bool  = False
    move_left:      bool  = False
    move_right:     bool  = False
    move_up:        bool  = False
    move_down:      bool  = False
    sprint:         bool  = False
    mouse_dx:       float = 0.0
    mouse_dy:       float = 0.0
    mouse_captured: bool  = False
    escape_pressed: bool  = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App(ShowBase):
    """
    Panda3D ShowBase wrapper — owns the window, frame loop, and input collection.

    Parameters
    ----------
    config    : Config    — frozen engine configuration.
    clock     : Clock     — frame clock + fixed-step accumulator.
    event_bus : EventBus  — deferred event bus (drained once per frame).

    Attributes
    ----------
    input_state  : InputState — populated from Panda3D input before each frame.
    camera_go    : GameObject — the camera GameObject (has CameraComponent).
    camera_comp  : CameraComponent — camera sync component.

    Integration hooks (search for "# integration hook" to find them):
      - Chunk streaming (Phase 3): after registry.run_frame, before event_bus.drain
      - Lighting dirty work (Phase 4): after chunk streaming, before event_bus.drain

    Example
    -------
        cfg   = load_config()
        bus   = EventBus()
        clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
        app   = App(cfg, clock, bus)
        app.run()
    """

    def __init__(
        self,
        config:    "Config",
        clock:     "Clock",
        event_bus: "EventBus",
    ) -> None:
        super().__init__()

        self._config    = config
        self._clock     = clock
        self._event_bus = event_bus

        self.input_state = InputState()
        self._escape_was_down = False
        # Skip the first mouse-delta sample after capture is (re)enabled so the
        # cursor's pre-capture position doesn't snap the view on frame 1.
        self._skip_mouse_delta = True
        # Whether the window currently holds OS focus.  Alt-tabbing away drops
        # the relative-mouse / hidden-cursor window properties, so we reassert
        # them on focus regain (see ``windowEvent``).  Starts True: the window
        # opens focused with the cursor captured.
        self._had_focus = True

        # ------------------------------------------------------------------
        # Terrain-render injection slots (set by main.py after construction).
        # Left None so the engine shell can run without terrain (tooling).
        # ------------------------------------------------------------------
        self.chunk_manager = None          # ChunkManager | None
        self.light_sampler = None          # Callable | None
        # GPU volumetric lighting (Phase 4 GPU backend) — set by main.py when
        # config.lighting_backend == "gpu".  Driven in step 6 of the frame
        # task; None keeps the legacy baked-vertex-light path.
        self.lighting_pipeline = None      # GpuLightingPipeline | None
        self.sky_system = None             # SkySystem | None (set by main.py)
        # Per-chunk NodePath bookkeeping: coord -> NodePath under terrain_root.
        self._chunk_nodes: dict[tuple[int, int, int], NodePath] = {}

        # ------------------------------------------------------------------
        # Window setup
        # ------------------------------------------------------------------
        props = WindowProperties()
        props.set_size(1280, 720)
        props.set_title("Torn Apart")
        props.set_fixed_size(False)
        self.win.request_properties(props)

        if config.show_fps:
            self.setFrameRateMeter(True)

        # Vsync — Panda3D uses sync-video property
        self.set_sleep(0.0)   # let OS vsync do its job

        # Anti-aliasing off (retro look)
        self.render.set_antialias(AntialiasAttrib.M_none)

        # ------------------------------------------------------------------
        # Disable Panda3D's default camera controller
        # ------------------------------------------------------------------
        self.disableMouse()

        # Capture the mouse immediately so free-look works the moment the window
        # opens (no need to hunt for ESC first).  ESC toggles capture off again
        # to free the cursor.  The actual cursor lock is requested in
        # _set_mouse_capture; the first delta is skipped via _skip_mouse_delta.
        self.input_state.mouse_captured = True
        self._set_mouse_capture(True)

        # ------------------------------------------------------------------
        # Camera GameObject
        # ------------------------------------------------------------------
        self.camera_go = instantiate()
        self.camera_go.name = "MainCamera"
        from torn_apart.core.math3d import Vec3
        self.camera_go.transform.local_position = Vec3(0.0, -20.0, 10.0)
        self.camera_comp = self.camera_go.add_component(CameraComponent, base=self)

        # ------------------------------------------------------------------
        # Terrain root NodePath
        # ------------------------------------------------------------------
        # Every streamed chunk's GeomNode is parented under this single node.
        # Chunk mesh positions are ABSOLUTE WORLD METERS (see meshing.py:
        # MeshArrays.positions doc — "vertex positions in world meters"), so
        # terrain_root stays at the origin and each chunk NodePath is added with
        # NO per-chunk offset.  Offsetting here would double the world position.
        self.terrain_root: NodePath = self.render.attach_new_node("terrain_root")

        # ------------------------------------------------------------------
        # Key bindings (Panda3D event strings)
        # ------------------------------------------------------------------
        self._key_state: dict[str, bool] = {}
        for key in ["w", "s", "a", "d", "space", "lcontrol", "rcontrol",
                    "lshift", "rshift", "e", "q"]:
            self.accept(key, self._key_down, [key])
            self.accept(key + "-up", self._key_up, [key])
        self.accept("escape", self._on_escape)

        # ------------------------------------------------------------------
        # Register the per-frame task
        # ------------------------------------------------------------------
        self.taskMgr.add(self._frame_task, "TornApartFrame")

    # ------------------------------------------------------------------
    # Input handlers
    # ------------------------------------------------------------------

    def _key_down(self, key: str) -> None:
        self._key_state[key] = True

    def _key_up(self, key: str) -> None:
        self._key_state[key] = False

    def _on_escape(self) -> None:
        """Toggle mouse capture on ESC."""
        self.input_state.escape_pressed = True

    def _collect_input(self) -> None:
        """
        Read current Panda3D input state and write to self.input_state.

        Mouse delta is read from the pointer and recentred.  This is the
        ONLY place panda3d input is read; everything else uses InputState.
        """
        ks = self._key_state
        inp = self.input_state

        inp.move_forward  = bool(ks.get("w",        False))
        inp.move_backward = bool(ks.get("s",        False))
        inp.move_left     = bool(ks.get("a",        False))
        inp.move_right    = bool(ks.get("d",        False))
        inp.move_up       = bool(ks.get("space",    False) or ks.get("e", False))
        inp.move_down     = bool(ks.get("lcontrol", False) or
                                  ks.get("rcontrol", False) or ks.get("q", False))
        inp.sprint        = bool(ks.get("lshift",   False) or ks.get("rshift", False))

        # Toggle mouse capture
        if inp.escape_pressed:
            inp.mouse_captured = not inp.mouse_captured
            self._set_mouse_capture(inp.mouse_captured)
            inp.escape_pressed = False

        # Mouse delta (only when captured).
        #
        # We read the RAW pixel pointer position (``win.get_pointer``) relative
        # to the window centre, then recentre the pointer every frame.  Reading
        # raw pixels (not the normalised mouseWatcher value) keeps BOTH axes
        # symmetric and avoids the edge-clamping that froze one axis under the
        # old confined-cursor + normalised-delta path.  The cursor is in
        # relative mode (see _set_mouse_capture) so the OS never clamps it at a
        # screen edge.  Note: get_pointer Y is pixels-from-TOP (Y-down).
        inp.mouse_dx = 0.0
        inp.mouse_dy = 0.0
        if inp.mouse_captured:
            win_w = self.win.get_x_size()
            win_h = self.win.get_y_size()
            cx = win_w // 2
            cy = win_h // 2
            ptr = self.win.get_pointer(0)
            if ptr.get_in_window() and not self._skip_mouse_delta:
                inp.mouse_dx = float(ptr.get_x() - cx)
                inp.mouse_dy = float(ptr.get_y() - cy)
            # Always recentre so the next frame's delta is measured from centre.
            self.win.move_pointer(0, cx, cy)
            self._skip_mouse_delta = False

    def windowEvent(self, win) -> None:
        """
        Handle Panda3D window events (focus, resize, close).

        Extends ShowBase's default handling to fix a mouse-capture desync: when
        the window loses OS focus (alt-tab), the platform releases our hidden /
        relative-mouse cursor properties.  Panda3D does not re-apply them on
        focus regain, so the engine would think the mouse is captured while the
        OS shows a free, absolute-mode cursor — free-look stays dead until the
        next ESC toggle.  Here we detect the focus-regain edge and reassert
        whatever capture state we want, re-arming the first-frame delta skip so
        the view doesn't snap.

        Parameters
        ----------
        win : panda3d.core.GraphicsWindow
            The window the event is about (ignored unless it is ``self.win``).
        """
        super().windowEvent(win)
        if win is not self.win:
            return
        has_focus = bool(win.get_properties().get_foreground())
        if has_focus and not self._had_focus:
            # Regained focus — reapply the capture state the engine believes in.
            self._set_mouse_capture(self.input_state.mouse_captured)
        self._had_focus = has_focus

    def _set_mouse_capture(self, captured: bool) -> None:
        """
        Lock/unlock the cursor for free-look.

        Captured → cursor hidden + **relative** mouse mode (the OS stops
        clamping the pointer at screen edges, so look never freezes on an axis).
        Released → cursor shown + absolute mode (normal desktop pointer).

        Re-enabling capture arms ``_skip_mouse_delta`` so the first post-capture
        frame doesn't snap the view by the pre-capture pointer offset.
        """
        props = WindowProperties()
        props.set_cursor_hidden(captured)
        props.set_mouse_mode(
            WindowProperties.M_relative if captured else WindowProperties.M_absolute
        )
        self.win.request_properties(props)
        if captured:
            self._skip_mouse_delta = True

    # ------------------------------------------------------------------
    # Frame task
    # ------------------------------------------------------------------

    def _frame_task(self, task):
        """
        Main per-frame driver.

        Order matches ARCHITECTURE.md §4a.1:
          1. Collect input
          2. clock.update(dt)
          3. Push input to FlyController (via InputState; no panda3d in controller)
          4. registry.run_frame(clock)
          5. [integration hook] chunk streaming   ← Phase 3 fills this
          6. [integration hook] lighting dirty    ← Phase 4 fills this
          7. event_bus.drain()
          8. Sync camera transform → Panda3D NodePath
          9. Return task.cont (let Panda3D render)
        """
        real_dt = globalClock.get_dt()  # Panda3D's frame time  # noqa: F821

        # 1. Input
        self._collect_input()

        # 2. Clock
        self._clock.update(real_dt)

        # 3. Push input state to FlyController components
        #    FlyController exposes set_input_state(InputState); App calls it here.
        self._push_input_to_controllers()

        # 4. Registry (awake / start / update / fixed / late)
        ComponentRegistry.run_frame(self._clock)

        # 5. integration hook: chunk streaming (Phase 3)
        #    Stream chunks around the camera, then drain the manager's
        #    pending_meshes / unloaded_this_frame into the scene graph.
        self._stream_and_upload_terrain()

        # 6. integration hook: lighting (Phase 4)
        #    CPU backend: no-op — sunlight is event-driven (SunlightComputer
        #    recomputes columns on bus events; remesh bakes vertex colours).
        #    GPU backend: drive the volumetric pipeline — cascade windows
        #    follow the camera, dirty volumes re-upload, compute passes
        #    (inject / propagate / fog) dispatch, terrain uniforms refresh.
        if self.lighting_pipeline is not None:
            sky_state = (self.sky_system.state
                         if self.sky_system is not None else None)
            self.lighting_pipeline.update(
                self.camera_go.transform.position, sky_state, real_dt)
            self.lighting_pipeline.update_surface_inputs(
                self.terrain_root, sky_state)

        # 7. EventBus drain
        self._event_bus.drain()

        # 8. Camera sync (Transform → Panda3D NodePath)
        self.camera_comp.sync_to_panda()

        return task.cont

    def _push_input_to_controllers(self) -> None:
        """
        Forward the current InputState to all FlyController components.

        FlyController.set_input_state(inp) is called here — the controller
        stays panda3d-free and reads the state on its next update().
        """
        # Import lazily to avoid circular imports
        try:
            from torn_apart.player.fly_controller import FlyController
        except ImportError:
            return

        from torn_apart.world.registry import _STATE
        bucket = _STATE.buckets.get(FlyController, [])
        for ctrl in bucket:
            ctrl.set_input_state(self.input_state)

    # ------------------------------------------------------------------
    # Terrain rendering (Phase 3 integration)
    # ------------------------------------------------------------------

    def setup_terrain_rendering(
        self, ground_texture=None, material_textures=None
    ) -> None:
        """
        Configure the terrain render state once at boot.

        Call after injecting ``self.chunk_manager``.  Stores the per-material
        texture map (used by the mesh upload path to texture grass and dirt
        faces separately), applies the optional fallback ground texture to
        ``terrain_root``, and turns Panda3D lighting OFF so the default
        fixed-function pipeline renders **texture × vertex colour**.  The
        mesher has already baked sunlight into the vertex colours (greyscale ×
        light), so adding a Panda3D light would double-light the scene.

        Parameters
        ----------
        ground_texture : panda3d.core.Texture | None
            Fallback texture applied at the ``terrain_root`` node level.  It
            covers blocky-mesher geometry (no ``face_materials``) and any
            material id missing from ``material_textures``.  If None, no
            node-level texture is set.
        material_textures : dict[int, panda3d.core.Texture] | None
            Material id → texture map for the faceted mesher's per-material
            Geom split (``{MATERIAL_DIRT: dirt_tex, MATERIAL_GRASS:
            grass_tex}``).  Forwarded to ``geometry_bridge.to_geom_node`` on
            every chunk upload.  Geom-level texture states compose over the
            node-level fallback, so they win where both exist.

        Notes
        -----
        - ``set_light_off()`` ensures no ambient/directional light multiplies the
          vertex colours; the baked-light look is preserved exactly.
        - The geom vertex format (geometry_bridge.make_vertex_format) includes a
          C4 colour column, so vertex colours are active by default — no
          ``set_color_off`` is issued.
        """
        self.material_textures = material_textures
        if ground_texture is not None:
            self.terrain_root.set_texture(ground_texture)
        # Baked light lives in vertex colours — disable scene lighting so the
        # pipeline renders texture × vertex-colour (no extra light term).
        self.terrain_root.set_light_off()

    def _stream_and_upload_terrain(self) -> None:
        """
        Drive chunk streaming and sync produced meshes to the scene graph.

        Per frame (when a ``chunk_manager`` is injected):
          1. ``stream_frame(camera_pos, light_sampler)`` — loads/meshes ≤2 chunks
             near the camera and remeshes dirty (edited/relit) chunks, populating
             ``pending_meshes`` and ``unloaded_this_frame``.
          2. Drain ``pending_meshes``: convert each ``MeshArrays`` to a GeomNode
             (bulk-write Geom) and parent it under ``terrain_root``.  Mesh
             positions are absolute world meters, so the NodePath is placed at the
             origin (no offset).  Any existing NodePath for that coord is detached
             first (remesh replaces stale geometry).
          3. Drain ``unloaded_this_frame``: detach + forget those coords' Geoms.

        All scene-graph writes are bulk Geom uploads (Hard Rule 7); no per-vertex
        Python loops (those live in the headless mesher / geometry_bridge).
        """
        cm = self.chunk_manager
        if cm is None:
            return

        # Lazy import: terrain → world is an allowed downward dependency, but we
        # import here to keep the module importable when panda3d-only tooling
        # constructs a bare App.
        from torn_apart.world.geometry_bridge import to_geom_node

        # 1. Stream around the camera (light_sampler may be None → full-bright).
        cm.stream_frame(self.camera_go.transform.position, self.light_sampler)

        # 2. Upload freshly produced meshes.  Copy keys first: we mutate the dict.
        for coord in list(cm.pending_meshes.keys()):
            mesh = cm.pending_meshes.pop(coord)
            # Replace any stale NodePath for this coord (remesh after a brush edit).
            old = self._chunk_nodes.pop(coord, None)
            if old is not None:
                old.remove_node()
            geom_node = to_geom_node(
                mesh,
                name=f"chunk_{coord[0]}_{coord[1]}_{coord[2]}",
                material_textures=getattr(self, "material_textures", None),
            )
            np_node = self.terrain_root.attach_new_node(geom_node)
            # Positions are absolute world meters — no per-chunk offset.
            self._chunk_nodes[coord] = np_node

        # 3. Remove Geoms for chunks unloaded this frame.
        for coord in cm.unloaded_this_frame:
            node = self._chunk_nodes.pop(coord, None)
            if node is not None:
                node.remove_node()
