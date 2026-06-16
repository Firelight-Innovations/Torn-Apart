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
    from fire_engine.core import load_config, Clock, EventBus
    from fire_engine.render.app import App

    cfg   = load_config()
    bus   = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    app   = App(cfg, clock, bus)
    app.run()

Docs: docs/systems/render.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AntialiasAttrib,
    ClockObject,
    NodePath,
    WindowProperties,
)

from fire_engine.core.profiler import init_profiler
from fire_engine.render._impl.app_input import (
    collect_input,
    push_input_to_controllers,
    set_mouse_capture,
)
from fire_engine.render._impl.app_input import (
    window_event as _window_event_impl,
)
from fire_engine.render._impl.app_profiler import maybe_write_snapshot, setup_profiler
from fire_engine.render._impl.app_terrain import (
    setup_terrain_rendering as _setup_terrain_impl,
)
from fire_engine.render._impl.app_terrain import (
    stream_and_upload_terrain,
)
from fire_engine.render.camera import CameraComponent
from fire_engine.render.registry import ComponentRegistry, instantiate
from fire_engine.render.types import InputState  # re-export for backward compatibility

if TYPE_CHECKING:
    from fire_engine.core.clock import Clock
    from fire_engine.core.config import Config
    from fire_engine.core.event_bus import EventBus
    from fire_engine.render.bridges.profiler_bridge import PStatsBridge
    from fire_engine.render.bridges.profiler_overlay import ProfilerOverlay

# Re-export InputState so existing ``from fire_engine.render.app import InputState`` works.
__all__ = ["App", "InputState"]


# App


class App(ShowBase):  # type: ignore[misc]  # panda3d ShowBase has no stubs; Any base is expected
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

    Docs: docs/systems/render.md
    """

    # Class-level attribute annotations so mypy resolves attributes set by _impl helpers.
    _config: Any
    _clock: Any
    _event_bus: Any
    _profiler: Any
    _profiler_overlay: ProfilerOverlay | None
    _profiler_bridge: PStatsBridge | None
    _snapshot_path: str | None
    _snapshot_interval_s: float
    _last_snapshot_t: float
    input_state: InputState
    _escape_was_down: bool
    _skip_mouse_delta: bool
    _had_focus: bool
    chunk_manager: Any
    light_sampler: Any
    lighting_pipeline: Any
    sky_system: Any
    post_process: Any
    _chunk_nodes: dict[tuple[int, int, int], NodePath]
    _key_state: dict[str, bool]
    terrain_root: NodePath
    material_textures: Any
    camera_go: Any
    camera_comp: CameraComponent

    def __init__(
        self,
        config: Config,
        clock: Clock,
        event_bus: EventBus,
    ) -> None:
        # The HDR post-processing buffers are full-window render targets.
        # Panda3D's default ``textures-power-2 down`` would round them to a
        # power-of-two (e.g. 1280×720 → 2048×1024) and the composite would then
        # sample the scene in a sub-rectangle of a padded texture (the rest
        # black).  This GPU supports NPOT textures (the lighting cascades are
        # already non-power-of-two 3-D volumes), so disable padding when
        # post-processing is on.  Must run before the GSG is created.
        if getattr(config, "gfx_post_process", True):
            from panda3d.core import loadPrcFileData

            loadPrcFileData("torn-apart-hdr", "textures-power-2 none")
        # MSAA must be requested BEFORE the window is opened (framebuffer
        # property).  It anti-aliases GEOMETRY edges only — facet silhouettes,
        # crater rims, the horizon line, which otherwise twinkle in motion —
        # while every surface interior is still shaded once per pixel, so the
        # crisp pixel-art texel look is untouched.
        if int(getattr(config, "msaa_samples", 0)) > 0:
            loadPrcFileData(
                "torn-apart-msaa",
                f"framebuffer-multisample 1\nmultisamples {int(config.msaa_samples)}",
            )
        super().__init__()

        self._config = config
        self._clock = clock
        self._event_bus = event_bus

        # Performance profiler (core, panda3d-free) — configure the process
        # singleton from config.  No-op + zero buffers when disabled.  The
        # overlay / PStats bridge are constructed only when enabled (see
        # _setup_profiler, called after the window + camera exist).
        self._profiler = init_profiler(config)
        self._profiler_overlay = None  # F3 in-game overlay
        self._profiler_bridge = None
        self._snapshot_path = None
        self._snapshot_interval_s = float(getattr(config, "profiler_snapshot_interval_s", 1.0))
        self._last_snapshot_t = 0.0

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

        # Terrain-render injection slots (set by main.py after construction).
        # Left None so the engine shell can run without terrain (tooling).
        self.chunk_manager = None  # ChunkManager | None
        self.light_sampler = None  # Callable | None
        # GPU volumetric lighting (Phase 4 GPU backend) — set by main.py when
        # config.lighting_backend == "gpu".  Driven in step 6 of the frame
        # task; None keeps the legacy baked-vertex-light path.
        self.lighting_pipeline = None  # GpuLightingPipeline | None
        self.sky_system = None  # SkySystem | None (set by main.py)
        # HDR post-processing pipeline (set by main.py after sky/grass exist;
        # None keeps the legacy in-shader-tonemap path).  Driven in step 6b of
        # the frame task.
        self.post_process = None  # PostProcessPipeline | None
        # Per-chunk NodePath bookkeeping: coord -> NodePath under terrain_root.
        self._chunk_nodes: dict[tuple[int, int, int], NodePath] = {}

        # Window setup
        props = WindowProperties()
        props.set_size(1280, 720)
        props.set_title("Torn Apart")
        props.set_fixed_size(False)
        self.win.request_properties(props)

        if config.show_fps:
            self.setFrameRateMeter(True)

        # Vsync — Panda3D uses sync-video property
        self.set_sleep(0.0)  # let OS vsync do its job

        # Geometry-edge AA only: with msaa_samples > 0, multisample the
        # triangle edges (interiors are single-sample — retro texels intact).
        # msaa_samples = 0 restores the fully unfiltered retro rasterisation.
        if int(getattr(config, "msaa_samples", 0)) > 0:
            self.render.set_antialias(AntialiasAttrib.M_multisample)
        else:
            self.render.set_antialias(AntialiasAttrib.M_none)

        # Default the HDR-output flag OFF so every surface shader tonemaps
        # internally (legacy look) unless the post-process pipeline turns it on.
        # Set on ``render`` so all surface shaders inherit one source of truth.
        self.render.set_shader_input("u_hdr_output", 0.0)

        # Disable Panda3D's default camera controller
        self.disableMouse()

        # Capture the mouse immediately so free-look works the moment the window
        # opens (no need to hunt for ESC first).  ESC toggles capture off again
        # to free the cursor.  The actual cursor lock is requested in
        # _set_mouse_capture; the first delta is skipped via _skip_mouse_delta.
        self.input_state.mouse_captured = True
        self._set_mouse_capture(True)

        # Camera GameObject
        self.camera_go = instantiate()
        self.camera_go.name = "MainCamera"
        from fire_engine.core.math3d import Vec3

        self.camera_go.transform.local_position = Vec3(0.0, -20.0, 10.0)
        self.camera_comp = self.camera_go.add_component(CameraComponent, base=self)

        # Terrain root NodePath
        # Every streamed chunk's GeomNode is parented under this single node.
        # Chunk mesh positions are ABSOLUTE WORLD METERS (see meshing.py:
        # MeshArrays.positions doc — "vertex positions in world meters"), so
        # terrain_root stays at the origin and each chunk NodePath is added with
        # NO per-chunk offset.  Offsetting here would double the world position.
        self.terrain_root = self.render.attach_new_node("terrain_root")

        # Key bindings (Panda3D event strings)
        self._key_state = {}
        for key in [
            "w",
            "s",
            "a",
            "d",
            "space",
            "lcontrol",
            "rcontrol",
            "lshift",
            "rshift",
            "e",
            "q",
        ]:
            self.accept(key, self._key_down, [key])
            self.accept(key + "-up", self._key_up, [key])
        self.accept("escape", self._on_escape)

        # Profiler render-side wiring (overlay + PStats bridge) — only when on.
        self._setup_profiler()

        # Register the per-frame task
        self.taskMgr.add(self._frame_task, "TornApartFrame")

    # Profiler setup (boot wiring) — delegated to _impl

    def _setup_profiler(self) -> None:
        """Wire the render-side profiler pieces when ``profiler_enabled``."""
        setup_profiler(self)

    # Input handlers

    def _key_down(self, key: str) -> None:
        self._key_state[key] = True

    def _key_up(self, key: str) -> None:
        self._key_state[key] = False

    def _on_escape(self) -> None:
        """Toggle mouse capture on ESC."""
        self.input_state.escape_pressed = True

    def _collect_input(self) -> None:
        """Read current Panda3D input state and write to self.input_state."""
        collect_input(self)

    def windowEvent(self, win: Any) -> None:
        """
        Handle Panda3D window events (focus, resize, close).

        Extends ShowBase's default handling to fix a mouse-capture desync: when
        the window loses OS focus (alt-tab), the platform releases our hidden /
        relative-mouse cursor properties.  Panda3D does not re-apply them on
        focus regain, so the engine would think the mouse is captured while the
        OS shows a free, absolute-mode cursor.

        Parameters
        ----------
        win : panda3d.core.GraphicsWindow
            The window the event is about (ignored unless it is ``self.win``).

        Docs: docs/systems/render.md
        """
        super().windowEvent(win)
        _window_event_impl(self, win)

    def _set_mouse_capture(self, captured: bool) -> None:
        """
        Lock/unlock the cursor for free-look.

        Captured → cursor hidden + relative mouse mode.
        Released → cursor shown + absolute mode.
        """
        set_mouse_capture(self, captured)

    # Frame task

    def _frame_task(self, task: Any) -> Any:
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
        prof = self._profiler
        # begin_frame finalizes the PREVIOUS frame (its full wall duration —
        # incl. the render/flip that happened after last frame's end_frame — is
        # now known) and resets the per-frame accumulators.  No-op when off.
        prof.begin_frame()

        real_dt = float(ClockObject.get_global_clock().get_dt())  # Panda3D's frame time

        # 1. Input
        with prof.scope("Input"):
            self._collect_input()

        # 2. Clock
        with prof.scope("Clock"):
            self._clock.update(real_dt)

        # 3. Push input state to FlyController components
        #    FlyController exposes set_input_state(InputState); App calls it here.
        with prof.scope("Input"):
            self._push_input_to_controllers()

        # 4. Registry (awake / start / update / fixed / late).  The registry
        #    adds child scopes per component type ("Update:<Type>").
        with prof.scope("Update"):
            ComponentRegistry.run_frame(self._clock)

        # 5. integration hook: chunk streaming (Phase 3)
        #    Stream chunks around the camera, then drain the manager's
        #    pending_meshes / unloaded_this_frame into the scene graph.
        with prof.scope("ChunkStream"):
            self._stream_and_upload_terrain()

        # 6. integration hook: lighting (Phase 4)
        #    CPU backend: no-op — sunlight is event-driven (SunlightComputer
        #    recomputes columns on bus events; remesh bakes vertex colours).
        #    GPU backend: drive the volumetric pipeline — cascade windows
        #    follow the camera, dirty volumes re-upload, compute passes
        #    (inject / gather / fog) dispatch, and the lit-surface uniforms
        #    refresh on ``render`` (inherited by every lit shader).
        if self.lighting_pipeline is not None:
            with prof.scope("Lighting"):
                sky_state = self.sky_system.state if self.sky_system is not None else None
                self.lighting_pipeline.update(self.camera_go.transform.position, sky_state, real_dt)
                self.lighting_pipeline.update_surface_inputs(self.render, sky_state)

        # 6b. integration hook: HDR post-processing (Phase 2+)
        #     Refresh per-frame post inputs (bloom strength, lens-flare sun
        #     position, …).  The scene already rendered into the HDR buffer; the
        #     composite + effect passes run as render2d cards after this task.
        if self.post_process is not None:
            with prof.scope("PostProcess"):
                self.post_process.update(self.lighting_pipeline)

        # 7. EventBus drain
        with prof.scope("EventDrain"):
            self._event_bus.drain()

        # 8. Camera sync (Transform → Panda3D NodePath)
        with prof.scope("CameraSync"):
            self.camera_comp.sync_to_panda()

        # end_frame closes the loop body (records the CPU-frame time); the full
        # frame_ms is finalized at the next begin_frame.  Then refresh the
        # low-Hz overlay and write the rolling JSON snapshot if it's due.
        prof.end_frame()
        if self._profiler_overlay is not None:
            self._profiler_overlay.update()
        self._maybe_write_snapshot()

        return task.cont

    def _maybe_write_snapshot(self) -> None:
        """Write the rolling profiler JSON snapshot if the interval elapsed."""
        maybe_write_snapshot(self)

    def _push_input_to_controllers(self) -> None:
        """Forward the current InputState to all FlyController components."""
        push_input_to_controllers(self)

    # Terrain rendering (Phase 3 integration) — delegated to _impl

    def setup_terrain_rendering(
        self, ground_texture: Any = None, material_textures: Any = None
    ) -> None:
        """
        Configure the terrain render state once at boot.

        Call after injecting ``self.chunk_manager``.  See
        render/_impl/app_terrain.py for the full docstring.

        Docs: docs/systems/render.md
        """
        _setup_terrain_impl(self, ground_texture, material_textures)

    def _stream_and_upload_terrain(self) -> None:
        """Drive chunk streaming and sync produced meshes to the scene graph."""
        stream_and_upload_terrain(self)
