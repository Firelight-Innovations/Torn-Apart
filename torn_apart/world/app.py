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

        # ------------------------------------------------------------------
        # Camera GameObject
        # ------------------------------------------------------------------
        self.camera_go = instantiate(name="MainCamera")
        from torn_apart.core.math3d import Vec3
        self.camera_go.transform.local_position = Vec3(0.0, -20.0, 10.0)
        self.camera_comp = self.camera_go.add_component(CameraComponent, base=self)

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

        # Mouse delta (only when captured)
        inp.mouse_dx = 0.0
        inp.mouse_dy = 0.0
        if inp.mouse_captured and self.mouseWatcherNode.hasMouse():
            mx = self.mouseWatcherNode.getMouseX()
            my = self.mouseWatcherNode.getMouseY()
            # Convert from Panda3D's [-1,1] normalised range to pixels
            # (approximate: multiply by half window size)
            win_w = self.win.get_x_size()
            win_h = self.win.get_y_size()
            inp.mouse_dx = mx * (win_w * 0.5)
            inp.mouse_dy = my * (win_h * 0.5)
            # Recentre cursor
            self.win.move_pointer(0, win_w // 2, win_h // 2)

    def _set_mouse_capture(self, captured: bool) -> None:
        """Lock/unlock the cursor."""
        props = WindowProperties()
        props.set_cursor_hidden(captured)
        props.set_mouse_mode(
            WindowProperties.M_confined if captured else WindowProperties.M_absolute
        )
        self.win.request_properties(props)

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

        # 5. integration hook: chunk streaming
        #    ChunkManager.stream_frame(camera_position) will go here in Phase 3.
        #    Leave as a clearly-named placeholder so the orchestrator can fill it.
        # --- integration hook: chunk streaming (Phase 3) ---
        # if hasattr(self, '_chunk_manager'):
        #     self._chunk_manager.stream_frame(self.camera_go.transform.position)

        # 6. integration hook: lighting dirty work
        #    LightGrid.update_dirty() will go here in Phase 4.
        # --- integration hook: lighting dirty work (Phase 4) ---
        # if hasattr(self, '_light_grid'):
        #     self._light_grid.update_dirty()

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
