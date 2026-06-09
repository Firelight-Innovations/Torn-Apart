"""
main.py — Torn Apart vertical-slice demo entry point.

Run with ``python main.py`` from the repo root.  This orchestrates the boot
sequence mandated by ARCHITECTURE.md §4a.1 and wires the demo loop described in
DEVELOPMENT_PLAN.md (Mission):

    Fly a free camera (WASD + mouse) over voxel terrain textured with the
    ``wasteland_ground`` procedural texture and lit by baked sunlight (dark
    under overhangs).  Chunks stream in around the camera.

    LEFT-CLICK fires an explosion: a camera ray is cast into the voxel field;
    on a hit a SphereBrush(REMOVE) carves a crater.  The brush marks touched
    chunks dirty + edited and publishes TerrainEditedEvent; the SunlightComputer
    (subscribed to the bus) recomputes the affected light column and re-marks
    those chunks dirty; the next stream_frame remeshes them with fresh light, so
    the crater AND its relight appear within a frame or two.

    F5 saves to saves/quick.ta (delta = edited chunks only).
    F9 reverts to the saved state: reset_to_baseline() wipes ALL edits back to
    the procedural baseline, then load() re-applies only the saved craters — so
    craters made AFTER the save are correctly undone on load.

Boot order (ARCHITECTURE §4a.1) — see inline numbered comments below.

This file holds DEV/demo key bindings (§5.5: brush bindings are dev tooling, not
engine), so they live here in main.py, not inside the engine.  App is a Panda3D
ShowBase, so ``app.accept(event, fn)`` registers the bindings.

``import main`` must NOT open a window — the boot + run is guarded behind
``if __name__ == "__main__":``.
"""

from __future__ import annotations

from pathlib import Path

# --- Foundation layer (panda3d-free) -------------------------------------
from torn_apart.core import (
    Clock,
    EventBus,
    load_config,
    get_logger,
)
from torn_apart.core.rng import set_world_seed
from torn_apart.core.math3d import Vec3

# --- Procedural content: importing the package auto-registers the
#     "wasteland_ground" texture def (and any other built-ins). --------------
import torn_apart.procedural  # noqa: F401  (import-for-side-effect: registration)
from torn_apart.procedural import get as get_procedural

# --- Terrain (Layer 2) ----------------------------------------------------
from torn_apart.terrain import (
    ChunkManager,
    SphereBrush,
    BrushMode,
    apply_brush,
    raycast_voxel,
)

# --- Lighting (Layer 2) ---------------------------------------------------
from torn_apart.lighting import (
    LightGrid,
    SunlightComputer,
    make_light_sampler,
)

# --- Player (thin control layer) ------------------------------------------
from torn_apart.player import FlyController

# --- Save subsystem -------------------------------------------------------
from torn_apart.save import SaveManager, SaveIncompatibleError

_log = get_logger("main")

# Demo constants (not gameplay magic numbers — these are dev-binding tuning).
_SAVE_PATH = "saves/quick.ta"
_EXPLOSION_RADIUS_M = 2.5      # SphereBrush radius for left-click explosions
_RAY_MAX_DISTANCE_M = 100.0    # how far the click ray probes for terrain
_PREWARM_STREAM_FRAMES = 80    # stream_frame iterations to pre-load spawn area


def _prewarm_terrain(app, chunk_manager, sunlight, light_sampler) -> None:
    """
    Pre-load chunks around spawn so the first rendered frame is not empty.

    ``stream_frame`` loads at most 2 chunks per call (the per-frame budget), so
    we call it many times here to fill the spawn area before the window appears.
    After the chunks exist we seed sunlight for every loaded column, then upload
    every pending mesh once so spawn geometry is on the GPU before frame 1.

    Parameters
    ----------
    app : App
        The application (owns terrain_root + the upload path).
    chunk_manager : ChunkManager
    sunlight : SunlightComputer
    light_sampler : Callable
    """
    spawn = app.camera_go.transform.position
    for _ in range(_PREWARM_STREAM_FRAMES):
        chunk_manager.stream_frame(spawn, light_sampler)
    # Seed sunlight for all loaded columns (events only covered incremental work).
    sunlight.recompute_all_loaded()
    # One more streaming pass so any chunk marked dirty by the light seed is
    # remeshed with baked light, then upload everything via the App's drain path.
    chunk_manager.stream_frame(spawn, light_sampler)
    app._stream_and_upload_terrain()


def _load_proof_model(app) -> None:
    """
    Load the test triangle fixture as proof the Resource Manager pipeline works.

    Wrapped in try/except so a missing/unloadable fixture never crashes the demo.
    The model is parented under render near the origin, slightly above ground.
    """
    try:
        from torn_apart.resources import default_manager, acquire
        fixture = Path(__file__).resolve().parent / "tests" / "fixtures" / "triangle.egg"
        handle = acquire(default_manager.load(str(fixture)))
        nodepath = handle.resource
        nodepath.reparent_to(app.render)
        nodepath.set_pos(0.0, 0.0, 12.0)   # near spawn, small + visible
        nodepath.set_scale(2.0)
        _log.info("Loaded proof model tests/fixtures/triangle.egg")
    except Exception as exc:  # noqa: BLE001  (demo proof; never fatal)
        _log.warning("Proof model load skipped: %s", exc)


def build_demo():
    """
    Boot the engine and wire the demo, returning the constructed ``App``.

    Does everything EXCEPT call ``app.run()`` — so callers can either run the
    blocking main loop (``main()``) or step the task manager headlessly for an
    offscreen screenshot (``tools/screenshot.py``).

    Follows ARCHITECTURE §4a.1 startup order exactly.  Constructs the App
    (window) before registering panda3d resource loaders (they need the global
    Panda3D loader that ShowBase creates) and before building the SunlightComputer
    (which needs the ChunkManager as its chunk provider — so ChunkManager is
    constructed before SunlightComputer, re-ordering the doc's steps 6/7).

    Returns
    -------
    App
        The fully wired application, ready for ``app.run()``.
    """
    # 1. Config + global seed + logging.
    cfg = load_config()
    set_world_seed(cfg.world_seed)
    _log.info("Booting Torn Apart (seed=%d)", cfg.world_seed)

    # 2. Event bus + clock.
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)

    # 3. Procedural content is already registered (import side-effect above).

    # 5. App — creates the window + camera_go (at (0,-20,10)) + CameraComponent.
    #    (Constructed before step 4 so the Panda3D global loader exists.)
    from torn_apart.world.app import App  # panda3d import lives behind world/
    app = App(cfg, clock, bus)

    # 4. Resource manager loaders — register AFTER the window/loader exists.
    from torn_apart.resources import default_manager
    from torn_apart.world.resource_adapter import register_panda_loaders
    register_panda_loaders(default_manager)

    # 7. Terrain manager — built BEFORE the SunlightComputer (which needs it as
    #    its chunk provider).
    chunk_manager = ChunkManager(cfg, bus)

    # 6. Lighting — light grid + sunlight computer (subscribes to the bus in its
    #    __init__) + the mesher's light sampler.
    light_grid = LightGrid()
    sunlight = SunlightComputer(cfg, chunk_manager, light_grid, bus)
    light_sampler = make_light_sampler(light_grid, cfg)

    # 8. Save manager — register terrain first (registration order matters).
    save_manager = SaveManager(cfg, clock)
    save_manager.register(chunk_manager)

    # 9. Player — attach a FlyController to the camera GameObject.  The App
    #    forwards InputState to all FlyControllers each frame.
    app.camera_go.add_component(FlyController)

    # --- Inject terrain-render deps into the App and configure render state ---
    app.chunk_manager = chunk_manager
    app.light_sampler = light_sampler
    ground_tex = _to_ground_texture()
    app.setup_terrain_rendering(ground_tex)

    # 10. Pre-stream spawn area + seed sunlight + upload initial meshes.
    _prewarm_terrain(app, chunk_manager, sunlight, light_sampler)

    # 11. Resource-manager proof model (non-fatal).
    _load_proof_model(app)

    # --- Demo key bindings (DEV tooling per §5.5) -------------------------
    Path("saves").mkdir(parents=True, exist_ok=True)

    def on_click() -> None:
        """
        Left-click → fire an explosion at the terrain under the camera ray.

        Builds the camera ray (origin = camera position, direction = camera
        forward), raycasts the voxel field, and on a hit carves a SphereBrush
        REMOVE crater at the hit point.  apply_brush flags touched chunks dirty +
        edited and publishes TerrainEditedEvent; the SunlightComputer relights
        the column; the next stream_frame remeshes → crater + relight appear.
        """
        origin = app.camera_go.transform.position
        direction = app.camera_go.transform.forward
        hit = raycast_voxel(
            origin,
            direction,
            chunk_manager.get_or_create,
            max_distance_m=_RAY_MAX_DISTANCE_M,
        )
        if hit is None:
            _log.debug("Click: no terrain hit within %.0f m", _RAY_MAX_DISTANCE_M)
            return
        # hit.point is the world-space entry point of the solid voxel — the
        # natural explosion centre.
        touched = apply_brush(
            SphereBrush(_EXPLOSION_RADIUS_M),
            hit.point,
            BrushMode.REMOVE,
            material=1,
            chunk_provider=chunk_manager.get_or_create,
            bus=bus,
        )
        _log.info("Explosion at %s — %d chunk(s) cratered", hit.point, len(touched))

    def on_save() -> None:
        """F5 → save the world (edited chunks only) to saves/quick.ta."""
        try:
            save_manager.save(_SAVE_PATH)
            _log.info("Saved to %s", _SAVE_PATH)
        except OSError as exc:
            _log.error("Save failed: %s", exc)

    def on_load() -> None:
        """
        F9 → revert to the saved state.

        reset_to_baseline() first wipes ALL current edits back to the procedural
        baseline (undoing craters dug after the save), then load() re-applies the
        saved craters via apply_delta.  Both mark chunks dirty; the streaming loop
        remeshes them and the App re-uploads their Geoms over the next frames.
        Missing save / incompatible save is logged, never fatal.
        """
        chunk_manager.reset_to_baseline()
        try:
            save_manager.load(_SAVE_PATH)
            _log.info("Loaded %s", _SAVE_PATH)
        except FileNotFoundError:
            _log.warning("No save at %s yet (press F5 first)", _SAVE_PATH)
        except SaveIncompatibleError as exc:
            _log.error("Save incompatible: %s", exc)

    app.accept("mouse1", on_click)
    app.accept("f5", on_save)
    app.accept("f9", on_load)

    _log.info("Demo ready — WASD+mouse to fly, ESC to capture mouse, "
              "left-click to explode, F5 save, F9 load.")

    return app


def main() -> None:
    """Boot the demo and run the blocking main loop (opens a window)."""
    app = build_demo()
    # 12. Run (blocks until the window closes).
    app.run()


def _to_ground_texture():
    """
    Build the Panda3D ``wasteland_ground`` texture for the terrain.

    Pulls the procedural RGBA array from the registry and bridges it to a
    Panda3D Texture via world/texture_bridge.  Returns None (and logs) on any
    failure so the demo can still run with untextured terrain.

    Returns
    -------
    panda3d.core.Texture | None
    """
    try:
        from torn_apart.world.texture_bridge import to_panda_texture
        rgba = get_procedural("wasteland_ground")   # (256,256,4) uint8
        return to_panda_texture(rgba)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Ground texture build failed (untextured terrain): %s", exc)
        return None


if __name__ == "__main__":
    main()
