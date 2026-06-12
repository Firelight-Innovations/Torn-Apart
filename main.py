"""
main.py — Torn Apart vertical-slice demo entry point.

Run with ``python main.py`` from the repo root.  This orchestrates the boot
sequence mandated by ARCHITECTURE.md §4a.1 and wires the demo loop described in
DEVELOPMENT_PLAN.md (Mission):

    Fly a free camera (WASD + mouse) over faceted voxel terrain (flat-shaded
    surface nets) textured with the pixel-art ``grass_ground`` /
    ``dirt_ground`` procedural textures (``wasteland_ground`` as fallback)
    and lit by baked sunlight (dark under overhangs).  Chunks stream in
    around the camera.

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

import math
import sys
from pathlib import Path

# --- Foundation layer (panda3d-free) -------------------------------------
from fire_engine.core import (
    Clock,
    EventBus,
    load_config,
    get_logger,
)
from fire_engine.core.rng import set_world_seed
from fire_engine.core.math3d import Vec3

# --- Procedural content: importing the package auto-registers the
#     "wasteland_ground" texture def (and any other built-ins). --------------
import fire_engine.procedural  # noqa: F401  (import-for-side-effect: registration)
from fire_engine.procedural import get as get_procedural

# --- Terrain (Layer 2) ----------------------------------------------------
from fire_engine.terrain import (
    ChunkManager,
    SphereBrush,
    BrushMode,
    apply_brush,
    raycast_voxel,
)

# --- Lighting (Layer 2) ---------------------------------------------------
from fire_engine.lighting import (
    LightGrid,
    SunlightComputer,
    make_light_sampler,
)

# --- Player (thin control layer) ------------------------------------------
from fire_engine.player import FlyController

# --- Save subsystem -------------------------------------------------------
from fire_engine.save import SaveManager, SaveIncompatibleError

_log = get_logger("main")

# Demo constants (not gameplay magic numbers — these are dev-binding tuning).
_SAVE_PATH = "saves/quick.ta"
_EXPLOSION_RADIUS_M = 2.5      # SphereBrush radius for left-click explosions
_RAY_MAX_DISTANCE_M = 100.0    # how far the click ray probes for terrain
_PREWARM_STREAM_FRAMES = 80    # stream_frame iterations to pre-load spawn area
_BOOT_TIME_OF_DAY_H = 10.0     # boot the demo mid-morning (game clock starts at
                               # 00:00 otherwise and the sky dims terrain to night)

# GI test-room materials (debug ids far above the terrain materials).  Albedo
# rows are patched into the lighting palette; flat-colour texture triples are
# added in _to_material_textures so the surfaces render in matching colours.
_MAT_GI_WHITE = 200
_MAT_GI_RED = 201
_MAT_GI_GREEN = 202
_MAT_GI_GLOW = 203
_GI_TEST_ALBEDO: dict[int, tuple[float, float, float]] = {
    _MAT_GI_WHITE: (0.86, 0.86, 0.86),
    _MAT_GI_RED:   (0.78, 0.06, 0.05),
    _MAT_GI_GREEN: (0.07, 0.66, 0.08),
    _MAT_GI_GLOW:  (0.90, 0.88, 0.84),
}
_GI_GLOW_RADIANCE = (4.0, 3.6, 2.8)   # ceiling-panel emission (linear HDR;
                                      # EMISSION_SCALE=8 is the storage cap).
                                      # Lowered from (8,7.2,5.6): a panel that
                                      # bright in a closed white box blows the
                                      # auto-exposed interior to flat gray and
                                      # hides the red/green wall bleed.  This
                                      # value still strongly lights the coloured
                                      # walls (saturated bounce) without blowout.
# AreaLight co-located with the panel (fills the room; meter-visible).
_GI_PANEL_COLOR = (1.0, 0.95, 0.85)   # warm white
_GI_PANEL_INTENSITY = 2.0             # HDR; a low white direct fill so the
                                      # red/green wall inter-reflection isn't
                                      # swamped by white light (was 6.0 → the
                                      # closed white box blew out flat gray).

def _gi_ground_lut_entries():
    """
    Flat ``material id → (palette, thresholds)`` LUT rows for the GI test-room
    surfaces, so the ground-palette LUT colours materials 200–203 with their
    solid albedo instead of clamping to the last (grass) row.

    The terrain shader gamma-decodes the LUT (``pow(alb, 2.2)``), so the
    palette stores the sRGB-encoded form of the linear ``_GI_TEST_ALBEDO``
    colours.  A single-colour palette uses zero thresholds (every noise bucket
    maps to the one colour → a flat, non-noisy surface).

    Returns
    -------
    dict[int, tuple[numpy.ndarray, numpy.ndarray]]
    """
    import numpy as np
    entries = {}
    for mid, rgb in _GI_TEST_ALBEDO.items():
        srgb = (np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
                ** (1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)
        entries[mid] = (srgb.reshape(1, 3), np.zeros((0,), np.float32))
    return entries


# Flashlight (F key) tuning.
_FLASHLIGHT_COLOR = (1.0, 0.96, 0.86)
_FLASHLIGHT_INTENSITY = 20.0
_FLASHLIGHT_RADIUS_M = 36.0
_FLASHLIGHT_CONE_DEG = 38.0


def _ensure_dedicated_gpu() -> None:
    """
    Register this python.exe for the high-performance GPU (Windows).

    Laptops with hybrid graphics default python.exe to the integrated GPU.
    Writing ``GpuPreference=2`` ("high performance") under
    ``HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences`` makes Windows
    hand the process the dedicated GPU instead — the same switch as
    Settings → Display → Graphics, no admin rights needed.

    CRITICAL: Windows matches the preference by the PROCESS IMAGE path.  A
    venv's ``python.exe`` is a launcher that spawns the BASE interpreter, so
    ``sys.executable`` is NOT the running image — register the real image
    (``GetModuleFileNameW``) plus ``sys.executable``/``sys._base_executable``
    for good measure.  The preference is read at the process's first GPU
    context creation, so a fresh write applies from the NEXT launch (check
    the "Rendering on:" log line).

    No-op (logged, never fatal) on non-Windows or if the registry write fails.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import winreg
        targets = {sys.executable}
        base = getattr(sys, "_base_executable", None)
        if base:
            targets.add(base)
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetModuleFileNameW(None, buf, 1024):
            targets.add(buf.value)            # the actual process image
        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        value = "GpuPreference=2;"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            for exe in sorted(targets):
                try:
                    current, _ = winreg.QueryValueEx(key, exe)
                except FileNotFoundError:
                    current = None
                if current != value:
                    winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, value)
                    _log.info(
                        "Registered %s for the high-performance GPU "
                        "(applies on next launch)", exe)
    except OSError as exc:
        _log.warning("GPU preference registry write failed: %s", exc)


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
    sunlight : SunlightComputer | None
        ``None`` on the GPU lighting backend (no CPU column pass to seed —
        the volumetric pipeline lights everything on the GPU).
    light_sampler : Callable | None
    """
    spawn = app.camera_go.transform.position
    for _ in range(_PREWARM_STREAM_FRAMES):
        chunk_manager.stream_frame(spawn, light_sampler)
    if sunlight is not None:
        # Seed sunlight for all loaded columns (events only covered
        # incremental work).
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
        from fire_engine.resources import default_manager, acquire
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
    # 0. Hybrid-graphics laptops: claim the dedicated GPU (next launch).
    _ensure_dedicated_gpu()

    # 1. Config + global seed + logging.
    cfg = load_config()
    set_world_seed(cfg.world_seed)
    _log.info("Booting Torn Apart (seed=%d)", cfg.world_seed)

    # 2. Event bus + clock.  Start mid-morning so the demo opens in daylight.
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    clock.game_time_of_day = _BOOT_TIME_OF_DAY_H * 3600.0

    # 3. Procedural content is already registered (import side-effect above).

    # 5. App — creates the window + camera_go (at (0,-20,10)) + CameraComponent.
    #    (Constructed before step 4 so the Panda3D global loader exists.)
    from fire_engine.world.app import App  # panda3d import lives behind world/
    app = App(cfg, clock, bus)

    # Which GPU did Windows actually give us?  On hybrid laptops an "Intel"
    # renderer here means the integrated GPU — _ensure_dedicated_gpu() has
    # registered the fix; it applies on the next launch.
    try:
        gsg = app.win.get_gsg()
        renderer = gsg.get_driver_renderer()
        _log.info("Rendering on: %s (%s)", renderer, gsg.get_driver_vendor())
        if "intel" in renderer.lower():
            _log.warning(
                "Integrated GPU in use — restart the game to pick up the "
                "high-performance GPU preference written this boot.")
    except Exception as exc:  # noqa: BLE001  (diagnostics; never fatal)
        _log.debug("Renderer query failed: %s", exc)

    # 4. Resource manager loaders — register AFTER the window/loader exists.
    from fire_engine.resources import default_manager
    from fire_engine.world.resource_adapter import register_panda_loaders
    register_panda_loaders(default_manager)

    # 7. Terrain manager — built BEFORE the SunlightComputer (which needs it as
    #    its chunk provider).
    chunk_manager = ChunkManager(cfg, bus)

    # 7b. Zone volumes — the demo world's default grass region: a box right in
    #     front of spawn (camera at (0,-20,10) looking +Y; ground top z=8).
    #     mark_baseline() makes these defaults the save baseline, so untouched
    #     worlds keep a ~0-byte "zones" delta.
    from fire_engine.zones import ZoneStore
    zone_store = ZoneStore()
    zone_store.add(
        "grass",
        (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0),
        params={"density": cfg.grass_density_per_m2},
    )
    # Demo "trees" volume — 3-D instanced trees (world/tree_renderer.py) AND
    # the seam the wind system's leaf litter renders on (one volume: plant a
    # forest, get its leaf fall for free).  Placed just to the +X side of the
    # grass box, ~20×20 m footprint, z 6→14 to straddle the terrain surface
    # (ground top z=8) and give leaves vertical room to stream.  species_mix
    # picks from the registered TreeSpeciesDefs (procedural/flora/species/).
    zone_store.add(
        "trees",
        (14.0, -5.0, 6.0), (34.0, 15.0, 14.0),
        params={"species_mix": "tree_gnarled_oak:3,tree_dead:1"},
    )
    # Flora volumes — wildflower sprites (world/flora_renderer.py) scattered
    # through the demo grass box, and a wider band of 3-D bushes
    # (world/tree_renderer.py again — a bush is a tree with a stub trunk)
    # running from the meadow into the treeline.
    zone_store.add(
        "flowers",
        (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0),
    )
    zone_store.add(
        "bushes",
        (-12.0, -5.0, 6.0), (34.0, 25.0, 11.0),
        params={"species_mix": "bush_scrub:2,bush_berry:1"},
    )
    zone_store.mark_baseline()

    # 6. Lighting.  Two backends (config.lighting_backend):
    #    "gpu" — volumetric radiance cascades; the mesher bakes NO light
    #            (light_sampler=None → full-bright vertex colours carrying
    #            only the facet accent) and the GpuLightingPipeline (built
    #            after terrain, below) lights every fragment on the GPU.
    #    "cpu" — legacy baked-vertex sunlight column pass.
    use_gpu_lighting = cfg.lighting_backend == "gpu"
    if use_gpu_lighting:
        sunlight = None
        light_sampler = None
    else:
        light_grid = LightGrid()
        sunlight = SunlightComputer(cfg, chunk_manager, light_grid, bus)
        light_sampler = make_light_sampler(light_grid, cfg)

    # 6b. Sky + weather (Layer 1 service, headless) — constructed after lighting;
    #     the SkyRendererComponent (added below) drives sky_system.update() once
    #     per frame from its update() and reads the SkyState in late_update().
    from fire_engine.sky import SkySystem, WeatherType
    sky_system = SkySystem(cfg, clock, bus)
    app.sky_system = sky_system   # exposed for tooling (tools/screenshot.py)

    # 8. Save manager — register terrain first (registration order matters),
    #    then the weather schedule (Saveable, save_key="weather").
    save_manager = SaveManager(cfg, clock)
    save_manager.register(chunk_manager)
    save_manager.register(sky_system.weather)
    save_manager.register(zone_store)

    # 9. Player — attach a FlyController to the camera GameObject.  The App
    #    forwards InputState to all FlyControllers each frame.
    app.camera_go.add_component(FlyController)

    # --- Inject terrain-render deps into the App and configure render state ---
    app.chunk_manager = chunk_manager
    app.light_sampler = light_sampler
    # GPU lighting: NO node-level fallback texture — the fixed-function
    # texture stage would steal a texture unit from the shader's 3-D
    # samplers (albedo arrives via the per-material stage triples instead).
    ground_tex = None if use_gpu_lighting else _to_ground_texture()
    app.setup_terrain_rendering(
        ground_tex, _to_material_textures(triples=use_gpu_lighting))

    # 9b. GPU volumetric lighting pipeline + terrain surface shader.  The
    #     palette is the default (texture-derived) one plus the GI test-room
    #     debug materials: bright white/red/green bounce surfaces and an
    #     emissive ceiling-panel material (tests the emission-map path).
    lighting_pipeline = None
    if use_gpu_lighting:
        from fire_engine.lighting.gpu import GpuLightingPipeline
        from fire_engine.lighting.palette import build_default_palette
        from fire_engine.world.terrain_shader import apply_terrain_shader
        palette = build_default_palette()
        for mid, rgb in _GI_TEST_ALBEDO.items():
            palette.albedo[mid] = rgb
        palette = palette.with_emission(_MAT_GI_GLOW, _GI_GLOW_RADIANCE)
        lighting_pipeline = GpuLightingPipeline(cfg, app, chunk_manager, bus,
                                                palette=palette)
        app.lighting_pipeline = lighting_pipeline
        # Deterministic per-world hash offset for the procedural ground pattern.
        from fire_engine.core.rng import for_domain
        ground_seed = float(for_domain("terrain", "ground").integers(0, 65536))
        apply_terrain_shader(app.terrain_root, lighting_pipeline,
                             seed=ground_seed,
                             texels_per_m=cfg.ground_texels_per_m,
                             extra_materials=_gi_ground_lut_entries())
        # Bind the lit-surface uniform contract on ``render`` so EVERY shader
        # that includes lit_surface.glsl — terrain, foliage, future
        # buildings/NPCs anywhere in the graph — inherits it.  The frame loop
        # refreshes it there (app.py step 6).
        lighting_pipeline.bind_surface_inputs(app.render)

    # 10. Pre-stream spawn area + seed sunlight + upload initial meshes.
    _prewarm_terrain(app, chunk_manager, sunlight, light_sampler)

    # 10b. Sky renderer — a GameObject with the render half of the sky system.
    #      SkyRendererComponent.update() calls sky_system.update() (registry
    #      runs update before late_update), so no App changes are needed.
    from fire_engine.world import instantiate
    from fire_engine.world.sky_renderer import SkyRendererComponent
    sky_go = instantiate()
    sky_go.name = "Sky"
    sky_go.add_component(
        SkyRendererComponent,
        base=app,
        sky_system=sky_system,
        terrain_root=app.terrain_root,
        clock=clock,
        external_lighting=use_gpu_lighting,
    )
    app.sky_go = sky_go

    # 10b-wind. Wind field — the spatially-varying, time-evolving wind that
    #      grass (and later flags/cloth/motes) samples instead of one flat
    #      scalar.  Construct the headless WindField (+ the venturi worker if
    #      WP2 has landed; it is optional — WindField(cfg, worker=None) runs an
    #      identity venturi) and seed the boot default u_wind_enabled = 0.0 on
    #      terrain_root BEFORE any component starts, so grass is valid (scalar
    #      fallback) until the WindSystemComponent's first upload flips it to 1.
    from fire_engine.wind import WindField
    try:
        from fire_engine.wind import VenturiWorker        # WP2 (may not exist yet)
        venturi_worker = VenturiWorker()
        venturi_worker.start()
    except ImportError:
        venturi_worker = None
        _log.info("Venturi worker unavailable (WP2 not landed) — wind runs "
                  "with identity venturi")
    wind_field = WindField(cfg, worker=venturi_worker)
    app.wind_worker = venturi_worker        # exposed so main() can stop it on exit
    app.terrain_root.set_shader_input("u_wind_enabled", 0.0)

    # 10c. GPU grass — instanced tufts inside every "grass" zone volume,
    #      placed entirely on the GPU (gl_InstanceID hash), lit by the same
    #      radiance cascades as the terrain, swaying with the weather.
    #      GPU lighting backend only (the component disables itself on cpu).
    from fire_engine.world.grass_renderer import GrassRendererComponent
    grass_go = instantiate()
    grass_go.name = "Grass"
    grass_go.add_component(
        GrassRendererComponent,
        base=app,
        sky_system=sky_system,
        zone_store=zone_store,
        chunk_provider=chunk_manager,
        lighting_pipeline=lighting_pipeline,
        bus=bus,
    )
    app.grass_go = grass_go

    # 10c2. GPU flora — instanced flower sprites inside "flowers" zone
    #      volumes; the grass idiom generalised (gl_InstanceID hash placement,
    #      baked height fields, cascade lighting, wind-texture sway,
    #      sprite-atlas variants).  GPU lighting backend only (disables
    #      itself on cpu).
    from fire_engine.world.flora_renderer import FloraRendererComponent
    flora_go = instantiate()
    flora_go.name = "Flora"
    flora_go.add_component(
        FloraRendererComponent,
        base=app,
        sky_system=sky_system,
        zone_store=zone_store,
        chunk_provider=chunk_manager,
        lighting_pipeline=lighting_pipeline,
        bus=bus,
    )
    app.flora_go = flora_go

    # 10c3. 3-D trees + bushes — per-species variant-mesh pools instanced
    #      over CPU-baked placements inside "trees" / "bushes" zone volumes,
    #      with billboard impostors past the mesh fade window (the ONLY
    #      billboarding trees get).  Species are authored as Python scripts —
    #      see docs/content/tree_species_authoring.md.  GPU lighting backend
    #      only (disables itself on cpu).
    from fire_engine.world.tree_renderer import TreeRendererComponent
    tree_go = instantiate()
    tree_go.name = "Trees"
    tree_go.add_component(
        TreeRendererComponent,
        base=app,
        sky_system=sky_system,
        zone_store=zone_store,
        chunk_provider=chunk_manager,
        lighting_pipeline=lighting_pipeline,
        bus=bus,
    )
    app.tree_go = tree_go

    # 10d. Wind system render component — uploads the WindField snapshot as
    #      u_wind_tex on terrain_root each frame and flips u_wind_enabled to 1
    #      after the first upload, so grass samples the travelling gust field.
    #      GPU lighting backend only (disables itself + leaves the scalar
    #      fallback on cpu); it OWNS the venturi worker and stops it on destroy.
    from fire_engine.world.wind_renderer import WindSystemComponent
    wind_go = instantiate()
    wind_go.name = "Wind"
    wind_go.add_component(
        WindSystemComponent,
        base=app,
        clock=clock,
        wind_field=wind_field,
        worker=venturi_worker,
        sky_system=sky_system,
        chunk_provider=chunk_manager,
        lighting_pipeline=lighting_pipeline,
        bus=bus,
    )
    app.wind_go = wind_go

    # 10e. Wind particles — dust motes + leaf litter, both GPU-instanced with
    #      zero CPU per-particle state, sampling the inherited u_wind_tex.  Dust
    #      drifts everywhere (camera-anchored wrapping lattice); leaf litter
    #      renders on every "trees" zone volume (settles in calm air, streams in
    #      gusts/storms).  GPU lighting backend only (they disable themselves on
    #      cpu / when no wind field — same gate as grass + the wind component).
    from fire_engine.world.mote_renderer import (
        DustMoteComponent,
        LeafLitterComponent,
    )
    dust_go = instantiate()
    dust_go.name = "DustMotes"
    dust_go.add_component(
        DustMoteComponent,
        base=app,
        lighting_pipeline=lighting_pipeline,
    )
    app.dust_go = dust_go

    leaf_go = instantiate()
    leaf_go.name = "LeafLitter"
    leaf_go.add_component(
        LeafLitterComponent,
        base=app,
        zone_store=zone_store,
        lighting_pipeline=lighting_pipeline,
    )
    app.leaf_go = leaf_go

    # 10f. Wind debug ball (dev-only, [debug] debug_wind_ball) — a bright
    #      procedural sphere on the ground near spawn pushed by WindField.sample
    #      each fixed step: the physics seam proof (it scoots on gusts, rolls in
    #      storms).  When the GPU WindSystemComponent is live it already calls
    #      wind_field.update() each frame, so the ball only SAMPLES (sky_system
    #      left None to avoid a redundant update); on the CPU backend that
    #      component disables itself, so the ball drives update() itself (pass
    #      sky_system) so it still has a field to sample.
    if cfg.debug_wind_ball and wind_field is not None:
        from fire_engine.world.wind_debug import WindBallDebugComponent
        wind_component_active = use_gpu_lighting and lighting_pipeline is not None
        ball_go = instantiate()
        ball_go.name = "WindDebugBall"
        ball_go.add_component(
            WindBallDebugComponent,
            base=app,
            clock=clock,
            wind_field=wind_field,
            sky_system=None if wind_component_active else sky_system,
        )
        app.wind_ball_go = ball_go

    # 10g. HDR post-processing pipeline — offscreen linear-HDR scene buffer +
    #      composite (bloom / lens flare / god rays / FXAA).  Built LAST, after
    #      every render node (terrain, sky, grass, flora, wind particles) exists,
    #      since FilterManager redirects the whole scene.  Gated by the
    #      [graphics] preset (gfx_post_process); on failure it disables itself
    #      and the surface shaders keep tonemapping internally (no crash).
    from fire_engine.world.post_process import PostProcessPipeline
    app.post_process = PostProcessPipeline(app, cfg)

    # 11. Resource-manager proof model (non-fatal).
    _load_proof_model(app)

    # --- Demo key bindings (DEV tooling per §5.5) -------------------------
    Path("saves").mkdir(parents=True, exist_ok=True)

    def fire_explosion() -> None:
        """
        Carve a SphereBrush(REMOVE) crater at the terrain under the camera ray.

        Builds the camera ray (origin = camera position, direction = camera
        forward), raycasts the voxel field, and on a hit carves a crater at the
        hit point.  apply_brush flags touched chunks dirty + edited and publishes
        TerrainEditedEvent; remesh_edited then rebuilds the crater chunks (and
        their border neighbours) immediately — bypassing the 2-chunk streaming
        budget — so the crater geometry and its relight (the lighting pipeline's
        own same-frame edit path) appear together on the very next rendered
        frame, with no see-through hole while neighbours wait their turn.
        Bound to left-click (while flying) and to the dev overlay's
        "Fire Explosion" action button.
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
        # Same-frame remesh: the crater (and the faces it exposed in border
        # neighbours) must exist before this frame renders, or the player sees
        # a black hole through the world until the stream budget catches up.
        chunk_manager.remesh_edited(touched, light_sampler)
        # Volumetric flash: a brief, bright point light in the radiance
        # volume — the GI gather carries it into the crater and the
        # froxel fog catches it as a glow.
        if lighting_pipeline is not None:
            from fire_engine.lighting.lights import PointLight
            lighting_pipeline.lights.add(PointLight(
                position=(hit.point.x, hit.point.y, hit.point.z),
                color=(1.0, 0.55, 0.2), intensity=40.0, radius=18.0,
                ttl_s=0.5))
        _log.info("Explosion at %s — %d chunk(s) cratered", hit.point, len(touched))

    def on_click() -> None:
        """
        Left-click dispatch.

        Priority:
          1. Dev *selection* — when the overlay is open with a free cursor the
             click picks/outlines the object or chunk under the cursor.
          2. Re-capture — when the cursor is free but the overlay is closed
             (the player pressed ESC, or alt-tabbed back in), a click re-grabs
             the mouse for free-look, mirroring how FPS games reacquire focus.
          3. Otherwise (flying, cursor captured) it fires the demo explosion.
        """
        if overlay is not None and overlay.handle_world_click():
            return
        if not app.input_state.mouse_captured:
            app.input_state.mouse_captured = True
            app._set_mouse_capture(True)
            return
        fire_explosion()

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

    # --- Sky/weather dev bindings (F6/F7/F8) — dev tooling per §5.5 ---------
    # F6 cycles a forced weather type (None = back to the natural schedule),
    # F7 toggles the game-time scale for day-cycle fast-forward, F8 jumps the
    # game clock forward 6 game-hours to snap to interesting skies.
    weather_cycle: list = [
        WeatherType.CLEAR, WeatherType.CLOUDY, WeatherType.OVERCAST,
        WeatherType.FOG, WeatherType.RAIN, WeatherType.STORM, None,
    ]
    weather_index = [len(weather_cycle) - 1]   # starts at None (natural)

    def on_cycle_weather() -> None:
        """F6 → force the next weather type in the cycle (None = natural)."""
        weather_index[0] = (weather_index[0] + 1) % len(weather_cycle)
        forced = weather_cycle[weather_index[0]]
        sky_system.weather.force_weather(forced)
        st = sky_system.state
        _log.info(
            "Weather forced to %s (current=%s, coverage=%.2f, fog=%.4f /m, "
            "rain=%.2f)",
            forced.name if forced is not None else "None (natural schedule)",
            sky_system.weather.current.name,
            st.cloud_coverage, st.fog_density, st.rain_intensity,
        )

    def on_toggle_time_scale() -> None:
        """F7 → toggle clock.game_time_scale between 60 (normal) and 1800 (fast)."""
        clock.game_time_scale = 1800.0 if clock.game_time_scale <= 60.0 else 60.0
        _log.info("game_time_scale = %.0f (1 real s = %.0f game s)",
                  clock.game_time_scale, clock.game_time_scale)

    def on_jump_time() -> None:
        """F8 → jump the game clock forward 6 game-hours (wraps the day)."""
        new_tod = clock.game_time_of_day + 6.0 * 3600.0
        if new_tod >= 24.0 * 3600.0:
            new_tod -= 24.0 * 3600.0
            clock.game_day += 1
        clock.game_time_of_day = new_tod
        _log.info("Game time jumped to day %d, %02d:%02d", clock.game_day,
                  int(new_tod // 3600), int(new_tod % 3600 // 60))

    # --- Dynamic-light dev bindings (L / K) — GPU lighting backend only -----
    # L drops a warm torch point-light at the camera (watch the gathered GI
    # carry it around corners); K clears all dropped lights.
    def on_drop_torch() -> None:
        """L → drop a permanent torch light at the camera position."""
        if lighting_pipeline is None:
            return
        from fire_engine.lighting.lights import PointLight
        pos = app.camera_go.transform.position
        lighting_pipeline.lights.add(PointLight(
            position=(pos.x, pos.y, pos.z),
            color=(1.0, 0.62, 0.28), intensity=8.0, radius=16.0))
        _log.info("Torch dropped at %s (%d light(s) active)",
                  pos, lighting_pipeline.lights.count)

    def on_clear_lights() -> None:
        """K → remove all dynamic lights."""
        if lighting_pipeline is None:
            return
        lighting_pipeline.lights.clear()
        _log.info("Dynamic lights cleared")

    # --- Flashlight (F) — a SpotLight glued to the camera ------------------
    # The follow task only touches the light set when the camera actually
    # moved/turned (re-injection is the expensive part), and the beam shows
    # up in the froxel fog automatically (GI radiance feeds the fog scatter).
    flashlight: dict = {"id": None, "light": None}

    def on_toggle_flashlight() -> None:
        """F → toggle a camera-mounted flashlight (GPU backend only)."""
        if lighting_pipeline is None:
            return
        from fire_engine.lighting.lights import SpotLight
        if flashlight["id"] is not None:
            lighting_pipeline.lights.remove(flashlight["id"])
            flashlight["id"] = flashlight["light"] = None
            _log.info("Flashlight OFF")
            return
        pos = app.camera_go.transform.position
        fwd = app.camera_go.transform.forward
        light = SpotLight(
            position=(pos.x, pos.y, pos.z),
            direction=(fwd.x, fwd.y, fwd.z),
            color=_FLASHLIGHT_COLOR,
            intensity=_FLASHLIGHT_INTENSITY,
            radius=_FLASHLIGHT_RADIUS_M,
            cone_deg=_FLASHLIGHT_CONE_DEG,
        )
        flashlight["id"] = lighting_pipeline.lights.add(light)
        flashlight["light"] = light
        _log.info("Flashlight ON")

    def _follow_flashlight(task):
        """Per-frame: keep the flashlight on the camera (move/turn eps)."""
        light = flashlight["light"]
        if light is not None:
            pos = app.camera_go.transform.position
            fwd = app.camera_go.transform.forward
            new_pos = (pos.x, pos.y, pos.z)
            new_dir = (fwd.x, fwd.y, fwd.z)
            moved = sum((a - b) ** 2 for a, b in
                        zip(new_pos, light.position)) > 0.15 ** 2
            turned = sum(a * b for a, b in
                         zip(new_dir, light.direction)) < math.cos(
                             math.radians(1.5))
            if moved or turned:
                light.position = new_pos
                light.direction = new_dir
                lighting_pipeline.lights.notify_changed()
        return task.cont

    if lighting_pipeline is not None:
        app.taskMgr.add(_follow_flashlight, "FlashlightFollow")

    app.accept("l", on_drop_torch)
    app.accept("k", on_clear_lights)
    app.accept("f", on_toggle_flashlight)
    app.accept("g", lambda: build_gi_test_room(app))

    app.accept("f6", on_cycle_weather)
    app.accept("f7", on_toggle_time_scale)
    app.accept("f8", on_jump_time)

    # --- Developer overlay (F1) — in-game debug menu / inspector / spawn -----
    # DirectGUI overlay rendered in world/ (the only place panda3d is allowed).
    # The headless DevToolsManager underneath holds the tools, selection, and
    # picking; see docs/systems/devtools.md.  Expose the demo explosion as an
    # action button too so it can be triggered from the menu.
    from fire_engine.world import DevOverlay
    overlay = DevOverlay(app) if DevOverlay is not None else None
    if overlay is not None:
        overlay.actions.add_action("Fire Explosion", fire_explosion)
        app.accept("f1", overlay.toggle)
        # Releasing the mouse ends an in-progress transform-gizmo drag.
        app.accept("mouse1-up", overlay.end_gizmo_drag)
        app.dev_overlay = overlay   # exposed for tooling (tools/screenshot.py)

    app.accept("mouse1", on_click)
    app.accept("f5", on_save)
    app.accept("f9", on_load)

    _log.info("Demo ready — WASD+mouse to fly, ESC to capture mouse, "
              "left-click to explode, F1 dev overlay, F5 save, F9 load, "
              "F6 cycle weather, F7 time scale, F8 +6h, F flashlight, "
              "G GI test room, L torch, K clear lights.")

    return app


def build_gi_test_room(app) -> tuple[float, float, float]:
    """
    Build a Cornell-style GI test room ~14 m ahead of the camera (G key).

    A hollow white room (interior 9×9×4.5 m, 1 m walls) with a RED wall on
    one side and a GREEN wall on the other, an emissive ceiling panel (the
    emission-map path), a 2.5×3 m doorway facing the camera, and a small
    roof hole for a sun shaft.  Walk in and look at the white surfaces:
    red/green colour bleed = bounce GI working; the glow panel lights the
    room with no sky contribution; the roof shaft shows god rays in fog.

    Parameters
    ----------
    app : world.app.App
        The running demo app (uses ``camera_go``, ``chunk_manager``,
        ``_event_bus``, ``_config``).

    Returns
    -------
    tuple[float, float, float] — the room's (cx, cy, floor_z) in meters.
    """
    from fire_engine.terrain import BoxBrush
    chunk_manager = app.chunk_manager
    bus = app._event_bus
    cfg = app._config
    pos = app.camera_go.transform.position
    fwd = app.camera_go.transform.forward
    # Axis-align the room on the camera's dominant horizontal axis so the
    # doorway squarely faces the player.
    along_x = abs(fwd.x) >= abs(fwd.y)
    sign = 1.0 if (fwd.x if along_x else fwd.y) >= 0.0 else -1.0
    cx = pos.x + (14.0 * sign if along_x else 0.0)
    cy = pos.y + (0.0 if along_x else 14.0 * sign)
    hit = raycast_voxel(Vec3(cx, cy, pos.z + 40.0), Vec3(0.0, 0.0, -1.0),
                        chunk_manager.get_or_create, max_distance_m=90.0)
    z0 = (hit.point.z if hit is not None else cfg.ground_height_m) + 0.5
    cz = z0 + 2.25                                       # interior mid-height

    room_touched: set = set()

    def box(half: tuple, at: tuple, mode: BrushMode, material: int = 1):
        room_touched.update(apply_brush(
            BoxBrush(half_extents_m=Vec3(*half)), Vec3(*at), mode,
            material=material,
            chunk_provider=chunk_manager.get_or_create, bus=bus))

    # Solid white block, then hollow the interior.
    box((5.5, 5.5, 3.25), (cx, cy, z0 + 2.25), BrushMode.ADD, _MAT_GI_WHITE)
    box((4.5, 4.5, 2.25), (cx, cy, cz), BrushMode.REMOVE)
    # Red / green side walls (overwrite the inner half of the white wall)
    # on the lateral axis (perpendicular to the doorway axis).
    if along_x:
        box((0.5, 4.5, 2.25), (cx, cy - 5.0, cz), BrushMode.ADD, _MAT_GI_RED)
        box((0.5, 4.5, 2.25), (cx, cy + 5.0, cz), BrushMode.ADD, _MAT_GI_GREEN)
    else:
        box((0.5, 4.5, 2.25), (cx - 5.0, cy, cz), BrushMode.ADD, _MAT_GI_RED)
        box((0.5, 4.5, 2.25), (cx + 5.0, cy, cz), BrushMode.ADD, _MAT_GI_GREEN)
    # Emissive ceiling panel (protrudes 0.25 m below the ceiling).
    box((1.5, 1.5, 0.5), (cx, cy, z0 + 4.75), BrushMode.ADD, _MAT_GI_GLOW)
    # Doorway through the camera-facing wall + a small roof shaft hole.
    if along_x:
        box((0.75, 1.25, 1.5), (cx - sign * 5.0, cy, z0 + 1.5),
            BrushMode.REMOVE)
    else:
        box((1.25, 0.75, 1.5), (cx, cy - sign * 5.0, z0 + 1.5),
            BrushMode.REMOVE)
    box((0.75, 0.75, 1.0), (cx + 2.8, cy + 2.8, z0 + 5.0), BrushMode.REMOVE)

    # Same-frame remesh of every chunk the room carved (plus border
    # neighbours) — one hitch instead of seconds of see-through walls while
    # the 2-chunk stream budget catches up.
    chunk_manager.remesh_edited(room_touched, app.light_sampler)

    # Co-locate an AreaLight with the emissive ceiling panel.  The voxel
    # emission alone glows but only fills its diffusion reach (~4 m); a real
    # box light gives the whole 9 m room inverse-square direct fill, the
    # coloured side walls then bleed red/green onto the white surfaces via the
    # GI gather (the bounce we want to demonstrate), and — unlike voxel
    # emission — the exposure meter can see it, so the aperture settles sanely.
    pipeline = getattr(app, "lighting_pipeline", None)
    if pipeline is not None:
        from fire_engine.lighting.lights import AreaLight
        pipeline.lights.add(AreaLight(
            center=(cx, cy, z0 + 4.1),            # just below the panel face
            half_extents=(1.5, 1.5, 0.15),
            color=_GI_PANEL_COLOR, intensity=_GI_PANEL_INTENSITY,
            radius=18.0))

    _log.info("GI test room built at (%.1f, %.1f, %.1f) — walk in and watch "
              "the white walls pick up red/green bounce", cx, cy, z0)
    return cx, cy, z0


def main() -> None:
    """Boot the demo and run the blocking main loop (opens a window)."""
    app = build_demo()
    # 12. Run (blocks until the window closes).  The try/finally stops the
    #     GPU-lighting assembly worker thread cleanly on exit (it is a daemon,
    #     so this is belt-and-suspenders, not strictly required).
    try:
        app.run()
    finally:
        pipeline = getattr(app, "lighting_pipeline", None)
        if pipeline is not None:
            pipeline.shutdown()
        # Stop the wind venturi worker thread cleanly on exit (mirrors the
        # lighting assembly worker shutdown above).  The WindSystemComponent
        # also stops it in on_destroy, so this is belt-and-suspenders for the
        # exit path that tears the window down without destroying components.
        wind_worker = getattr(app, "wind_worker", None)
        if wind_worker is not None:
            wind_worker.stop(join=True)


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
        from fire_engine.world.texture_bridge import to_panda_texture
        rgba = get_procedural("wasteland_ground")   # (256,256,4) uint8
        return to_panda_texture(rgba)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Ground texture build failed (untextured terrain): %s", exc)
        return None


def _to_material_textures(triples: bool = False):
    """
    Build the material id → Panda3D texture map for terrain rendering.

    Grass-skin faces (``MATERIAL_GRASS``, the baseline's top voxel layer) get
    the pixel-art ``grass_ground`` texture; dirt bulk (``MATERIAL_DIRT``,
    exposed by digging) gets ``dirt_ground``.  Used by the faceted mesher's
    per-material Geom split (world/geometry_bridge.to_geom_node).  Returns
    None (and logs) on any failure so the demo falls back to the node-level
    ``wasteland_ground`` texture.

    Parameters
    ----------
    triples : bool, default False
        When True (GPU lighting backend), each material maps to an
        ``(albedo, normal_map, emission_map)`` texture triple for the
        volumetric terrain shader; normal maps are derived from the albedo
        luminance (procedural/maps.py), emission defaults to black.

    Returns
    -------
    dict[int, panda3d.core.Texture | tuple] | None
    """
    try:
        from fire_engine.world.texture_bridge import to_panda_texture
        from fire_engine.terrain import MATERIAL_DIRT, MATERIAL_GRASS
        if not triples:
            return {
                MATERIAL_DIRT: to_panda_texture(get_procedural("dirt_ground")),
                MATERIAL_GRASS: to_panda_texture(get_procedural("grass_ground")),
            }
        from fire_engine.procedural.maps import (
            black_emission_map,
            derive_normal_map,
            flat_normal_map,
        )
        import numpy as np
        emis_tex = to_panda_texture(black_emission_map())
        flat_n_tex = to_panda_texture(flat_normal_map())

        def triple(def_name: str):
            rgba = get_procedural(def_name)
            return (to_panda_texture(rgba),
                    to_panda_texture(derive_normal_map(rgba)),
                    emis_tex)

        def flat_rgba(rgb_linear, alpha: int = 255) -> np.ndarray:
            """16×16 solid-colour RGBA from linear RGB (sRGB-encoded)."""
            srgb = (np.clip(np.asarray(rgb_linear, np.float32), 0.0, 1.0)
                    ** (1.0 / 2.2) * 255.0).astype(np.uint8)
            arr = np.empty((16, 16, 4), dtype=np.uint8)
            arr[..., :3] = srgb
            arr[..., 3] = alpha
            return arr

        def flat_triple(material_id: int, emissive: bool = False):
            alb = to_panda_texture(flat_rgba(_GI_TEST_ALBEDO[material_id]))
            em = to_panda_texture(flat_rgba((1.0, 0.92, 0.78))) if emissive \
                else emis_tex
            return (alb, flat_n_tex, em)

        textures = {
            MATERIAL_DIRT: triple("dirt_ground"),
            MATERIAL_GRASS: triple("grass_ground"),
        }
        # GI test-room debug materials: flat colours + (for the glow panel)
        # a bright emission map so the surface itself glows on screen.
        for mid in (_MAT_GI_WHITE, _MAT_GI_RED, _MAT_GI_GREEN):
            textures[mid] = flat_triple(mid)
        textures[_MAT_GI_GLOW] = flat_triple(_MAT_GI_GLOW, emissive=True)
        return textures
    except Exception as exc:  # noqa: BLE001
        _log.warning("Material textures build failed (fallback texture): %s", exc)
        return None


if __name__ == "__main__":
    main()
