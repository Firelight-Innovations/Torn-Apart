"""
lighting/gpu.py — GPU volumetric lighting pipeline (panda3d-side).

The one lighting file allowed to touch the GPU (ARCHITECTURE §4 rule 4:
"Lighting API excepted for light-grid GPU work").  Owns:

- the per-cascade 3-D textures (geometry/emission uploads from
  `lighting/volume.py` numpy blocks; GPU-written visibility, lit-source and
  ping-pong radiance volumes),
- compute-shader dispatch (`GraphicsEngine.dispatch_compute`) for injection,
  the ray-marched GI gather and froxel fog (sources in `lighting/glsl.py`),
- the per-frame schedule: re-assemble/upload only when a cascade window
  recenters or terrain changes; re-inject + re-gather only when the volume,
  sun/moon, sky or dynamic lights changed; fog every frame,
- the shader-input contract consumed by every lit-surface shader (the GLSL
  side lives in `world/shaders/lit_surface.glsl`): `bind_surface_inputs`
  once on ``app.render`` + `update_surface_inputs` there per frame —
  terrain, foliage and future buildings/NPCs inherit it scene-graph-wide.

Everything headless (window math, assembly, light packing) lives in the
sibling panda3d-free modules so it stays unit-testable; this module is
excluded from the headless suite.

Example (wired by main.py when ``config.lighting_backend == "gpu"``)
--------------------------------------------------------------------
    pipeline = GpuLightingPipeline(cfg, app, chunk_manager, bus)
    app.lighting_pipeline = pipeline          # App.update calls it per frame
    apply_terrain_shader(app.terrain_root, pipeline)   # world/terrain_shader
    pipeline.bind_surface_inputs(app.render)  # lit-surface contract for ALL
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import LVecBase4f, NodePath, Texture

from fire_engine.core import (
    ChunkLoadedEvent,
    Config,
    EventBus,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.lighting import glsl
from fire_engine.lighting._impl.gpu_cascade import _Cascade as _CascadeImpl
from fire_engine.lighting._impl.gpu_cascade_assembly import (
    apply_edits_sync as _apply_edits_sync_fn,
)
from fire_engine.lighting._impl.gpu_cascade_assembly import (
    assemble_and_upload_sync as _assemble_and_upload_sync_fn,
)
from fire_engine.lighting._impl.gpu_cascade_assembly import (
    drain_assembly_results as _drain_assembly_results_fn,
)
from fire_engine.lighting._impl.gpu_cascade_assembly import (
    schedule_assembly as _schedule_assembly_fn,
)
from fire_engine.lighting._impl.gpu_fog import dispatch_fog as _dispatch_fog_fn
from fire_engine.lighting._impl.gpu_fog import setup_fog as _setup_fog_fn
from fire_engine.lighting._impl.gpu_inject_gather import inject_and_gather as _inject_and_gather_fn
from fire_engine.lighting._impl.gpu_surface import (
    bind_surface_inputs as _bind_surface_inputs_fn,
)
from fire_engine.lighting._impl.gpu_surface import (
    sky_inputs as _sky_inputs_fn,
)
from fire_engine.lighting._impl.gpu_surface import (
    update_surface_inputs as _update_surface_inputs_fn,
)
from fire_engine.lighting.assembly_worker import CascadeAssemblyWorker
from fire_engine.lighting.exposure import ExposureMeter
from fire_engine.lighting.lights import LightSet, OccluderSet
from fire_engine.lighting.occluders import TreeOccluderSet
from fire_engine.lighting.palette import MaterialPalette, build_default_palette
from fire_engine.lighting.volume import VolumeWindow

if TYPE_CHECKING:
    from fire_engine.world.sky.sky_state import SkyState

_log = get_logger("lighting.gpu")

# Reassembles triggered by chunk *loads* are batched to at most one per this
# interval (boot streaming loads chunks every frame); brush edits and window
# recenters reassemble immediately.
_LOAD_REASSEMBLE_INTERVAL_S = 0.25


class GpuLightingPipeline:
    """
    Owner of the GPU lighting state; one instance per App.

    Parameters
    ----------
    config : Config
        Engine config (``[lighting]`` / ``[fog]`` fields).
    base : ShowBase
        The running App — provides ``graphicsEngine``, ``win`` (GSG) and the
        camera/lens for froxel fog.
    chunk_provider : object
        Anything with a ``.chunks`` dict (``ChunkManager``).
    bus : EventBus
        Subscribes to ``TerrainEditedEvent`` / ``ChunkLoadedEvent`` to keep
        the volumes current.
    palette : MaterialPalette | None
        Material light-response palette; defaults to
        :func:`build_default_palette`.

    Attributes
    ----------
    lights : LightSet
        Public registry for dynamic point/area/spot lights.
    occluders : OccluderSet
        Public registry for dynamic shadow-caster AABBs (dev cubes, props):
        objects not in the voxel field that should still cast shadows and
        cut god rays.  Sync world boxes once per frame via ``set_boxes``.
    exposure : float
        Current tonemap exposure = ``config.light_exposure`` × the
        auto-exposure (eye adaptation) multiplier.  Written to the terrain
        shader per frame.
    exposure_sky : float
        The sky dome's exposure: the same adaptation COMPRESSED
        (``mult ** 0.35``) so a dark-adapted eye brightens the terrain
        without washing the night sky milky (stars/galaxy composite in LDR
        and would lose all contrast at the full multiplier).
    """

    # ------------------------------------------------------------------
    # Class-level attribute annotations (for mypy and _impl helpers)
    # ------------------------------------------------------------------
    _config: Config
    _base: Any
    _provider: Any
    _palette: MaterialPalette
    _threaded: bool
    _assembly_seq: int
    _assembly_worker: CascadeAssemblyWorker | None
    _tree_occluders: TreeOccluderSet | None
    _tree_occ_stale: set[int]
    _geometry_providers: list[Any]
    _box_uniforms: tuple[Any, ...] | None
    _pending_coords: set[tuple[int, int, int]]
    _edited_coords: set[tuple[int, int, int]]
    _force_all_dirty: bool
    _load_dirty_timer: float
    _last_sun: tuple[Any, ...] | None
    _gi_iters: int
    _gi_smooth: int
    _fog_dim: tuple[int, int, int]
    _fog_near: float
    _fog_far: float
    _fog_scatter_np: NodePath
    _fog_integrate_np: NodePath
    fog_scatter_tex: Texture
    fog_integrated_tex: Texture

    def __init__(
        self,
        config: Config,
        base: Any,
        chunk_provider: Any,
        bus: EventBus,
        palette: MaterialPalette | None = None,
        *,
        threaded: bool = True,
    ) -> None:
        self._config = config
        self._base = base
        self._provider = chunk_provider
        self._palette = palette if palette is not None else build_default_palette()
        self._threaded = bool(threaded)
        self._assembly_seq = 0
        self._assembly_worker = CascadeAssemblyWorker() if self._threaded else None
        if self._assembly_worker is not None:
            self._assembly_worker.start()
        self.lights = LightSet()
        self._lights_version_seen = -1
        self.occluders = OccluderSet()
        self._occluders_version_seen = -1
        self._tree_occluders = None
        self._tree_occ_stale: set[int] = set()
        self._geometry_providers: list[Any] = []
        self._box_uniforms = None
        self.exposure_meter = ExposureMeter(config)
        self.exposure = float(config.light_exposure)
        self.exposure_sky = float(config.light_exposure)

        from panda3d.core import Shader

        inject_shader = Shader.make_compute(Shader.SL_GLSL, glsl.INJECT_COMPUTE)
        gather_shader = Shader.make_compute(Shader.SL_GLSL, glsl.GATHER_COMPUTE)
        smooth_shader = Shader.make_compute(Shader.SL_GLSL, glsl.SMOOTH_COMPUTE)
        shift_shader = Shader.make_compute(Shader.SL_GLSL, glsl.SHIFT_COMPUTE)

        bounce = float(config.light_bounce_strength)
        gi_rays = int(config.light_gi_rays)
        gi_steps = int(config.light_gi_steps)
        self._gi_iters = max(1, int(config.light_gi_iters))
        self._gi_smooth = max(0, int(config.light_gi_smooth_passes))
        self.cascades: list[_CascadeImpl] = [
            _CascadeImpl(
                0,
                config.light_c0_cells,
                config.light_c0_cell_m,
                inject_shader,
                gather_shader,
                smooth_shader,
                shift_shader,
                bounce,
                gi_rays,
                gi_steps,
            ),
            _CascadeImpl(
                1,
                config.light_c1_cells,
                config.light_c1_cell_m,
                inject_shader,
                gather_shader,
                smooth_shader,
                shift_shader,
                bounce,
                gi_rays,
                gi_steps,
            ),
            _CascadeImpl(
                2,
                config.light_c2_cells,
                config.light_c2_cell_m,
                inject_shader,
                gather_shader,
                smooth_shader,
                shift_shader,
                bounce,
                gi_rays,
                gi_steps,
                margin_cells=16,
            ),
        ]

        self.fog_enabled = bool(config.fog_enabled)
        self._fog_dim = (config.fog_froxels_x, config.fog_froxels_y, config.fog_froxels_z)
        self._fog_near = 0.5
        self._fog_far = float(config.fog_far_m)
        if self.fog_enabled:
            _setup_fog_fn(self)

        self._pending_coords: set[tuple[int, int, int]] = set()
        self._edited_coords: set[tuple[int, int, int]] = set()
        self._force_all_dirty = True
        self._load_dirty_timer = 0.0
        self._last_sun = None
        bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)
        bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)

        _log.info(
            "GPU lighting: cascade0 %d^3 @ %.2f m, cascade1 %d^3 @ %.2f m, "
            "cascade2 %d^3 @ %.2f m, fog %s",
            config.light_c0_cells,
            config.light_c0_cell_m,
            config.light_c1_cells,
            config.light_c1_cell_m,
            config.light_c2_cells,
            config.light_c2_cell_m,
            "x".join(map(str, self._fog_dim)) if self.fog_enabled else "off",
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        """Brush edit → reassemble the affected cascades immediately."""
        coords = event.chunk_coords
        if isinstance(coords, tuple) and len(coords) == 3 and isinstance(coords[0], int):
            edited: tuple[Any, ...] = (coords,)
        else:
            edited = tuple(coords)
        self._pending_coords.update(edited)
        self._edited_coords.update(edited)
        if self._assembly_worker is not None:
            for c in edited:
                self._assembly_worker.invalidate_chunk(c)
        self._load_dirty_timer = _LOAD_REASSEMBLE_INTERVAL_S

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        """Chunk streamed in → reassemble soon (batched while streaming)."""
        self._pending_coords.add(event.coord)

    def _coords_hit_window(self, window: VolumeWindow) -> bool:
        """True when any pending chunk overlaps the window's world box."""
        return self._any_coord_hits(self._pending_coords, window)

    def _any_coord_hits(self, coords: Any, window: VolumeWindow) -> bool:
        """True when any coord in ``coords`` overlaps the window's world box."""
        if window.origin_cell is None:
            return True
        chunk_m = self._config.chunk_meters
        lo = window.world_origin_m
        size = window.size_m
        for c in coords:
            if all(
                c[i] * chunk_m < lo[i] + size and (c[i] + 1) * chunk_m > lo[i] for i in range(3)
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Per-frame driver
    # ------------------------------------------------------------------

    def update(self, camera_pos: Any, sky_state: SkyState | None, dt: float) -> None:
        """
        Advance the GPU lighting one frame.

        Parameters
        ----------
        camera_pos : Vec3 | sequence of 3 floats
            Camera world position (meters) — cascade windows follow it.
        sky_state : SkyState | None
            Current sky snapshot (sun/moon/ambient/fog).  ``None`` falls
            back to a fixed overhead white sun (tooling without a sky).
        dt : float
            Real frame seconds (transient-light fades).
        """
        self.lights.update(dt)
        self._load_dirty_timer += dt

        sun = self._sky_inputs(sky_state)
        packed, count = self.lights.pack(glsl.MAX_LIGHTS)

        # 0. Auto-exposure (eye adaptation).
        mult = self.exposure_meter.update(
            camera_pos, sky_state, self._provider.chunks, (packed, count), dt
        )
        base = float(self._config.light_exposure)
        self.exposure = base * mult
        self.exposure_sky = base * (mult**0.35)

        # 1. Window follow + geometry reassembly.
        if self._force_all_dirty:
            for casc in self.cascades:
                casc.window.recenter(camera_pos)
                _assemble_and_upload_sync_fn(self, casc)
            self._pending_coords.clear()
            self._load_dirty_timer = 0.0
            self._force_all_dirty = False
        else:
            _apply_edits_sync_fn(self)
            _schedule_assembly_fn(self, camera_pos)
            _drain_assembly_results_fn(self)

        # 2. Celestial / sky change detection (affects every cascade).
        sun_changed = self._last_sun is None or any(
            _changed(a, b) for a, b in zip(sun, self._last_sun, strict=True)
        )
        if sun_changed:
            for casc in self.cascades:
                casc.needs_inject = True
            self._last_sun = sun

        # 3. Dynamic lights / occluder change detection.
        if self.lights.version != self._lights_version_seen:
            for casc in self.cascades:
                casc.needs_inject = True
            self._lights_version_seen = self.lights.version
        if self.occluders.version != self._occluders_version_seen:
            for casc in self.cascades:
                casc.needs_inject = True
            self._occluders_version_seen = self.occluders.version
            self._box_uniforms = None

        gsg = self._base.win.get_gsg()
        engine = self._base.graphicsEngine

        if self._box_uniforms is None:
            mins, maxs, n_boxes = self.occluders.pack()
            self._box_uniforms = (
                [LVecBase4f(mins[i, 0], mins[i, 1], mins[i, 2], 0.0) for i in range(mins.shape[0])],
                [LVecBase4f(maxs[i, 0], maxs[i, 1], maxs[i, 2], 0.0) for i in range(maxs.shape[0])],
                int(n_boxes),
            )
        box_min, box_max, n_boxes = self._box_uniforms

        # 4. Injection + gather.
        if any(c.needs_inject for c in self.cascades):
            _inject_and_gather_fn(self, sun, packed, count, box_min, box_max, n_boxes, engine, gsg)

        # 5. Froxel fog.
        if self.fog_enabled:
            _dispatch_fog_fn(self, camera_pos, sun, sky_state, engine, gsg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_static_occluders(self, occluders: TreeOccluderSet | None) -> None:
        """
        Replace the static tree/bush occluder set the cascades are lit with.

        Called by ``world/tree_renderer.py`` after every placement (re)bake.
        The set is splatted as fractional occupancy + bounce albedo into every
        geometry volume assembled from now on (``lighting/occluders.py``), and
        every cascade with a committed volume is queued for an async re-splat
        at its current origin.  Canopies are a translucent leaf medium (per-
        meter extinction from each instance's ``canopy_sigma`` ×
        ``light_tree_canopy_extinction_gain``); trunks splat at
        ``light_tree_trunk_occ``.

        Parameters
        ----------
        occluders : TreeOccluderSet | None
            Merged instance set for ALL tree/bush volumes.  ``None`` or an
            empty set clears tree occlusion entirely.
        """
        if occluders is not None and occluders.count == 0:
            occluders = None
        self._tree_occluders = occluders
        self._tree_occ_stale = {c.index for c in self.cascades if c.window.origin_cell is not None}

    def register_geometry_provider(self, provider: Any) -> None:
        """
        Register a non-terrain geometry occupancy provider (e.g. building
        occlusion rasterizer) so it can splat its solids into the cascades.

        **v1 is store-only**: the provider is recorded but NOT yet threaded
        into the async :class:`CascadeAssemblyWorker`.  The synchronous
        ``assemble_geometry(..., providers=...)`` path already honours
        providers; this registry is where the pipeline will pass them once
        snapshotting lands.

        Parameters
        ----------
        provider : GeometryOccupancyProvider
            Structural provider (no import coupling to the buildings package).
        """
        self._geometry_providers.append(provider)

    def geometry_providers(self) -> tuple[Any, ...]:
        """The registered geometry providers (store-only in v1)."""
        return tuple(self._geometry_providers)

    def bind_surface_inputs(self, node: NodePath) -> None:
        """
        Bind the static lighting samplers/uniforms onto a render NodePath.

        Call once at boot with ``app.render`` (main.py does) so every shader
        that includes ``world/shaders/lit_surface.glsl`` — terrain, foliage,
        future buildings/NPCs — inherits the contract scene-graph-wide.
        Per-frame values are refreshed by :meth:`update_surface_inputs`.
        Shaders that don't declare these uniforms simply ignore them.
        """
        _bind_surface_inputs_fn(self, node)

    def update_surface_inputs(self, node: NodePath, sky_state: SkyState | None) -> None:
        """
        Refresh the per-frame lighting uniforms on a render NodePath.

        Parameters
        ----------
        node : NodePath
            Same node given to :meth:`bind_surface_inputs` (``app.render``).
        sky_state : SkyState | None
            For sun/moon direction + radiance uniforms.
        """
        _update_surface_inputs_fn(self, node, sky_state)

    def shutdown(self) -> None:
        """
        Stop the background assembly worker.  Call once on app exit.

        Idempotent and safe to call when running unthreaded (no-op).
        """
        if self._assembly_worker is not None:
            self._assembly_worker.stop(join=True)
            self._assembly_worker = None

    @staticmethod
    def _sky_inputs(sky_state: SkyState | None) -> tuple[Any, ...]:
        """
        Extract (sun_dir, sun_radiance, moon_dir, moon_radiance, sky_ambient)
        from a SkyState, with graceful fallbacks for older SkyState versions
        (radiance derived from sun_color × intensity) and for ``None``.
        """
        return _sky_inputs_fn(sky_state)


def _groups(n: int, local: int) -> int:
    """Workgroup count covering ``n`` invocations at ``local`` per group."""
    return (n + local - 1) // local


def _changed(a: Any, b: Any, eps: float = 0.004) -> bool:
    """True when two float tuples differ beyond ``eps`` on any component."""
    return any(abs(float(x) - float(y)) > eps for x, y in zip(a, b, strict=True))
