"""
lighting/gpu.py — GPU volumetric lighting pipeline (panda3d-side).

The one lighting file allowed to touch the GPU (ARCHITECTURE §4 rule 4:
"Lighting API excepted for light-grid GPU work").  Owns:

- the per-cascade 3-D textures (geometry/emission uploads from
  `lighting/volume.py` numpy blocks; GPU-written visibility, direct and
  ping-pong radiance volumes),
- compute-shader dispatch (`GraphicsEngine.dispatch_compute`) for injection,
  flood-fill propagation and froxel fog (sources in `lighting/glsl.py`),
- the per-frame schedule: re-assemble/upload only when a cascade window
  recenters or terrain changes; re-inject only when the volume, sun/moon,
  sky or dynamic lights changed; propagate + fog every frame,
- the shader-input contract consumed by `world/terrain_shader.py`
  (`bind_surface_inputs` once + `update_surface_inputs` per frame).

Everything headless (window math, assembly, light packing) lives in the
sibling panda3d-free modules so it stays unit-testable; this module is
excluded from the headless suite.

Example (wired by main.py when ``config.lighting_backend == "gpu"``)
--------------------------------------------------------------------
    pipeline = GpuLightingPipeline(cfg, app, chunk_manager, bus)
    app.lighting_pipeline = pipeline          # App.update calls it per frame
    apply_terrain_shader(app.terrain_root, pipeline)   # world/terrain_shader
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

# panda3d imports are allowed in lighting/ (ARCHITECTURE §3).
from panda3d.core import (  # type: ignore[import]
    LVecBase2f,
    LVecBase3f,
    LVecBase3i,
    LVecBase4f,
    NodePath,
    SamplerState,
    Shader,
    ShaderAttrib,
    Texture,
)

from fire_engine.core import (
    ChunkLoadedEvent,
    Config,
    EventBus,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.lighting import glsl
from fire_engine.lighting.exposure import ExposureMeter
from fire_engine.lighting.lights import LightSet, OccluderSet
from fire_engine.lighting.palette import MaterialPalette, build_default_palette
from fire_engine.lighting.assembly_worker import (
    AssemblyJob,
    CascadeAssemblyWorker,
    assemble_packed,
)
from fire_engine.lighting.volume import (
    EMISSION_SCALE,
    VolumeWindow,
    assemble_geometry,
    pack_volume,
    window_chunk_span,
)

if TYPE_CHECKING:
    from fire_engine.sky.sky_state import SkyState

_log = get_logger("lighting.gpu")

# Reassembles triggered by chunk *loads* are batched to at most one per this
# interval (boot streaming loads chunks every frame); brush edits and window
# recenters reassemble immediately.
_LOAD_REASSEMBLE_INTERVAL_S = 0.25

# Multiplier turning the weather's subtle exponential fog density into a
# visually volumetric medium (tuned in-game).
_FOG_DENSITY_BOOST = 2.0


def _make_volume_texture(name: str, cells: int, *, hdr: bool,
                         linear: bool) -> Texture:
    """
    Allocate one cascade 3-D texture.

    Parameters
    ----------
    name : str
        Debug name.
    cells : int
        Texels per axis.
    hdr : bool
        True → ``rgba16f`` (GPU-written radiance), False → ``rgba8``
        (CPU-uploaded geometry/emission or GPU-written visibility).
    linear : bool
        Trilinear filtering (radiance/visibility sampling) vs nearest.
    """
    tex = Texture(name)
    if hdr:
        tex.setup_3d_texture(cells, cells, cells,
                             Texture.T_float, Texture.F_rgba16)
        tex.set_keep_ram_image(False)
    else:
        tex.setup_3d_texture(cells, cells, cells,
                             Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.set_clear_color((0.0, 0.0, 0.0, 0.0))
    filt = SamplerState.FT_linear if linear else SamplerState.FT_nearest
    tex.set_minfilter(filt)
    tex.set_magfilter(filt)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    tex.set_wrap_w(SamplerState.WM_clamp)
    return tex


def _upload_volume(tex: Texture, arr: np.ndarray) -> None:
    """
    Upload a ``uint8 (N, N, N, 4)`` ``[x, y, z]``-indexed block to a 3-D texture.

    Panda3D RAM layout for 3-D textures is page-major ``(z, y, x)`` with BGRA
    channel order, so the block is transposed and channel-swapped (in
    ``volume.pack_volume``) — one bulk ``set_ram_image`` write, no loops.  Used
    by the synchronous boot path; the steady-state async path packs on the
    worker thread and calls ``set_ram_image`` with the bytes directly.
    """
    tex.set_ram_image(pack_volume(arr))


class _Cascade:
    """One radiance cascade: window + textures + compute node paths."""

    def __init__(self, index: int, cells: int, cell_m: float,
                 inject_shader: Shader, propagate_shader: Shader,
                 decay: float, bounce: float) -> None:
        self.index = index
        self.window = VolumeWindow(cells=cells, cell_m=cell_m)
        self.cells = cells
        self.cell_m = cell_m
        self.decay = decay

        self.geom = _make_volume_texture(f"lit_geom_{index}", cells,
                                         hdr=False, linear=True)
        self.emis = _make_volume_texture(f"lit_emis_{index}", cells,
                                         hdr=False, linear=True)
        self.vis = _make_volume_texture(f"lit_vis_{index}", cells,
                                        hdr=True, linear=True)
        self.direct = _make_volume_texture(f"lit_direct_{index}", cells,
                                           hdr=True, linear=True)
        self.radiance = [
            _make_volume_texture(f"lit_rad_{index}_a", cells,
                                 hdr=True, linear=True),
            _make_volume_texture(f"lit_rad_{index}_b", cells,
                                 hdr=True, linear=True),
        ]
        self.ping = 0   # index of the radiance texture holding current light
        self.needs_inject = True   # re-run the injection pass next update
        # Async assembly bookkeeping (main thread only): one job in flight per
        # cascade at a time; ``window.origin_cell`` is the COMMITTED origin (the
        # one the uploaded geom + shader uniforms use) and only advances when a
        # result lands.  ``_pending_seq`` matches the in-flight job.
        self._assembly_inflight = False
        self._pending_seq = -1

        # Injection node (inputs refreshed before each dirty dispatch).
        self.inject_np = NodePath(f"lit_inject_{index}")
        self.inject_np.set_shader(inject_shader)
        self.inject_np.set_shader_input("u_geom", self.geom)
        self.inject_np.set_shader_input("u_emis", self.emis)
        self.inject_np.set_shader_input("u_vis", self.vis)
        self.inject_np.set_shader_input("u_direct", self.direct)
        self.inject_np.set_shader_input("u_cells", cells)
        self.inject_np.set_shader_input("u_emission_scale",
                                        float(EMISSION_SCALE))

        # Two pre-bound propagate nodes: a→b and b→a.
        self.prop_np: list[NodePath] = []
        for src, dst in ((0, 1), (1, 0)):
            pn = NodePath(f"lit_prop_{index}_{src}{dst}")
            pn.set_shader(propagate_shader)
            pn.set_shader_input("u_prev", self.radiance[src])
            pn.set_shader_input("u_next", self.radiance[dst])
            pn.set_shader_input("u_direct", self.direct)
            pn.set_shader_input("u_geom", self.geom)
            pn.set_shader_input("u_cells", cells)
            pn.set_shader_input("u_decay", float(decay))
            pn.set_shader_input("u_bounce", float(bounce))
            self.prop_np.append(pn)

    @property
    def radiance_current(self) -> Texture:
        """The radiance texture holding the latest propagated light."""
        return self.radiance[self.ping]

    def origin_m(self) -> tuple[float, float, float]:
        """World min-corner of the window (meters)."""
        return self.window.world_origin_m


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

    def __init__(
        self,
        config: Config,
        base,
        chunk_provider,
        bus: EventBus,
        palette: MaterialPalette | None = None,
        *,
        threaded: bool = True,
    ) -> None:
        self._config = config
        self._base = base
        self._provider = chunk_provider
        self._palette = palette if palette is not None \
            else build_default_palette()
        # Background cascade-volume assembly: the CPU gather+pack (~90 ms p99 on
        # a fly-around) runs off the main thread so flying stays smooth.  Set
        # ``threaded=False`` for deterministic tooling/tests (assembles inline).
        self._threaded = bool(threaded)
        self._assembly_seq = 0
        self._assembly_worker = CascadeAssemblyWorker() if self._threaded \
            else None
        if self._assembly_worker is not None:
            self._assembly_worker.start()
        self.lights = LightSet()
        self._lights_version_seen = -1
        self.occluders = OccluderSet()
        self._occluders_version_seen = -1
        self._box_uniforms: tuple | None = None   # cached LVecBase4f lists
        # Auto-exposure (eye adaptation): headless meter; `exposure` is the
        # final tonemap exposure consumed by the terrain + sky shaders.
        self.exposure_meter = ExposureMeter(config)
        self.exposure = float(config.light_exposure)
        self.exposure_sky = float(config.light_exposure)

        inject_shader = Shader.make_compute(
            Shader.SL_GLSL, glsl.INJECT_COMPUTE)
        propagate_shader = Shader.make_compute(
            Shader.SL_GLSL, glsl.PROPAGATE_COMPUTE)

        # Diffusion reach ≈ 1/(1−decay) cells; aim a few metres on each cascade.
        # Cascade 2 is the coarse FAR cascade (8 m cells, 512 m box): it keeps
        # distant terrain lit with low-resolution shadows + GI once a surface
        # leaves cascade 1, instead of the old hard fall-back to flat sky
        # ambient.  It rides the exact same off-thread assembly + inject +
        # propagate machinery as the others (the per-frame loops iterate
        # ``self.cascades``), so "bake far chunks on a separate thread at a
        # lower resolution" needs no new subsystem.
        self.cascades = [
            _Cascade(0, config.light_c0_cells, config.light_c0_cell_m,
                     inject_shader, propagate_shader,
                     decay=math.exp(-config.light_c0_cell_m / 4.0),
                     bounce=float(config.light_bounce_strength)),
            _Cascade(1, config.light_c1_cells, config.light_c1_cell_m,
                     inject_shader, propagate_shader,
                     decay=math.exp(-config.light_c1_cell_m / 10.0),
                     bounce=float(config.light_bounce_strength)),
            _Cascade(2, config.light_c2_cells, config.light_c2_cell_m,
                     inject_shader, propagate_shader,
                     decay=math.exp(-config.light_c2_cell_m / 24.0),
                     bounce=float(config.light_bounce_strength)),
        ]

        # --- froxel fog -------------------------------------------------
        self.fog_enabled = bool(config.fog_enabled)
        self._fog_dim = (config.fog_froxels_x, config.fog_froxels_y,
                         config.fog_froxels_z)
        self._fog_near = 0.5
        self._fog_far = float(config.fog_far_m)
        if self.fog_enabled:
            w, h, z = self._fog_dim
            self.fog_scatter_tex = Texture("fog_scatter")
            self.fog_scatter_tex.setup_3d_texture(
                w, h, z, Texture.T_float, Texture.F_rgba16)
            self.fog_integrated_tex = Texture("fog_integrated")
            self.fog_integrated_tex.setup_3d_texture(
                w, h, z, Texture.T_float, Texture.F_rgba16)
            for t in (self.fog_scatter_tex, self.fog_integrated_tex):
                t.set_clear_color((0, 0, 0, 1))
                t.set_keep_ram_image(False)
                t.set_minfilter(SamplerState.FT_linear)
                t.set_magfilter(SamplerState.FT_linear)
                t.set_wrap_u(SamplerState.WM_clamp)
                t.set_wrap_v(SamplerState.WM_clamp)
                t.set_wrap_w(SamplerState.WM_clamp)

            self._fog_scatter_np = NodePath("fog_scatter")
            self._fog_scatter_np.set_shader(Shader.make_compute(
                Shader.SL_GLSL, glsl.FOG_SCATTER_COMPUTE))
            sn = self._fog_scatter_np
            sn.set_shader_input("u_froxels", self.fog_scatter_tex)
            sn.set_shader_input("u_froxel_dim", LVecBase3i(w, h, z))
            sn.set_shader_input("u_fog_near", self._fog_near)
            sn.set_shader_input("u_fog_far", self._fog_far)
            sn.set_shader_input("u_ground_z",
                                float(config.ground_height_m))
            sn.set_shader_input("u_anisotropy",
                                float(config.fog_anisotropy))
            c1 = self.cascades[1]
            sn.set_shader_input("u_c1_vis", c1.vis)
            sn.set_shader_input("u_c1_cells", float(c1.cells))
            sn.set_shader_input("u_c1_cell_m", float(c1.cell_m))

            self._fog_integrate_np = NodePath("fog_integrate")
            self._fog_integrate_np.set_shader(Shader.make_compute(
                Shader.SL_GLSL, glsl.FOG_INTEGRATE_COMPUTE))
            fi = self._fog_integrate_np
            fi.set_shader_input("u_froxels", self.fog_scatter_tex)
            fi.set_shader_input("u_integrated", self.fog_integrated_tex)
            fi.set_shader_input("u_froxel_dim", LVecBase3i(w, h, z))
            fi.set_shader_input("u_fog_near", self._fog_near)
            fi.set_shader_input("u_fog_far", self._fog_far)

        # --- dirty tracking ----------------------------------------------
        # Chunk coords loaded/edited since the last reassembly; each cascade
        # only reassembles when one of them actually intersects its window
        # (streaming-frontier chunks are far outside cascade 0's 48 m box).
        self._pending_coords: set[tuple[int, int, int]] = set()
        # Brush-edited chunk coords (explosions, digging) — handled SAME-FRAME
        # by a synchronous reassembly so the crater lights immediately instead
        # of flashing black for the 1-2 frames an async reassembly lags.
        self._edited_coords: set[tuple[int, int, int]] = set()
        self._force_all_dirty = True         # first frame: build everything
        self._load_dirty_timer = 0.0
        self._last_sun: tuple | None = None  # (sun_dir, sun_rad, moon, sky)
        bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)
        bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)

        _log.info(
            "GPU lighting: cascade0 %d^3 @ %.2f m, cascade1 %d^3 @ %.2f m, "
            "cascade2 %d^3 @ %.2f m, fog %s",
            config.light_c0_cells, config.light_c0_cell_m,
            config.light_c1_cells, config.light_c1_cell_m,
            config.light_c2_cells, config.light_c2_cell_m,
            "x".join(map(str, self._fog_dim)) if self.fog_enabled else "off")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        """Brush edit → reassemble the affected cascades immediately."""
        coords = event.chunk_coords
        if isinstance(coords, tuple) and len(coords) == 3 \
                and isinstance(coords[0], int):
            self._pending_coords.add(coords)
            self._edited_coords.add(coords)
        else:
            self._pending_coords.update(coords)
            self._edited_coords.update(coords)
        self._load_dirty_timer = _LOAD_REASSEMBLE_INTERVAL_S  # no batching

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        """Chunk streamed in → reassemble soon (batched while streaming)."""
        self._pending_coords.add(event.coord)

    def _coords_hit_window(self, window: VolumeWindow) -> bool:
        """True when any pending chunk overlaps the window's world box."""
        return self._any_coord_hits(self._pending_coords, window)

    def _any_coord_hits(self, coords, window: VolumeWindow) -> bool:
        """True when any coord in ``coords`` overlaps the window's world box."""
        if window.origin_cell is None:
            return True
        chunk_m = self._config.chunk_meters
        lo = window.world_origin_m
        size = window.size_m
        for c in coords:
            if all(c[i] * chunk_m < lo[i] + size
                   and (c[i] + 1) * chunk_m > lo[i] for i in range(3)):
                return True
        return False

    # ------------------------------------------------------------------
    # Per-frame driver
    # ------------------------------------------------------------------

    def update(self, camera_pos, sky_state: "SkyState | None",
               dt: float) -> None:
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

        # 0. Auto-exposure (eye adaptation): meter the light reaching the
        #    camera (sky openness through the voxel field + nearby dynamic
        #    lights) and smooth toward the target — slow when entering the
        #    dark (caves, nightfall), fast when stepping into bright light.
        mult = self.exposure_meter.update(
            camera_pos, sky_state, self._provider.chunks, (packed, count), dt)
        base = float(self._config.light_exposure)
        self.exposure = base * mult
        self.exposure_sky = base * (mult ** 0.35)

        # 1. Window follow + geometry reassembly — the heavy CPU gather + pack
        #    runs OFF the main thread (CascadeAssemblyWorker).  The main thread
        #    only snapshots which chunks to read, uploads finished volumes, and
        #    commits the window origin once its volume is on the GPU.  A cascade
        #    reassembles when its window moved, or when a pending loaded/edited
        #    chunk actually intersects its box (the streaming frontier never
        #    touches the 48 m cascade-0 box, so it stays untouched while the
        #    world fills).
        if self._force_all_dirty:
            # Boot / first frame: assemble synchronously so the world is lit on
            # frame 1 (no async latency at startup; matches prior behaviour).
            for casc in self.cascades:
                casc.window.recenter(camera_pos)
                self._assemble_and_upload_sync(casc)
            self._pending_coords.clear()
            self._load_dirty_timer = 0.0
            self._force_all_dirty = False
        else:
            self._apply_edits_sync()
            self._schedule_assembly(camera_pos)
            self._drain_assembly_results()

        # 2. Celestial / sky change detection (affects every cascade).
        if self._last_sun is None or any(
                _changed(a, b) for a, b in zip(sun, self._last_sun)):
            for casc in self.cascades:
                casc.needs_inject = True
            self._last_sun = sun

        # 3. Dynamic lights / occluder change detection (every cascade).
        if self.lights.version != self._lights_version_seen:
            for casc in self.cascades:
                casc.needs_inject = True
            self._lights_version_seen = self.lights.version
        if self.occluders.version != self._occluders_version_seen:
            for casc in self.cascades:
                casc.needs_inject = True
            self._occluders_version_seen = self.occluders.version
            self._box_uniforms = None          # repack the uniform lists

        gsg = self._base.win.get_gsg()
        engine = self._base.graphicsEngine

        if self._box_uniforms is None:
            mins, maxs, n_boxes = self.occluders.pack()
            self._box_uniforms = (
                [LVecBase4f(mins[i, 0], mins[i, 1], mins[i, 2], 0.0)
                 for i in range(mins.shape[0])],
                [LVecBase4f(maxs[i, 0], maxs[i, 1], maxs[i, 2], 0.0)
                 for i in range(maxs.shape[0])],
                int(n_boxes),
            )
        box_min, box_max, n_boxes = self._box_uniforms

        # 4. Injection — only for cascades whose light or geometry changed.
        if any(c.needs_inject for c in self.cascades):
            pos_r = [LVecBase4f(*packed[i, 0:4]) for i in range(glsl.MAX_LIGHTS)]
            col_t = [LVecBase4f(*packed[i, 4:8]) for i in range(glsl.MAX_LIGHTS)]
            ext = [LVecBase4f(*packed[i, 8:12]) for i in range(glsl.MAX_LIGHTS)]
            sun_dir, sun_rad, moon_dir, moon_rad, sky_amb = sun
            for casc in self.cascades:
                if not casc.needs_inject:
                    continue
                casc.needs_inject = False
                n = casc.inject_np
                n.set_shader_input("u_sun_dir", LVecBase3f(*sun_dir))
                n.set_shader_input("u_sun_radiance", LVecBase3f(*sun_rad))
                n.set_shader_input("u_moon_dir", LVecBase3f(*moon_dir))
                n.set_shader_input("u_moon_radiance", LVecBase3f(*moon_rad))
                n.set_shader_input("u_sky_ambient", LVecBase3f(*sky_amb))
                n.set_shader_input(
                    "u_bounce", float(self._config.light_bounce_strength))
                n.set_shader_input("u_num_lights", int(count))
                n.set_shader_input("u_light_pos_r", pos_r)
                n.set_shader_input("u_light_col_t", col_t)
                n.set_shader_input("u_light_ext", ext)
                n.set_shader_input("u_num_boxes", n_boxes)
                n.set_shader_input("u_box_min", box_min)
                n.set_shader_input("u_box_max", box_max)
                n.set_shader_input("u_origin_m", LVecBase3f(*casc.origin_m()))
                n.set_shader_input("u_cell_m", float(casc.cell_m))
                groups = (_groups(casc.cells, 4),) * 3
                engine.dispatch_compute(
                    groups, n.get_attrib(ShaderAttrib), gsg)

        # 5. Propagation — every frame; light visibly flows toward steady state.
        for casc in self.cascades:
            groups = (_groups(casc.cells, 4),) * 3
            for _ in range(max(1, self._config.light_prop_iters)):
                node = casc.prop_np[casc.ping]      # ping → pong
                engine.dispatch_compute(
                    groups, node.get_attrib(ShaderAttrib), gsg)
                casc.ping ^= 1

        # 6. Froxel fog.
        if self.fog_enabled:
            self._dispatch_fog(camera_pos, sun, sky_state, engine, gsg)

    # ------------------------------------------------------------------
    # Cascade volume assembly (off-thread; see assembly_worker.py)
    # ------------------------------------------------------------------

    def _assemble_and_upload_sync(self, casc: "_Cascade") -> None:
        """
        Gather + upload one cascade volume inline on the main thread.

        Used only for the boot/first frame so the world is lit immediately;
        steady-state reassembly goes through the worker.
        """
        vol = assemble_geometry(
            casc.window, self._provider.chunks, self._palette,
            chunk_size=self._config.chunk_size,
            voxel_size=self._config.voxel_size)
        _upload_volume(casc.geom, vol.albedo_occ)
        _upload_volume(casc.emis, vol.emission)
        casc.needs_inject = True

    def _apply_edits_sync(self) -> None:
        """
        Refresh brush-edited cascades' geometry SAME-FRAME (kills the black flash).

        Terrain edits (explosions, digging) must light immediately.  The
        steady-state async reassembly lags 1-2 frames; until it lands the stale
        occupancy still marks the new crater as solid (so its sun visibility is
        shadowed and its GI cell unlit) and it renders black, then pops to lit
        once the volume catches up — the "black then lit" artefact.

        Edits are discrete events (not per-frame like flying), so a synchronous
        gather of the few hit cascades is affordable.  A cascade window does not
        move on an edit, so this re-slices the live (already-edited) materials at
        the committed origin and forces re-injection this frame.  Cascades with
        an async job already in flight are skipped here and left to the normal
        ``_pending_coords`` path (rare for the camera-local cascade 0 during a
        stationary edit); the eager uploads here are otherwise redundant with —
        not conflicting with — that path, which keeps the committed-origin
        bookkeeping untouched.
        """
        if not self._edited_coords:
            return
        for casc in self.cascades:
            # Only the near/mid cascades refresh synchronously: their crater is
            # what the player is looking at, and their full-window gather is
            # cheap (cascade 0 ~27, cascade 1 ~1.7k chunk coords).  The coarse
            # FAR cascade (index 2, ~33k coords over 512 m) would add a one-frame
            # hitch on every edit for a relight that lags invisibly at 96 m+, so
            # it stays on the async ``_pending_coords`` path.
            if casc.index >= 2:
                continue
            if casc._assembly_inflight:
                continue
            if casc.window.origin_cell is None:
                continue
            if not self._any_coord_hits(self._edited_coords, casc.window):
                continue
            self._assemble_and_upload_sync(casc)   # sets needs_inject = True
        self._edited_coords.clear()

    def _schedule_assembly(self, camera_pos) -> None:
        """
        Submit reassembly jobs for cascades that moved or whose terrain changed.

        Non-mutating w.r.t. the committed window origin: ``needs_recenter`` and
        ``_coords_hit_window`` test against the *committed* origin, and a job is
        submitted for the new (snapped) origin without advancing it — the origin
        commits only when the matching volume lands (``_drain_assembly_results``).
        At most one job per cascade is in flight at a time.
        """
        batch_ready = bool(self._pending_coords) and \
            self._load_dirty_timer >= _LOAD_REASSEMBLE_INTERVAL_S
        deferred = False   # a cascade still owes pending edits but is busy
        for casc in self.cascades:
            moved = casc.window.needs_recenter(camera_pos)
            hit = batch_ready and self._coords_hit_window(casc.window)
            if casc._assembly_inflight:
                if hit:
                    deferred = True
                continue
            if not (moved or hit):
                continue
            # 'moved' → assemble for the new snapped origin; 'hit' (terrain
            # changed inside the window, origin unchanged) → re-assemble the
            # committed origin.  Either way the snapshot reads live materials,
            # so a submit also satisfies any pending 'hit'.
            origin = (casc.window._desired_origin(camera_pos)
                      if moved else casc.window.origin_cell)
            self._submit_assembly(casc, origin)
        # Clear pending edits only once every cascade they touch has an
        # up-to-date (just-submitted or current) volume — otherwise a busy
        # cascade would silently miss the edit.
        if batch_ready and not deferred:
            self._pending_coords.clear()
            self._load_dirty_timer = 0.0

    def _submit_assembly(self, casc: "_Cascade", origin_cell) -> None:
        """Snapshot the chunks a reassembly will read and enqueue the job."""
        coords = window_chunk_span(
            origin_cell, casc.cells, casc.cell_m,
            int(self._config.chunk_size), float(self._config.voxel_size))
        live = self._provider.chunks
        # Snapshot *references* to the material arrays (not copies): cheap, and
        # safe against streaming (dict membership changes don't affect captured
        # arrays).  A concurrent in-place brush edit of a captured array is the
        # only race; it self-corrects on the next reassembly.
        materials = {c: live[c].materials for c in coords if c in live}
        self._assembly_seq += 1
        job = AssemblyJob(
            cascade_index=casc.index, origin_cell=tuple(origin_cell),
            cells=casc.cells, cell_m=casc.cell_m,
            chunk_size=int(self._config.chunk_size),
            voxel_size=float(self._config.voxel_size),
            materials=materials, palette=self._palette,
            seq=self._assembly_seq)
        if self._threaded:
            casc._assembly_inflight = True
            casc._pending_seq = self._assembly_seq
            self._assembly_worker.submit(job)
        else:
            # Inline (tooling/tests): assemble + commit immediately.
            self._commit_assembly_result(assemble_packed(job))

    def _drain_assembly_results(self) -> None:
        """Upload finished volumes and commit their window origins (main thread)."""
        if self._assembly_worker is None:
            return
        for res in self._assembly_worker.drain_results():
            self._commit_assembly_result(res)

    def _commit_assembly_result(self, res) -> None:
        """Upload one finished volume and advance its committed window origin."""
        casc = self.cascades[res.cascade_index]
        if self._threaded and res.seq != casc._pending_seq:
            casc._assembly_inflight = False
            return   # superseded (single-inflight makes this rare)
        casc._assembly_inflight = False
        if not res.albedo_bytes:
            return   # assembly failed → flag cleared, retry next frame
        casc.geom.set_ram_image(res.albedo_bytes)
        casc.emis.set_ram_image(res.emis_bytes)
        # Commit: the GPU geom texture now matches this origin, so the shader
        # origin uniforms (read from window.origin_cell) line up exactly.
        casc.window.origin_cell = tuple(res.origin_cell)
        casc.needs_inject = True

    def shutdown(self) -> None:
        """
        Stop the background assembly worker.  Call once on app exit.

        Idempotent and safe to call when running unthreaded (no-op).
        """
        if self._assembly_worker is not None:
            self._assembly_worker.stop(join=True)
            self._assembly_worker = None

    # ------------------------------------------------------------------

    def _dispatch_fog(self, camera_pos, sun, sky_state,
                      engine, gsg) -> None:
        """Fill + integrate the froxel volume for this frame's camera."""
        sun_dir, sun_rad, moon_dir, moon_rad, sky_amb = sun
        cam = self._base.camera
        quat = cam.get_quat(self._base.render)
        fwd = quat.get_forward()
        right = quat.get_right()
        up = quat.get_up()
        lens = self._base.camLens
        fov = lens.get_fov()    # degrees (h, v)
        tan_h = math.tan(math.radians(float(fov[0]) * 0.5))
        tan_v = math.tan(math.radians(float(fov[1]) * 0.5))

        density = 0.0015
        if sky_state is not None:
            density = float(sky_state.fog_density) * _FOG_DENSITY_BOOST

        c1 = self.cascades[1]
        sn = self._fog_scatter_np
        sn.set_shader_input("u_cam_pos", LVecBase3f(*[float(camera_pos[i])
                                                      for i in range(3)]))
        sn.set_shader_input("u_cam_fwd", LVecBase3f(fwd[0], fwd[1], fwd[2]))
        sn.set_shader_input("u_cam_right",
                            LVecBase3f(right[0], right[1], right[2]))
        sn.set_shader_input("u_cam_up", LVecBase3f(up[0], up[1], up[2]))
        sn.set_shader_input("u_tan_half_fov", LVecBase2f(tan_h, tan_v))
        sn.set_shader_input("u_fog_density", float(density))
        sn.set_shader_input("u_sun_dir", LVecBase3f(*sun_dir))
        sn.set_shader_input("u_sun_radiance", LVecBase3f(*sun_rad))
        sn.set_shader_input("u_moon_dir", LVecBase3f(*moon_dir))
        sn.set_shader_input("u_moon_radiance", LVecBase3f(*moon_rad))
        sn.set_shader_input("u_sky_ambient", LVecBase3f(*sky_amb))
        sn.set_shader_input("u_c1_radiance", c1.radiance_current)
        sn.set_shader_input("u_c1_origin_m", LVecBase3f(*c1.origin_m()))
        box_min, box_max, n_boxes = self._box_uniforms
        sn.set_shader_input("u_num_boxes", n_boxes)
        sn.set_shader_input("u_box_min", box_min)
        sn.set_shader_input("u_box_max", box_max)

        w, h, z = self._fog_dim
        engine.dispatch_compute(
            (_groups(w, 8), _groups(h, 8), z),
            sn.get_attrib(ShaderAttrib), gsg)
        engine.dispatch_compute(
            (_groups(w, 8), _groups(h, 8), 1),
            self._fog_integrate_np.get_attrib(ShaderAttrib), gsg)

    # ------------------------------------------------------------------
    # Surface-shader contract (consumed by world/terrain_shader.py)
    # ------------------------------------------------------------------

    def bind_surface_inputs(self, node: NodePath) -> None:
        """
        Bind the static lighting samplers/uniforms onto a render NodePath.

        Call once after applying the terrain shader; per-frame values are
        refreshed by :meth:`update_surface_inputs`.
        """
        c0, c1, c2 = self.cascades
        node.set_shader_input("u_c0_geom", c0.geom)
        node.set_shader_input("u_c0_vis", c0.vis)
        node.set_shader_input("u_c0_cells", float(c0.cells))
        node.set_shader_input("u_c0_cell_m", float(c0.cell_m))
        node.set_shader_input("u_c1_geom", c1.geom)
        node.set_shader_input("u_c1_vis", c1.vis)
        node.set_shader_input("u_c1_cells", float(c1.cells))
        node.set_shader_input("u_c1_cell_m", float(c1.cell_m))
        node.set_shader_input("u_c2_geom", c2.geom)
        node.set_shader_input("u_c2_vis", c2.vis)
        node.set_shader_input("u_c2_cells", float(c2.cells))
        node.set_shader_input("u_c2_cell_m", float(c2.cell_m))
        node.set_shader_input("u_quant_m",
                              float(self._config.light_quant_m))
        node.set_shader_input("u_ao_strength",
                              float(self._config.light_ao_strength))
        node.set_shader_input("u_emission_scale", float(EMISSION_SCALE))
        node.set_shader_input("u_fog_near", self._fog_near)
        node.set_shader_input("u_fog_far", self._fog_far)
        node.set_shader_input("u_fog_enabled",
                              1.0 if self.fog_enabled else 0.0)
        if self.fog_enabled:
            node.set_shader_input("u_fog_integrated", self.fog_integrated_tex)
        else:
            # Bind *something* valid for the sampler.
            node.set_shader_input("u_fog_integrated", self.cascades[0].vis)
        # Radiance/origins are per-frame (ping-pong + window scroll).
        self.update_surface_inputs(node, None)

    def update_surface_inputs(self, node: NodePath,
                              sky_state: "SkyState | None") -> None:
        """
        Refresh the per-frame lighting uniforms on a render NodePath.

        Parameters
        ----------
        node : NodePath
            Same node given to :meth:`bind_surface_inputs` (terrain_root).
        sky_state : SkyState | None
            For sun/moon direction + radiance uniforms.
        """
        c0, c1, c2 = self.cascades
        node.set_shader_input("u_c0_radiance", c0.radiance_current)
        node.set_shader_input("u_c1_radiance", c1.radiance_current)
        node.set_shader_input("u_c2_radiance", c2.radiance_current)
        node.set_shader_input("u_c0_emis", c0.emis)
        # Auto-exposure: the adapted tonemap exposure changes every frame.
        node.set_shader_input("u_exposure", float(self.exposure))
        if c0.window.origin_cell is not None:
            node.set_shader_input("u_c0_origin_m", LVecBase3f(*c0.origin_m()))
        if c1.window.origin_cell is not None:
            node.set_shader_input("u_c1_origin_m", LVecBase3f(*c1.origin_m()))
        if c2.window.origin_cell is not None:
            node.set_shader_input("u_c2_origin_m", LVecBase3f(*c2.origin_m()))
        sun_dir, sun_rad, moon_dir, moon_rad, sky_amb = \
            self._sky_inputs(sky_state)
        node.set_shader_input("u_sun_dir", LVecBase3f(*sun_dir))
        node.set_shader_input("u_sun_radiance", LVecBase3f(*sun_rad))
        node.set_shader_input("u_moon_dir", LVecBase3f(*moon_dir))
        node.set_shader_input("u_moon_radiance", LVecBase3f(*moon_rad))
        node.set_shader_input("u_sky_ambient", LVecBase3f(*sky_amb))
        win = self._base.win
        node.set_shader_input(
            "u_viewport", LVecBase2f(float(win.get_x_size()),
                                     float(win.get_y_size())))
        cam_pos = self._base.camera.get_pos(self._base.render)
        node.set_shader_input(
            "u_cam_pos", LVecBase3f(cam_pos[0], cam_pos[1], cam_pos[2]))

    # ------------------------------------------------------------------

    @staticmethod
    def _sky_inputs(sky_state: "SkyState | None") -> tuple:
        """
        Extract (sun_dir, sun_radiance, moon_dir, moon_radiance, sky_ambient)
        from a SkyState, with graceful fallbacks for older SkyState versions
        (radiance derived from sun_color × intensity) and for ``None``.
        """
        if sky_state is None:
            return ((0.3, 0.2, 0.93), (3.0, 2.9, 2.6),
                    (0.0, 0.0, -1.0), (0.0, 0.0, 0.0),
                    (0.35, 0.45, 0.70))
        sun_dir = tuple(float(v) for v in
                        (sky_state.sun_dir.x, sky_state.sun_dir.y,
                         sky_state.sun_dir.z))
        moon_dir = tuple(float(v) for v in
                         (sky_state.moon_dir.x, sky_state.moon_dir.y,
                          sky_state.moon_dir.z))
        sun_rad = getattr(sky_state, "sun_radiance", None)
        if sun_rad is None:
            s = float(sky_state.sun_intensity) * 3.2
            sun_rad = tuple(c * s for c in sky_state.sun_color)
        moon_rad = getattr(sky_state, "moon_radiance", None)
        if moon_rad is None:
            up = max(moon_dir[2], 0.0)
            full = 1.0 - abs(sky_state.moon_phase - 0.5) * 2.0
            moon_rad = (0.05 * up * full, 0.06 * up * full, 0.09 * up * full)
        sky_amb = getattr(sky_state, "sky_ambient", None)
        if sky_amb is None:
            d = float(sky_state.daylight)
            z = sky_state.zenith_color
            sky_amb = (0.02 + z[0] * 0.55 * d, 0.02 + z[1] * 0.6 * d,
                       0.03 + z[2] * 0.75 * d)
        return (sun_dir, tuple(map(float, sun_rad)), moon_dir,
                tuple(map(float, moon_rad)), tuple(map(float, sky_amb)))


def _groups(n: int, local: int) -> int:
    """Workgroup count covering ``n`` invocations at ``local`` per group."""
    return (n + local - 1) // local


def _changed(a, b, eps: float = 0.004) -> bool:
    """True when two float tuples differ beyond ``eps`` on any component."""
    return any(abs(float(x) - float(y)) > eps for x, y in zip(a, b))
