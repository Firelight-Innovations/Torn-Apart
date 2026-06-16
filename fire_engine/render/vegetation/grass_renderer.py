"""
world/grass_renderer.py — GPU-instanced grass volumes (render component).

``GrassRendererComponent`` draws every ``tag="grass"`` :class:`ZoneVolume`
as hardware-instanced tuft geometry.  The CPU holds **no per-blade state**:
one shared 3-crossed-quad Geom is drawn N times per volume via
``set_instance_count``, and each instance derives its position/rotation/
scale/sway phase in the vertex shader from ``gl_InstanceID``
(``world/grass_shaders.py``; placement math mirrored headlessly in
``zones/grass_placement.py``).

What the CPU does do, exactly once per volume (and again only when terrain
under the volume changes):

- bake the tiny per-volume **height field** texture
  (``zones.bake_grass_height_field``) so blades stand on the actual surface
  and vanish where it was carved away,
- compute the instance count and hash seed,
- set an explicit ``BoundingBox`` + ``set_final`` on the instanced node —
  Panda3D culls instanced geometry by the single base Geom's bounds
  otherwise, which would cull every off-origin blade.

Lighting and fog come **for free by scene-graph inheritance**: the node
lives under ``App.terrain_root``, where ``GpuLightingPipeline`` binds and
refreshes the radiance-cascade / froxel-fog shader inputs every frame — the
grass shader simply declares the same uniform names.  GPU lighting backend
only; on the legacy CPU backend the component disables itself with a log
line.

Per frame (``late_update``) it syncs the weather sway uniforms from
``SkyState`` (storms lean and shake the grass harder) and re-bakes any
volume whose terrain was edited (``TerrainEditedEvent``/``ChunkLoadedEvent``
mark volumes dirty — state-change events only, never per-frame plumbing).

Wind: the ``SkyState`` scalar sway uniforms written here (``u_wind_dir``,
``u_sway_base``, ``u_sway_gust``, ``u_gust_freq``) are now the documented
**fallback** path.  When ``WindSystemComponent`` (``world/wind_renderer.py``)
is live it binds the spatially-varying wind field on ``terrain_root`` and sets
``u_wind_enabled = 1.0``; ``grass.vert`` then samples ``u_wind_tex`` per blade
(advecting gust bands travel across the field) and ignores these scalars.  They
still drive the grass on the CPU lighting backend / when no wind component runs
(``u_wind_enabled = 0.0``).

Example (wired by main.py)
--------------------------
    grass_go = instantiate()
    grass_go.add_component(
        GrassRendererComponent,
        base=app, sky_system=sky_system, zone_store=zone_store,
        chunk_provider=chunk_manager, lighting_pipeline=pipeline, bus=bus)
"""

from __future__ import annotations

import math
from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LVecBase2f,
    LVecBase3f,
    NodePath,
    Shader,
)

from fire_engine.core import (
    ChunkLoadedEvent,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.render.component import Component
from fire_engine.render.vegetation import grass_shaders
from fire_engine.render.vegetation._impl.zone_renderer import (
    _GUST_FREQ_MIN,
    _SWAY_BASE_MIN_M,
    _SWAY_GUST_MIN_M,
    init_zone_renderer,
    set_volume_bounds,
    subscribe_terrain_events,
    sync_sway_uniforms,
    unsubscribe_terrain_events,
)
from fire_engine.render.vegetation._impl.zone_renderer import (
    on_chunk_loaded as _on_chunk_loaded_impl,
)
from fire_engine.render.vegetation._impl.zone_renderer import (
    on_terrain_edited as _on_terrain_edited_impl,
)
from fire_engine.zones import (
    bake_grass_height_field,
    grass_hash_seed,
    grass_instance_count,
)

__all__ = ["GrassRendererComponent"]

_log = get_logger("world.grass")

# Crossed-quad tuft geometry: 3 quads fanned every 60°, each this fraction of
# the blade height wide (the silhouette texture fills the quad).
_QUAD_WIDTH_RATIO = 0.8
# Per-blade scale jitter tops out at 1.3× (zones/grass_placement.py); bounds
# add this margin so swaying tips never poke outside the culling box.
_BOUNDS_PAD_M = 0.5


class GrassRendererComponent(Component):
    """
    Render component for GPU-instanced grass volumes.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``terrain_root`` and ``_config``.
    sky_system : fire_engine.world.sky.SkySystem
        Read-only weather source (``state.wind_dir/wind_speed/
        rain_intensity`` drive the sway uniforms).
    zone_store : fire_engine.zones.ZoneStore
        Volumes tagged ``"grass"`` are rendered; the store's ``version``
        counter triggers a full rebuild when volumes change.
    chunk_provider : object
        Anything with a ``.chunks`` dict (``ChunkManager``) — height-field
        bakes read voxel materials from it.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` (CPU backend)
        disables the component.
    bus : EventBus | None
        Subscribes to ``TerrainEditedEvent`` / ``ChunkLoadedEvent`` for
        height-field re-bakes.

    Units: meters, seconds, radians.  World-space Z-up.
    """

    # Class-level annotations for attributes set by init_zone_renderer and
    # accessed by zone_renderer helpers (mypy --strict requires these).
    base: Any
    sky_system: Any
    zone_store: Any
    chunk_provider: Any
    lighting_pipeline: Any
    bus: Any
    _root: NodePath | None
    _shader: Shader | None
    _volume_nodes: dict[int, NodePath]
    _dirty_fields: set[int]
    _store_version_built: int
    _time_s: float

    def __init__(
        self,
        base: Any = None,
        sky_system: Any = None,
        zone_store: Any = None,
        chunk_provider: Any = None,
        lighting_pipeline: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__()
        init_zone_renderer(
            self, base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus
        )
        self._root = None
        self._shader = None
        self._tuft_geom: Geom | None = None
        self._volume_nodes = {}
        self._dirty_fields = set()
        self._store_version_built = -1
        self._time_s = 0.0
        self._blade_h: float = 0.6

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the shared tuft Geom, shader and per-volume nodes (once)."""
        if self.base is None or self.zone_store is None or self.chunk_provider is None:
            _log.warning(
                "GrassRendererComponent: missing base/zone_store/chunk_provider — disabled"
            )
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "GrassRendererComponent: GPU lighting pipeline "
                'required (lighting_backend = "gpu") — disabled'
            )
            self.enabled = False
            return

        cfg = self.base._config
        self._blade_h = float(cfg.grass_blade_height_m)
        self._tuft_geom = _build_tuft_geom(self._blade_h)

        # The lit-surface lighting contract (cascade/fog/celestial uniforms)
        # is bound + refreshed on ``render`` by GpuLightingPipeline and
        # inherited here; terrain_root parenting groups the world geometry.
        self._root = self.base.terrain_root.attach_new_node("grass_root")
        self._shader = Shader.make(
            Shader.SL_GLSL, vertex=grass_shaders.GRASS_VERTEX, fragment=grass_shaders.GRASS_FRAGMENT
        )
        self._root.set_two_sided(True)  # crossed quads face both ways
        self._root.set_shader_input("u_tuft", self._tuft_texture())
        self._root.set_shader_input("u_blade_height_m", self._blade_h)
        self._root.set_shader_input("u_fade_start_m", float(cfg.grass_fade_start_m))
        self._root.set_shader_input("u_fade_end_m", float(cfg.grass_fade_end_m))
        # Shadow-refinement gate (lit_surface.glsl).  Bound HERE, not
        # inherited: terrain_root above us pins u_refine = 1.0 for the
        # terrain, foliage follows the graphics preset.
        self._root.set_shader_input("u_refine", 1.0 if cfg.gfx_foliage_shadow_refine else 0.0)
        # Wind defaults until the first late_update sees a SkyState.
        self._root.set_shader_input("u_wind_dir", LVecBase2f(1.0, 0.0))
        self._root.set_shader_input("u_sway_base", _SWAY_BASE_MIN_M)
        self._root.set_shader_input("u_sway_gust", _SWAY_GUST_MIN_M)
        self._root.set_shader_input("u_gust_freq", _GUST_FREQ_MIN)
        self._root.set_shader_input("u_time_s", 0.0)

        self._build_volumes()
        subscribe_terrain_events(self)

    def late_update(self, dt: float) -> None:
        """Sync weather sway uniforms; rebuild/re-bake what changed."""
        if self._root is None:
            return

        if self.zone_store.version != self._store_version_built:
            self._build_volumes()
        elif self._dirty_fields:
            for vol_id in tuple(self._dirty_fields):
                self._rebake_field(vol_id)
            self._dirty_fields.clear()

        sync_sway_uniforms(self, dt)

    def on_destroy(self) -> None:
        """Detach all grass nodes and unsubscribe from the bus."""
        unsubscribe_terrain_events(self)
        if self._root is not None:
            self._root.remove_node()
            self._root = None
        self._volume_nodes.clear()

    # ------------------------------------------------------------------
    # Event handlers (mark dirty only — work happens in late_update)
    # ------------------------------------------------------------------

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        _on_terrain_edited_impl(self, event)

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        _on_chunk_loaded_impl(self, event)

    def _mark_dirty_for_coords(self, coords: Any) -> None:
        """Queue a height-field re-bake for volumes touching these chunks."""
        if self.base is None:
            return
        chunk_m = float(self.base._config.chunk_meters)
        for vol in self.zone_store.volumes("grass"):
            if vol.id in self._dirty_fields or vol.id not in self._volume_nodes:
                continue
            if any(vol.intersects_chunk(c, chunk_m) for c in coords):
                self._dirty_fields.add(vol.id)

    # ------------------------------------------------------------------
    # Build / re-bake
    # ------------------------------------------------------------------

    def _build_volumes(self) -> None:
        """(Re)create one instanced NodePath per grass volume."""
        for node in self._volume_nodes.values():
            node.remove_node()
        self._volume_nodes.clear()
        self._dirty_fields.clear()

        cfg = self.base._config
        total = 0
        for vol in self.zone_store.volumes("grass"):
            count = grass_instance_count(vol, cfg)
            if count <= 0:
                continue
            geom_node = GeomNode(f"grass_vol_{vol.id}")
            geom_node.add_geom(self._tuft_geom)
            assert self._root is not None
            node = self._root.attach_new_node(geom_node)
            # Shader and instance count MUST live on the same node:
            # set_instance_count creates this node's own ShaderAttrib, and a
            # node-level ShaderAttrib REPLACES (not composes with) an
            # inherited one — an ancestor-set shader would be dropped here and
            # the instances would render fixed-function at the origin.
            # ShaderInput attribs DO compose, so shared uniforms stay on
            # grass_root / terrain_root.
            node.set_shader(self._shader)
            node.set_instance_count(count)
            node.set_shader_input("u_bounds_min", LVecBase3f(*vol.min_corner))
            node.set_shader_input("u_bounds_max", LVecBase3f(*vol.max_corner))
            node.set_shader_input("u_hash_seed", grass_hash_seed(vol))
            self._upload_field(node, vol)

            # Instances are positioned in the shader — Panda3D would cull by
            # the base Geom's tiny origin bounds.  Give the node the volume's
            # real box (plus blade reach) and stop bounds recomputation.
            pad = self._blade_h * 1.3 + _BOUNDS_PAD_M
            set_volume_bounds(geom_node, vol, pad)
            self._volume_nodes[vol.id] = node
            total += count

        self._store_version_built = self.zone_store.version
        _log.info("Grass built: %d volume(s), %d instances total", len(self._volume_nodes), total)

    def _rebake_field(self, vol_id: int) -> None:
        """Re-bake + re-upload one volume's height field (terrain changed)."""
        vol = self.zone_store.get(vol_id)
        node = self._volume_nodes.get(vol_id)
        if vol is None or node is None:
            return
        self._upload_field(node, vol)
        _log.debug("Grass height field re-baked for volume %d", vol_id)

    def _upload_field(self, node: NodePath, vol: Any) -> None:
        """Bake the volume's height field and bind it as u_height_field."""
        from fire_engine.render.bridges.texture_bridge import to_field_texture

        field = bake_grass_height_field(vol, self.chunk_provider.chunks, self.base._config)
        node.set_shader_input("u_height_field", to_field_texture(field))

    def _tuft_texture(self) -> Any:
        """The pixel-art ``grass_tuft`` silhouette as a Panda3D texture."""
        from fire_engine.procedural import get as get_procedural
        from fire_engine.render.bridges.texture_bridge import to_panda_texture

        return to_panda_texture(get_procedural("grass_tuft"))


# ---------------------------------------------------------------------------
# Tuft geometry (built once, shared by every volume's GeomNode)
# ---------------------------------------------------------------------------


def _build_tuft_geom(blade_height_m: float) -> Geom:
    """
    Build the shared tuft Geom: 3 quads crossed at 60°, base at the origin.

    Each quad is ``blade_height_m`` tall and ``_QUAD_WIDTH_RATIO ×`` that
    wide, UV-mapped 0–1 (V=0 at the ground).  12 vertices / 6 triangles —
    a fixed handful, not a per-element loop.

    Parameters
    ----------
    blade_height_m : float
        Unscaled tuft height (``config.grass_blade_height_m``); the shader
        jitters per-instance scale 0.7–1.3×.
    """
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("grass_tuft", fmt, Geom.UH_static)
    vdata.set_num_rows(12)
    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")
    tris = GeomTriangles(Geom.UH_static)

    half_w = blade_height_m * _QUAD_WIDTH_RATIO * 0.5
    for k in range(3):  # 3 quads — fixed tiny loop
        ang = k * math.pi / 3.0
        dx, dy = math.cos(ang) * half_w, math.sin(ang) * half_w
        base = k * 4
        vw.add_data3(-dx, -dy, 0.0)
        tw.add_data2(0.0, 0.0)
        vw.add_data3(dx, dy, 0.0)
        tw.add_data2(1.0, 0.0)
        vw.add_data3(dx, dy, blade_height_m)
        tw.add_data2(1.0, 1.0)
        vw.add_data3(-dx, -dy, blade_height_m)
        tw.add_data2(0.0, 1.0)
        tris.add_vertices(base, base + 1, base + 2)
        tris.add_vertices(base, base + 2, base + 3)

    geom = Geom(vdata)
    geom.add_primitive(tris)
    return geom
