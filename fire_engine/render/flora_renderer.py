"""
world/flora_renderer.py — GPU-instanced flora volumes (render component).

``FloraRendererComponent`` draws every ``tag="flowers"`` :class:`ZoneVolume`
as hardware-instanced crossed-quad sprites — the grass idiom
(``world/grass_renderer.py``) generalised over a table of **flora kinds**
(one row today; the table stays so future sprite-scale flora is one row,
not one component).  The CPU holds **no per-plant state**: one shared
crossed-quad Geom per kind is drawn N times per volume via
``set_instance_count``, and each instance derives its position / rotation /
scale / sway phase / sprite-atlas variant in the vertex shader from
``gl_InstanceID`` (``world/shaders/flora.vert``; placement math mirrored
headlessly in ``zones/flora_placement.py``).

Per kind it binds a procedural sprite **atlas** (``flower_sprite``, seeded
per world) and a sway shape: flowers bend from the ground like grass.  They
sample the SAME wind texture grass uses (inherited from ``terrain_root``),
so one gust band rolls visibly through grass and meadow together; with the
wind field off they fall back to the scalar SkyState sway, scaled per kind.

Trees and bushes are NOT sprites any more: ``world/tree_renderer.py`` draws
them as real instanced 3-D meshes (billboards survive only as its far-LOD
impostors).

Lighting and fog come by scene-graph inheritance under ``App.terrain_root``
(radiance cascades + froxel fog, identical to grass).  GPU lighting backend
only; on the CPU backend the component disables itself with a log line.

Example (wired by main.py)
--------------------------
    flora_go = instantiate()
    flora_go.add_component(
        FloraRendererComponent,
        base=app, sky_system=sky_system, zone_store=zone_store,
        chunk_provider=chunk_manager, lighting_pipeline=pipeline, bus=bus)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    BoundingBox,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LPoint3,
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
from fire_engine.render import flora_shaders

# Weather → sway mapping shared with grass so the scalar fallback moves all
# vegetation in lockstep (flora scales it per kind via u_sway_gain).
from fire_engine.render.grass_renderer import (
    _GUST_FREQ_MIN,
    _GUST_FREQ_PER_WIND,
    _GUST_FREQ_RAIN,
    _SWAY_BASE_MIN_M,
    _SWAY_BASE_WIND_M,
    _SWAY_GUST_MIN_M,
    _SWAY_GUST_RAIN_M,
    _SWAY_GUST_WIND_M,
    _WIND_SPEED_MAX,
)
from fire_engine.zones import (
    bake_grass_height_field,
    flora_hash_seed,
    flora_instance_count,
)

__all__ = ["FloraRendererComponent"]

_log = get_logger("world.flora")

# Bounds margin past the scaled plant reach (sway never exceeds this).
_BOUNDS_PAD_M = 0.5


@dataclass(frozen=True)
class _FloraKind:
    """One row of the flora table — everything that differs between kinds."""

    tag: str  # ZoneVolume tag this kind renders
    texture: str  # procedural sprite-atlas def name
    variants: int  # atlas cell count
    n_quads: int  # crossed quads per instance
    aspect: float  # quad width / height (match the atlas cell aspect)
    height_cfg: str  # Config field: unscaled plant height (m)
    fade_start_cfg: str  # Config fields: distance fade window (m)
    fade_end_cfg: str
    scale_min: float  # per-instance size jitter range
    scale_span: float
    sway_gain: float  # sway amplitude vs grass (1.0 = grass-equal)
    sway_pivot: float  # normalised height where bending starts
    light_offset_m: float  # cascade sample height above the plant base


# Flowers bend like grass.  (Bushes and trees left this table for
# world/tree_renderer.py's 3-D mesh pipeline.)
_FLORA_KINDS: tuple[_FloraKind, ...] = (
    _FloraKind(
        "flowers",
        "flower_sprite",
        4,
        2,
        1.0,
        "flora_flower_height_m",
        "flora_flower_fade_start_m",
        "flora_flower_fade_end_m",
        0.7,
        0.6,
        0.8,
        0.0,
        0.5,
    ),
)


class FloraRendererComponent(Component):
    """
    Render component for GPU-instanced sprite flora (flowers).

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``terrain_root`` and ``_config``.
    sky_system : fire_engine.world.sky.SkySystem
        Read-only weather source for the scalar sway fallback uniforms.
    zone_store : fire_engine.zones.ZoneStore
        Volumes tagged ``"flowers"`` are rendered; the store's ``version``
        counter triggers a rebuild.
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
        self.base = base
        self.sky_system = sky_system
        self.zone_store = zone_store
        self.chunk_provider = chunk_provider
        self.lighting_pipeline = lighting_pipeline
        self.bus = bus

        self._root: NodePath | None = None
        self._shader: Shader | None = None
        self._kind_roots: dict[str, NodePath] = {}
        self._kind_geoms: dict[str, Geom] = {}
        self._volume_nodes: dict[tuple[str, int], NodePath] = {}
        self._dirty_fields: set[tuple[str, int]] = set()
        self._store_version_built: int = -1
        self._time_s: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the shared geoms, shader and per-kind/volume nodes (once)."""
        if self.base is None or self.zone_store is None or self.chunk_provider is None:
            _log.warning(
                "FloraRendererComponent: missing base/zone_store/chunk_provider — disabled"
            )
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "FloraRendererComponent: GPU lighting pipeline "
                'required (lighting_backend = "gpu") — disabled'
            )
            self.enabled = False
            return

        cfg = self.base._config
        self._shader = Shader.make(
            Shader.SL_GLSL, vertex=flora_shaders.FLORA_VERTEX, fragment=flora_shaders.FLORA_FRAGMENT
        )

        # Lighting (cascade/fog/celestial) inherits from ``render`` where
        # GpuLightingPipeline binds the lit-surface contract; the wind
        # shader inputs inherit from terrain_root.
        self._root = self.base.terrain_root.attach_new_node("flora_root")
        self._root.set_two_sided(True)  # crossed quads face both ways
        # Scalar-fallback sway defaults until the first late_update.
        self._root.set_shader_input("u_wind_dir", LVecBase2f(1.0, 0.0))
        self._root.set_shader_input("u_sway_base", _SWAY_BASE_MIN_M)
        self._root.set_shader_input("u_sway_gust", _SWAY_GUST_MIN_M)
        self._root.set_shader_input("u_gust_freq", _GUST_FREQ_MIN)
        # u_time_s is NOT inherited from terrain_root (grass binds its own on
        # grass_root) — flora accumulates and binds its own, in lockstep.
        self._root.set_shader_input("u_time_s", 0.0)
        # Shadow-refinement gate (lit_surface.glsl).  Bound HERE, not
        # inherited: terrain_root above us pins u_refine = 1.0 for the
        # terrain, foliage follows the graphics preset.
        self._root.set_shader_input("u_refine", 1.0 if cfg.gfx_foliage_shadow_refine else 0.0)

        for kind in _FLORA_KINDS:
            height = float(getattr(cfg, kind.height_cfg))
            self._kind_geoms[kind.tag] = _build_cross_geom(
                height, height * kind.aspect, kind.n_quads
            )
            kroot = self._root.attach_new_node(f"flora_{kind.tag}")
            kroot.set_shader_input("u_sprite", self._sprite_texture(kind))
            kroot.set_shader_input("u_plant_height_m", height)
            kroot.set_shader_input("u_fade_start_m", float(getattr(cfg, kind.fade_start_cfg)))
            kroot.set_shader_input("u_fade_end_m", float(getattr(cfg, kind.fade_end_cfg)))
            kroot.set_shader_input("u_scale_min", kind.scale_min)
            kroot.set_shader_input("u_scale_span", kind.scale_span)
            kroot.set_shader_input("u_sway_gain", kind.sway_gain)
            kroot.set_shader_input("u_sway_pivot", kind.sway_pivot)
            kroot.set_shader_input("u_variants", float(kind.variants))
            kroot.set_shader_input("u_light_offset_m", kind.light_offset_m)
            self._kind_roots[kind.tag] = kroot

        self._build_volumes()

        if self.bus is not None:
            self.bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)
            self.bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)

    def late_update(self, dt: float) -> None:
        """Sync scalar sway uniforms; rebuild/re-bake what changed."""
        if self._root is None:
            return
        self._time_s += dt

        if self.zone_store.version != self._store_version_built:
            self._build_volumes()
        elif self._dirty_fields:
            for key in tuple(self._dirty_fields):
                self._rebake_field(key)
            self._dirty_fields.clear()

        st = getattr(self.sky_system, "state", None) if self.sky_system is not None else None
        if st is not None:
            wind = float(st.wind_speed)
            rain = float(st.rain_intensity)
            wn = max(0.0, min(wind / _WIND_SPEED_MAX, 1.0))
            self._root.set_shader_input(
                "u_wind_dir", LVecBase2f(float(st.wind_dir[0]), float(st.wind_dir[1]))
            )
            self._root.set_shader_input("u_sway_base", _SWAY_BASE_MIN_M + _SWAY_BASE_WIND_M * wn)
            self._root.set_shader_input(
                "u_sway_gust", _SWAY_GUST_MIN_M + _SWAY_GUST_WIND_M * wn + _SWAY_GUST_RAIN_M * rain
            )
            self._root.set_shader_input(
                "u_gust_freq", _GUST_FREQ_MIN + _GUST_FREQ_PER_WIND * wind + _GUST_FREQ_RAIN * rain
            )
        self._root.set_shader_input("u_time_s", self._time_s)

    def on_destroy(self) -> None:
        """Detach all flora nodes and unsubscribe from the bus."""
        if self.bus is not None:
            self.bus.unsubscribe(TerrainEditedEvent, self._on_terrain_edited)
            self.bus.unsubscribe(ChunkLoadedEvent, self._on_chunk_loaded)
        if self._root is not None:
            self._root.remove_node()
            self._root = None
        self._kind_roots.clear()
        self._volume_nodes.clear()

    # ------------------------------------------------------------------
    # Event handlers (mark dirty only — work happens in late_update)
    # ------------------------------------------------------------------

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        coords = event.chunk_coords
        if isinstance(coords, tuple) and len(coords) == 3 and isinstance(coords[0], int):
            coords = (coords,)
        self._mark_dirty_for_coords(coords)

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        self._mark_dirty_for_coords((event.coord,))

    def _mark_dirty_for_coords(self, coords) -> None:
        """Queue a height-field re-bake for volumes touching these chunks."""
        if self.base is None:
            return
        chunk_m = float(self.base._config.chunk_meters)
        for kind in _FLORA_KINDS:
            for vol in self.zone_store.volumes(kind.tag):
                key = (kind.tag, vol.id)
                if key in self._dirty_fields or key not in self._volume_nodes:
                    continue
                if any(vol.intersects_chunk(c, chunk_m) for c in coords):
                    self._dirty_fields.add(key)

    # ------------------------------------------------------------------
    # Build / re-bake
    # ------------------------------------------------------------------

    def _build_volumes(self) -> None:
        """(Re)create one instanced NodePath per (kind, volume)."""
        for node in self._volume_nodes.values():
            node.remove_node()
        self._volume_nodes.clear()
        self._dirty_fields.clear()

        cfg = self.base._config
        total = 0
        for kind in _FLORA_KINDS:
            height = float(getattr(cfg, kind.height_cfg))
            for vol in self.zone_store.volumes(kind.tag):
                count = flora_instance_count(vol, cfg, kind.tag)
                if count <= 0:
                    continue
                geom_node = GeomNode(f"flora_{kind.tag}_vol_{vol.id}")
                geom_node.add_geom(self._kind_geoms[kind.tag])
                node = self._kind_roots[kind.tag].attach_new_node(geom_node)
                # Shader and instance count MUST live on the same node (the
                # grass caveat): set_instance_count creates a node-level
                # ShaderAttrib that REPLACES an inherited one.  ShaderInput
                # attribs compose, so the kind/root uniforms still arrive.
                node.set_shader(self._shader)
                node.set_instance_count(count)
                node.set_shader_input("u_bounds_min", LVecBase3f(*vol.min_corner))
                node.set_shader_input("u_bounds_max", LVecBase3f(*vol.max_corner))
                node.set_shader_input("u_hash_seed", flora_hash_seed(vol, kind.tag))
                self._upload_field(node, vol)

                # Instances are shader-positioned — give the node the
                # volume's real box plus the scaled plant reach, and stop
                # bounds recomputation (Panda3D would cull by the base
                # Geom's tiny origin bounds otherwise).
                pad = height * (kind.scale_min + kind.scale_span) + _BOUNDS_PAD_M
                geom_node.set_bounds(
                    BoundingBox(
                        LPoint3(
                            vol.min_corner[0] - pad,
                            vol.min_corner[1] - pad,
                            vol.min_corner[2] - pad,
                        ),
                        LPoint3(
                            vol.max_corner[0] + pad,
                            vol.max_corner[1] + pad,
                            vol.max_corner[2] + pad,
                        ),
                    )
                )
                geom_node.set_final(True)
                self._volume_nodes[(kind.tag, vol.id)] = node
                total += count

        self._store_version_built = self.zone_store.version
        _log.info(
            "Flora built: %d volume node(s), %d instances total", len(self._volume_nodes), total
        )

    def _rebake_field(self, key: tuple[str, int]) -> None:
        """Re-bake + re-upload one volume's height field (terrain changed)."""
        tag, vol_id = key
        vol = self.zone_store.get(vol_id)
        node = self._volume_nodes.get(key)
        if vol is None or node is None:
            return
        self._upload_field(node, vol)
        _log.debug("Flora height field re-baked for %s volume %d", tag, vol_id)

    def _upload_field(self, node: NodePath, vol) -> None:
        """Bake the volume's height field and bind it as u_height_field."""
        from fire_engine.render.texture_bridge import to_field_texture

        field = bake_grass_height_field(vol, self.chunk_provider.chunks, self.base._config)
        node.set_shader_input("u_height_field", to_field_texture(field))

    def _sprite_texture(self, kind: _FloraKind):
        """The kind's procedural sprite atlas as a Panda3D texture (nearest)."""
        from fire_engine.procedural import get as get_procedural
        from fire_engine.render.texture_bridge import to_panda_texture

        return to_panda_texture(get_procedural(kind.texture))


# ---------------------------------------------------------------------------
# Crossed-quad geometry (built once per kind, shared by its volume GeomNodes)
# ---------------------------------------------------------------------------


def _build_cross_geom(height_m: float, width_m: float, n_quads: int) -> Geom:
    """
    Build a crossed-quad Geom: ``n_quads`` quads fanned evenly around Z,
    base at the origin.

    Each quad is ``height_m`` tall and ``width_m`` wide, UV-mapped 0–1 with
    V=0 at the ground (the sprite atlas's cell-local U; the vertex shader
    remaps U into the chosen variant cell).  ``4·n_quads`` vertices — a
    fixed handful, not a per-element loop.

    Parameters
    ----------
    height_m : float
        Unscaled plant height (the kind's config height); the shader
        jitters per-instance scale.
    width_m : float
        Quad width — keep ``width_m / height_m`` equal to the atlas cell
        aspect so texels stay square.
    n_quads : int
        Crossed quads per instance (2 for flowers, 3 for bushes/trees).
    """
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("flora_cross", fmt, Geom.UH_static)
    vdata.set_num_rows(4 * n_quads)
    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")
    tris = GeomTriangles(Geom.UH_static)

    half_w = width_m * 0.5
    for k in range(n_quads):  # fixed tiny loop
        ang = k * math.pi / n_quads
        dx, dy = math.cos(ang) * half_w, math.sin(ang) * half_w
        base = k * 4
        vw.add_data3(-dx, -dy, 0.0)
        tw.add_data2(0.0, 0.0)
        vw.add_data3(dx, dy, 0.0)
        tw.add_data2(1.0, 0.0)
        vw.add_data3(dx, dy, height_m)
        tw.add_data2(1.0, 1.0)
        vw.add_data3(-dx, -dy, height_m)
        tw.add_data2(0.0, 1.0)
        tris.add_vertices(base, base + 1, base + 2)
        tris.add_vertices(base, base + 2, base + 3)

    geom = Geom(vdata)
    geom.add_primitive(tris)
    return geom
