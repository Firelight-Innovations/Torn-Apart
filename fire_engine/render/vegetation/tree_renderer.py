"""
world/tree_renderer.py — instanced 3-D trees and bushes (render component).

``TreeRendererComponent`` draws every ``tag="trees"`` and ``tag="bushes"``
:class:`ZoneVolume` as REAL geometry: per-species variant-mesh pools
(``procedural/flora/species_def.py``) hardware-instanced over CPU-baked
placements (``zones/tree_placement.py``).  This replaces the billboard tree
and bush sprites — billboards survive only as the far-distance LOD
**impostor** stage this component also draws.

Draw structure (per volume)
---------------------------
- One ``GeomNode`` per ``(species, mesh variant)`` with at least one
  instance: the variant's :class:`TreeMesh` uploaded once via
  ``geometry_bridge.to_geom`` (cached per species+variant) and drawn
  ``set_instance_count(n)`` times.  Each draw binds its OWN RGBA32F data
  texture (``texture_bridge.to_data_texture_f32``) holding exactly that
  subset's transforms — the shader indexes it with ``gl_InstanceID``, so
  the rows must be the draw's instances and nothing else.
- One impostor ``GeomNode`` per ``species``: crossed quads sized
  ``impostor_width_m × impostor_height_m`` (the pool-common raster scale —
  the quad overlays every variant exactly), instanced over ALL the species'
  rows with the opposite fade window.  Its fragment stage is
  ``flora_shaders.FLORA_FRAGMENT`` verbatim.

The mesh window comes from config (``tree_mesh_fade_start_m`` …): meshes
shrink to zero across it while impostors grow in, then impostors fade out
across the impostor window.  Both stages read the SAME data texture, so the
crossfade can never desynchronise.

Wind: trunks pin and canopies sway via the per-vertex sway weight baked
into ``color.a`` (mesh path) / the ``u_sway_pivot`` height ramp (impostor
path); both sample the grass wind field inherited from ``terrain_root``,
with the scalar SkyState fallback synced here each frame.  ``u_time_s`` is
NOT inherited — this component accumulates and binds its own.

Lighting and fog come by scene-graph inheritance under ``App.terrain_root``
(radiance cascades + froxel fog).  GPU lighting backend only; on the CPU
backend the component disables itself with a log line.

A ``"trees"`` volume is shared infrastructure: this component draws the
trees, and the wind system's ``LeafLitterComponent`` independently scatters
gust-driven leaves over the same volume.

Example (wired by main.py)
--------------------------
    tree_go = instantiate()
    tree_go.add_component(
        TreeRendererComponent,
        base=app, sky_system=sky_system, zone_store=zone_store,
        chunk_provider=chunk_manager, lighting_pipeline=pipeline, bus=bus)

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Panda3D imports allowed in render/ per ARCHITECTURE §3.
from panda3d.core import (
    Geom,
    LVecBase2f,
    NodePath,
    Shader,
    Texture,
)

from fire_engine.core import (
    ChunkLoadedEvent,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.render.component import Component
from fire_engine.render.vegetation import flora_shaders, tree_shaders
from fire_engine.render.vegetation._impl.tree_build import (
    build_volume as _build_volume,
)
from fire_engine.render.vegetation._impl.tree_build import (
    build_volumes as _build_volumes_fn,
)
from fire_engine.render.vegetation._impl.tree_build import (
    rebuild_volume as _rebuild_volume,
)
from fire_engine.render.vegetation._impl.tree_occluders import push_occluders as _push_occluders
from fire_engine.render.vegetation._impl.zone_renderer import (
    _GUST_FREQ_MIN,
    _SWAY_BASE_MIN_M,
    _SWAY_GUST_MIN_M,
    init_zone_renderer,
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

__all__ = ["TreeRendererComponent"]

_log = get_logger("world.trees")

# Bounds margin past the scaled tree reach (sway never exceeds this).
_BOUNDS_PAD_M = 1.0

# Cascade sample height for the impostor stage (flora.frag's
# u_light_offset_m), as a fraction of the species' max height — roughly
# the canopy centre.
_LIGHT_OFFSET_FRAC = 0.45


@dataclass(frozen=True)
class _TreeKind:
    """One row of the kind table — everything that differs trees vs bushes."""

    tag: str  # ZoneVolume tag this kind renders
    prefix: str  # Config field prefix ("tree" / "bush")
    sway_gain: float  # canopy sway amplitude (meters of lean at weight 1)
    sway_pivot: float  # impostor: normalised height where canopy sway starts


# A 6 m oak tip leaning like a grass blade reads as jelly — gains stay
# small; bushes sway from lower down (smaller, springier plants).
_TREE_KINDS: tuple[_TreeKind, ...] = (
    _TreeKind("trees", "tree", 0.5, 0.45),
    _TreeKind("bushes", "bush", 0.4, 0.30),
)


class TreeRendererComponent(Component):
    """
    Render component for instanced 3-D trees and bushes.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``terrain_root`` and ``_config``.
    sky_system : fire_engine.world.sky.SkySystem
        Read-only weather source for the scalar sway fallback uniforms.
    zone_store : fire_engine.zones.ZoneStore
        Volumes tagged ``"trees"`` / ``"bushes"`` are rendered; the store's
        ``version`` counter triggers a full rebuild.
    chunk_provider : object
        Anything with a ``.chunks`` dict (``ChunkManager``) — placement Z
        comes from the height-field bake over its voxels.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` (CPU backend)
        disables the component.
    bus : EventBus | None
        Subscribes to ``TerrainEditedEvent`` / ``ChunkLoadedEvent``: edits
        under a volume re-bake its placements (trees keep their feet on
        the ground).

    Units: meters, seconds, radians.  World-space Z-up.
    """

    # Class-level annotations for attributes set by init_zone_renderer and
    # accessed by _impl helpers (mypy --strict requires these).
    base: Any
    sky_system: Any
    zone_store: Any
    chunk_provider: Any
    lighting_pipeline: Any
    bus: Any
    _root: NodePath | None
    _mesh_shader: Shader | None
    _impostor_shader: Shader | None
    _kind_roots: dict[str, NodePath]
    _volume_nodes: dict[tuple[str, int], list[NodePath]]
    _dirty_volumes: set[tuple[str, int]]
    _store_version_built: int
    _time_s: float
    _mesh_geoms: dict[tuple[str, int], Geom]
    _impostor_geoms: dict[str, Geom]
    _atlas_tex: dict[str, Texture]
    _impostor_tex: dict[str, Texture]
    _volume_occluders: dict[tuple[str, int], Any]
    _species_occ_rgb: dict[str, tuple[Any, Any]]
    _species_sigma: dict[str, float]

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
        self._mesh_shader = None
        self._impostor_shader = None
        self._kind_roots = {}
        # Per-(kind, volume.id): every NodePath built for that volume —
        # removed together on rebuild/re-bake.
        self._volume_nodes = {}
        self._dirty_volumes = set()
        self._store_version_built = -1
        self._time_s = 0.0
        # Caches keyed by species (+ variant) — uploads happen once even
        # when many volumes share a species.
        self._mesh_geoms = {}
        self._impostor_geoms = {}
        self._atlas_tex = {}
        self._impostor_tex = {}
        # Per-(kind, volume.id) static-occluder sets (lighting/occluders.py)
        # — merged + pushed to the pipeline so the light cascades see trees.
        self._volume_occluders = {}
        # Per-species mean (bark_rgb, leaf_rgb) splat colours from the atlas.
        self._species_occ_rgb = {}
        # Per-species leaf-derived canopy extinction (per meter, scale 1.0).
        self._species_sigma = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Compile shaders, build the root and every volume's draws (once)."""
        if self.base is None or self.zone_store is None or self.chunk_provider is None:
            _log.warning("TreeRendererComponent: missing base/zone_store/chunk_provider — disabled")
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "TreeRendererComponent: GPU lighting pipeline "
                'required (lighting_backend = "gpu") — disabled'
            )
            self.enabled = False
            return

        self._mesh_shader = Shader.make(
            Shader.SL_GLSL, vertex=tree_shaders.TREE_VERTEX, fragment=tree_shaders.TREE_FRAGMENT
        )
        self._impostor_shader = Shader.make(
            Shader.SL_GLSL,
            vertex=tree_shaders.TREE_IMPOSTOR_VERTEX,
            fragment=flora_shaders.FLORA_FRAGMENT,
        )

        # Lighting (cascade/fog/celestial) inherits from ``render`` where
        # GpuLightingPipeline binds the lit-surface contract; the wind
        # shader inputs inherit from terrain_root.  Two-sided for the leaf
        # quads (tree.frag flips normals on back faces).
        self._root = self.base.terrain_root.attach_new_node("tree_root")
        self._root.set_two_sided(True)
        # Scalar-fallback sway defaults until the first late_update.
        self._root.set_shader_input("u_wind_dir", LVecBase2f(1.0, 0.0))
        self._root.set_shader_input("u_sway_base", _SWAY_BASE_MIN_M)
        self._root.set_shader_input("u_sway_gust", _SWAY_GUST_MIN_M)
        self._root.set_shader_input("u_gust_freq", _GUST_FREQ_MIN)
        # u_time_s is NOT inherited from terrain_root (grass binds its own
        # on grass_root) — trees accumulate and bind their own, in lockstep.
        self._root.set_shader_input("u_time_s", 0.0)

        cfg = self.base._config
        # Shadow-refinement gate (lit_surface.glsl) for BOTH the mesh and
        # impostor draws.  Bound HERE, not inherited: terrain_root above us
        # pins u_refine = 1.0 for the terrain, foliage follows the preset.
        self._root.set_shader_input("u_refine", 1.0 if cfg.gfx_foliage_shadow_refine else 0.0)
        for kind in _TREE_KINDS:
            kroot = self._root.attach_new_node(f"tree_{kind.tag}")
            # Mesh draws fade OUT across the mesh window; impostor nodes
            # override u_fade_* with their own window below (node-level
            # ShaderInputs win over inherited ones).
            kroot.set_shader_input(
                "u_fade_start_m", float(getattr(cfg, f"{kind.prefix}_mesh_fade_start_m"))
            )
            kroot.set_shader_input(
                "u_fade_end_m", float(getattr(cfg, f"{kind.prefix}_mesh_fade_end_m"))
            )
            kroot.set_shader_input("u_sway_gain", kind.sway_gain)
            self._kind_roots[kind.tag] = kroot

        _build_volumes_fn(self)
        subscribe_terrain_events(self)

    def late_update(self, dt: float) -> None:
        """Sync scalar sway uniforms; rebuild/re-bake what changed."""
        if self._root is None:
            return

        if self.zone_store.version != self._store_version_built:
            _build_volumes_fn(self)
        elif self._dirty_volumes:
            for key in tuple(self._dirty_volumes):
                _rebuild_volume(self, key)
            self._dirty_volumes.clear()

        sync_sway_uniforms(self, dt)

    def on_destroy(self) -> None:
        """Detach all tree nodes and unsubscribe from the bus."""
        unsubscribe_terrain_events(self)
        if self._root is not None:
            self._root.remove_node()
            self._root = None
        if self.lighting_pipeline is not None:
            self.lighting_pipeline.set_static_occluders(None)
        self._volume_occluders.clear()
        self._kind_roots.clear()
        self._volume_nodes.clear()
        self._mesh_geoms.clear()
        self._impostor_geoms.clear()
        self._atlas_tex.clear()
        self._impostor_tex.clear()

    # ------------------------------------------------------------------
    # Event handlers (mark dirty only — work happens in late_update)
    # ------------------------------------------------------------------

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        _on_terrain_edited_impl(self, event)

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        _on_chunk_loaded_impl(self, event)

    def _mark_dirty_for_coords(self, coords: Any) -> None:
        """Queue a placement re-bake for volumes touching these chunks."""
        if self.base is None:
            return
        chunk_m = float(self.base._config.chunk_meters)
        for kind in _TREE_KINDS:
            for vol in self.zone_store.volumes(kind.tag):
                key = (kind.tag, vol.id)
                if key in self._dirty_volumes:
                    continue
                if any(vol.intersects_chunk(c, chunk_m) for c in coords):
                    self._dirty_volumes.add(key)

    # ------------------------------------------------------------------
    # Build — delegated to _impl.tree_build
    # ------------------------------------------------------------------

    def _build_volumes(self) -> None:
        """(Re)create every volume's draw nodes from the zone store."""
        _build_volumes_fn(self)

    def _rebuild_volume(self, key: tuple[str, int]) -> None:
        """Re-bake + rebuild one volume (terrain under it changed)."""
        _rebuild_volume(self, key)

    def _build_volume(self, kind: _TreeKind, vol: Any) -> None:
        """Bake one volume's placements and create its mesh+impostor draws."""
        _build_volume(self, kind, vol)

    # ------------------------------------------------------------------
    # Static occluders — delegated to _impl.tree_occluders
    # ------------------------------------------------------------------

    def _push_occluders(self) -> None:
        """Merge every volume's occluder set and hand it to the pipeline."""
        _push_occluders(self)
