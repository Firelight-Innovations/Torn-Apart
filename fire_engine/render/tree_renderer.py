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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    BoundingBox,
    Geom,
    GeomNode,
    LPoint3,
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
from fire_engine.lighting import TreeOccluderSet
from fire_engine.render import flora_shaders, tree_shaders
from fire_engine.render.component import Component
from fire_engine.render.flora_renderer import _build_cross_geom

# Weather → sway mapping shared with grass/flora so the scalar fallback
# moves all vegetation in lockstep (scaled per kind via u_sway_gain).
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
    SCALE_JITTER,
    bake_tree_instances,
    instances_data_block,
    species_mix_from_params,
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
        self._mesh_shader: Shader | None = None
        self._impostor_shader: Shader | None = None
        self._kind_roots: dict[str, NodePath] = {}
        # Per-(kind, volume.id): every NodePath built for that volume —
        # removed together on rebuild/re-bake.
        self._volume_nodes: dict[tuple[str, int], list[NodePath]] = {}
        self._dirty_volumes: set[tuple[str, int]] = set()
        self._store_version_built: int = -1
        self._time_s: float = 0.0
        # Caches keyed by species (+ variant) — uploads happen once even
        # when many volumes share a species.
        self._mesh_geoms: dict[tuple[str, int], Geom] = {}
        self._impostor_geoms: dict[str, Geom] = {}
        self._atlas_tex: dict[str, Texture] = {}
        self._impostor_tex: dict[str, Texture] = {}
        # Per-(kind, volume.id) static-occluder sets (lighting/occluders.py)
        # — merged + pushed to the pipeline so the light cascades see trees.
        self._volume_occluders: dict[tuple[str, int], TreeOccluderSet] = {}
        # Per-species mean (bark_rgb, leaf_rgb) splat colours from the atlas.
        self._species_occ_rgb: dict[str, tuple] = {}
        # Per-species leaf-derived canopy extinction (per meter, scale 1.0).
        self._species_sigma: dict[str, float] = {}

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
        elif self._dirty_volumes:
            for key in tuple(self._dirty_volumes):
                self._rebuild_volume(key)
            self._dirty_volumes.clear()

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
        """Detach all tree nodes and unsubscribe from the bus."""
        if self.bus is not None:
            self.bus.unsubscribe(TerrainEditedEvent, self._on_terrain_edited)
            self.bus.unsubscribe(ChunkLoadedEvent, self._on_chunk_loaded)
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
        coords = event.chunk_coords
        if isinstance(coords, tuple) and len(coords) == 3 and isinstance(coords[0], int):
            coords = (coords,)
        self._mark_dirty_for_coords(coords)

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        self._mark_dirty_for_coords((event.coord,))

    def _mark_dirty_for_coords(self, coords) -> None:
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
    # Build
    # ------------------------------------------------------------------

    def _build_volumes(self) -> None:
        """(Re)create every volume's draw nodes from the zone store."""
        for nodes in self._volume_nodes.values():
            for node in nodes:
                node.remove_node()
        self._volume_nodes.clear()
        self._volume_occluders.clear()
        self._dirty_volumes.clear()

        total_inst = 0
        total_nodes = 0
        for kind in _TREE_KINDS:
            for vol in self.zone_store.volumes(kind.tag):
                self._build_volume(kind, vol)
                nodes = self._volume_nodes.get((kind.tag, vol.id), ())
                total_nodes += len(nodes)

        for nodes in self._volume_nodes.values():
            for node in nodes:
                total_inst += node.get_instance_count()
        self._store_version_built = self.zone_store.version
        self._push_occluders()
        _log.info(
            "Trees built: %d draw node(s), %d instance draw(s) total", total_nodes, total_inst
        )

    def _rebuild_volume(self, key: tuple[str, int]) -> None:
        """Re-bake + rebuild one volume (terrain under it changed)."""
        tag, vol_id = key
        for node in self._volume_nodes.pop(key, ()):
            node.remove_node()
        self._volume_occluders.pop(key, None)
        vol = self.zone_store.get(vol_id)
        if vol is not None:
            kind = next(k for k in _TREE_KINDS if k.tag == tag)
            self._build_volume(kind, vol)
        self._push_occluders()
        _log.debug("Tree volume %d (%s) re-baked", vol_id, tag)

    def _build_volume(self, kind: _TreeKind, vol) -> None:
        """Bake one volume's placements and create its mesh+impostor draws."""
        from fire_engine.procedural import get as get_procedural
        from fire_engine.render.texture_bridge import to_data_texture_f32

        cfg = self.base._config
        mix = species_mix_from_params(
            vol.params, str(getattr(cfg, f"{kind.prefix}_default_species"))
        )
        variant_sets = {name: get_procedural(name) for name, _ in mix}
        variant_counts = {name: vs.n_variants for name, vs in variant_sets.items()}
        inst = bake_tree_instances(
            vol, cfg, self.chunk_provider.chunks, mix, variant_counts, kind=kind.tag
        )
        if inst.count == 0:
            self._volume_nodes[(kind.tag, vol.id)] = []
            self._volume_occluders.pop((kind.tag, vol.id), None)
            return

        # Static-occluder set for the lighting cascades: per-instance height
        # and canopy reach scaled from the species pool extents, per-meter
        # canopy extinction from the species' REAL leaf area (a dense oak
        # shades harder than a near-bare snag; sigma scales 1/instance-scale
        # since leaf area grows s² against canopy volume s³), splat colours
        # averaged from the species atlas (bounce albedo).  Pushed to the
        # pipeline by the caller (_build_volumes / _rebuild_volume).
        occ_h = np.empty(inst.count, np.float32)
        occ_r = np.empty(inst.count, np.float32)
        occ_sigma = np.empty(inst.count, np.float32)
        occ_bark = np.empty((inst.count, 3), np.float32)
        occ_leaf = np.empty((inst.count, 3), np.float32)
        for s_idx, name in enumerate(inst.species_names):
            mask = inst.species_idx == s_idx
            if not bool(mask.any()):
                continue
            vs = variant_sets[name]
            occ_h[mask] = np.float32(vs.max_height_m) * inst.scale[mask]
            occ_r[mask] = np.float32(vs.max_radius_m) * inst.scale[mask]
            occ_sigma[mask] = np.float32(self._species_canopy_sigma(name, vs)) / np.maximum(
                inst.scale[mask], 1e-3
            )
            bark_rgb, leaf_rgb = self._species_splat_rgb(name, vs)
            occ_bark[mask] = bark_rgb
            occ_leaf[mask] = leaf_rgb
        self._volume_occluders[(kind.tag, vol.id)] = TreeOccluderSet(
            x=inst.x,
            y=inst.y,
            z=inst.z,
            height_m=occ_h,
            canopy_r_m=occ_r,
            canopy_sigma=occ_sigma,
            bark_rgb=occ_bark,
            leaf_rgb=occ_leaf,
        )

        scale_min, scale_span = SCALE_JITTER[kind.tag]
        scale_max = scale_min + scale_span
        kroot = self._kind_roots[kind.tag]
        nodes: list[NodePath] = []

        for s_idx, name in enumerate(inst.species_names):
            vs = variant_sets[name]
            species_mask = inst.species_idx == s_idx
            if not bool(species_mask.any()):
                continue
            pad = max(vs.max_height_m, vs.max_radius_m) * scale_max + _BOUNDS_PAD_M
            bounds = BoundingBox(
                LPoint3(vol.min_corner[0] - pad, vol.min_corner[1] - pad, vol.min_corner[2] - pad),
                LPoint3(vol.max_corner[0] + pad, vol.max_corner[1] + pad, vol.max_corner[2] + pad),
            )

            # --- variant-mesh draws (near LOD) ---------------------------
            for v in range(vs.n_variants):
                mask = species_mask & (inst.variant == v)
                n = int(np.count_nonzero(mask))
                if n == 0:
                    continue
                geom_node = GeomNode(f"tree_{kind.tag}_vol_{vol.id}_{name}_v{v}")
                geom_node.add_geom(self._mesh_geom(name, v, vs))
                node = kroot.attach_new_node(geom_node)
                # Shader and instance count MUST live on the same node (the
                # grass caveat): set_instance_count creates a node-level
                # ShaderAttrib that REPLACES an inherited one.  ShaderInput
                # attribs compose, so the kind/root uniforms still arrive.
                node.set_shader(self._mesh_shader)
                node.set_instance_count(n)
                node.set_shader_input(
                    "u_inst_tex", to_data_texture_f32(instances_data_block(inst, mask))
                )
                node.set_shader_input("u_atlas", self._species_atlas(name, vs))
                # Instances are shader-positioned — give the node the
                # volume's real box plus the scaled tree reach, and stop
                # bounds recomputation (Panda3D would cull by the base
                # Geom's origin-local bounds otherwise).
                geom_node.set_bounds(bounds)
                geom_node.set_final(True)
                nodes.append(node)

            # --- impostor draw (far LOD), one per species ----------------
            imp_node = GeomNode(f"tree_imp_{kind.tag}_vol_{vol.id}_{name}")
            imp_node.add_geom(self._impostor_geom(name, vs))
            inode = kroot.attach_new_node(imp_node)
            inode.set_shader(self._impostor_shader)
            inode.set_instance_count(int(np.count_nonzero(species_mask)))
            inode.set_shader_input(
                "u_inst_tex", to_data_texture_f32(instances_data_block(inst, species_mask))
            )
            inode.set_shader_input("u_sprite", self._species_impostor(name, vs))
            inode.set_shader_input("u_variants", float(vs.n_variants))
            # Fade IN over the mesh window, OUT over the impostor window —
            # overriding the kind root's mesh-window u_fade_*.
            inode.set_shader_input(
                "u_mesh_fade_start_m", float(getattr(cfg, f"{kind.prefix}_mesh_fade_start_m"))
            )
            inode.set_shader_input(
                "u_mesh_fade_end_m", float(getattr(cfg, f"{kind.prefix}_mesh_fade_end_m"))
            )
            inode.set_shader_input(
                "u_fade_start_m", float(getattr(cfg, f"{kind.prefix}_impostor_fade_start_m"))
            )
            inode.set_shader_input(
                "u_fade_end_m", float(getattr(cfg, f"{kind.prefix}_impostor_fade_end_m"))
            )
            inode.set_shader_input("u_sway_pivot", kind.sway_pivot)
            inode.set_shader_input("u_light_offset_m", vs.max_height_m * _LIGHT_OFFSET_FRAC)
            imp_node.set_bounds(bounds)
            imp_node.set_final(True)
            nodes.append(inode)

        self._volume_nodes[(kind.tag, vol.id)] = nodes

    # ------------------------------------------------------------------
    # Static occluders (the light cascades see the trees)
    # ------------------------------------------------------------------

    def _push_occluders(self) -> None:
        """Merge every volume's occluder set and hand it to the pipeline."""
        if self.lighting_pipeline is None:
            return
        sets = [s for s in self._volume_occluders.values() if s.count]
        self.lighting_pipeline.set_static_occluders(TreeOccluderSet.merge(sets) if sets else None)

    def _species_canopy_sigma(self, name: str, vs) -> float:
        """
        Per-meter canopy extinction for a species at scale 1.0.

        How thick the leaves are, measured from the actual meshes: mean
        one-sided leaf area over the variant pool
        (``procedural.flora.mesh_leaf_area_m2``) ÷ the canopy ellipsoid
        volume from the pool extents, × 0.5 (randomly-oriented flat cards
        present half their area to any direction).  Transmittance through
        ``X`` meters of crown centre is then ``exp(-sigma·X)`` — a leafy
        oak shades hard, a two-tuft snag barely dims the ground.  Cached
        per species; deterministic (meshes are seeded procedural content).
        """
        sigma = self._species_sigma.get(name)
        if sigma is None:
            from fire_engine.lighting.occluders import CANOPY_HALF_HEIGHT_FRAC
            from fire_engine.procedural.flora import mesh_leaf_area_m2

            leaf_area = float(np.mean([mesh_leaf_area_m2(m) for m in vs.meshes]))
            cv = CANOPY_HALF_HEIGHT_FRAC * float(vs.max_height_m)
            r = float(vs.max_radius_m)
            volume = (4.0 / 3.0) * np.pi * r * r * max(cv, 1e-3)
            sigma = 0.5 * leaf_area / max(volume, 1e-3)
            self._species_sigma[name] = sigma
        return sigma

    def _species_splat_rgb(self, name: str, vs) -> tuple:
        """
        Mean linear bark/leaf splat colours for a species, from its atlas.

        The atlas is bark on the left half (opaque) and the leaf card on the
        right half (binary alpha) — averaging each half gives the GI bounce
        colour for trunk/canopy cells without any new species API.  Cached
        per species; deterministic (the atlas is seeded procedural content).
        """
        cached = self._species_occ_rgb.get(name)
        if cached is None:
            atlas = vs.atlas.astype(np.float32) / 255.0
            half = atlas.shape[1] // 2
            bark = atlas[:, :half, :3].reshape(-1, 3).mean(axis=0)
            leaf_px = atlas[:, half:, :].reshape(-1, 4)
            sel = leaf_px[:, 3] > 0.5
            leaf = leaf_px[sel, :3].mean(axis=0) if bool(sel.any()) else bark
            # sRGB → linear (the cascade albedo channel is linear).
            cached = ((bark**2.2).astype(np.float32), (leaf**2.2).astype(np.float32))
            self._species_occ_rgb[name] = cached
        return cached

    # ------------------------------------------------------------------
    # Cached species resources (geoms + textures upload once per species)
    # ------------------------------------------------------------------

    def _mesh_geom(self, name: str, variant: int, vs) -> Geom:
        """The species' variant mesh as a Geom (uploaded once, shared)."""
        key = (name, variant)
        geom = self._mesh_geoms.get(key)
        if geom is None:
            from fire_engine.render.geometry_bridge import to_geom

            geom = to_geom(vs.meshes[variant])
            self._mesh_geoms[key] = geom
        return geom

    def _impostor_geom(self, name: str, vs) -> Geom:
        """Crossed-quad billboard sized to the species' impostor raster."""
        geom = self._impostor_geoms.get(name)
        if geom is None:
            geom = _build_cross_geom(vs.impostor_height_m, vs.impostor_width_m, 2)
            self._impostor_geoms[name] = geom
        return geom

    def _species_atlas(self, name: str, vs) -> Texture:
        """The species' bark/leaf atlas as a nearest-filtered texture."""
        tex = self._atlas_tex.get(name)
        if tex is None:
            from fire_engine.render.texture_bridge import to_panda_texture

            tex = to_panda_texture(vs.atlas)
            self._atlas_tex[name] = tex
        return tex

    def _species_impostor(self, name: str, vs) -> Texture:
        """The species' impostor sprite strip as a texture."""
        tex = self._impostor_tex.get(name)
        if tex is None:
            from fire_engine.render.texture_bridge import to_panda_texture

            tex = to_panda_texture(vs.impostors)
            self._impostor_tex[name] = tex
        return tex
