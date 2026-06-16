"""
Volume-build helpers for TreeRendererComponent, extracted from tree_renderer.py
to satisfy the ≤500-line module limit.

Each function takes the component instance as ``self_obj`` and operates on it
directly, preserving identical runtime behaviour.  Called from the class as
``_build_volume(self, kind, vol)``, etc.

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from panda3d.core import BoundingBox, GeomNode, LPoint3

from fire_engine.lighting import TreeOccluderSet
from fire_engine.zones import (
    SCALE_JITTER,
    bake_tree_instances,
    instances_data_block,
    species_mix_from_params,
)

if TYPE_CHECKING:
    from fire_engine.render.vegetation.tree_renderer import TreeRendererComponent

__all__ = ["build_volume", "build_volumes", "rebuild_volume"]

# Bounds margin past the scaled tree reach (sway never exceeds this).
_BOUNDS_PAD_M = 1.0

# Cascade sample height for the impostor stage, as a fraction of max height.
_LIGHT_OFFSET_FRAC = 0.45


def build_volumes(self_obj: TreeRendererComponent) -> None:
    """(Re)create every volume's draw nodes from the zone store.

    Docs: docs/systems/render.vegetation._impl.md
    """
    from fire_engine.render.vegetation.tree_renderer import _TREE_KINDS

    for nodes in self_obj._volume_nodes.values():
        for node in nodes:
            node.remove_node()
    self_obj._volume_nodes.clear()
    self_obj._volume_occluders.clear()
    self_obj._dirty_volumes.clear()

    total_inst = 0
    total_nodes = 0
    for kind in _TREE_KINDS:
        for vol in self_obj.zone_store.volumes(kind.tag):
            build_volume(self_obj, kind, vol)
            nodes = self_obj._volume_nodes.get((kind.tag, vol.id), [])
            total_nodes += len(nodes)

    for nodes in self_obj._volume_nodes.values():
        for node in nodes:
            total_inst += node.get_instance_count()
    self_obj._store_version_built = self_obj.zone_store.version
    from fire_engine.render.vegetation._impl.tree_occluders import push_occluders

    push_occluders(self_obj)
    from fire_engine.core import get_logger

    _log = get_logger("world.trees")
    _log.info("Trees built: %d draw node(s), %d instance draw(s) total", total_nodes, total_inst)


def rebuild_volume(self_obj: TreeRendererComponent, key: tuple[str, int]) -> None:
    """Re-bake + rebuild one volume (terrain under it changed).

    Docs: docs/systems/render.vegetation._impl.md
    """
    from fire_engine.core import get_logger
    from fire_engine.render.vegetation._impl.tree_occluders import push_occluders
    from fire_engine.render.vegetation.tree_renderer import _TREE_KINDS

    tag, vol_id = key
    for node in self_obj._volume_nodes.pop(key, ()):
        node.remove_node()
    self_obj._volume_occluders.pop(key, None)
    vol = self_obj.zone_store.get(vol_id)
    if vol is not None:
        kind = next(k for k in _TREE_KINDS if k.tag == tag)
        build_volume(self_obj, kind, vol)
    push_occluders(self_obj)
    get_logger("world.trees").debug("Tree volume %d (%s) re-baked", vol_id, tag)


def build_volume(self_obj: TreeRendererComponent, kind: Any, vol: Any) -> None:
    """Bake one volume's placements and create its mesh+impostor draws.

    Docs: docs/systems/render.vegetation._impl.md
    """
    from fire_engine.procedural import get as get_procedural
    from fire_engine.render.bridges.texture_bridge import to_data_texture_f32
    from fire_engine.render.vegetation._impl.tree_occluders import (
        species_canopy_sigma,
        species_splat_rgb,
    )
    from fire_engine.render.vegetation._impl.tree_species_cache import (
        impostor_geom,
        mesh_geom,
        species_atlas,
        species_impostor,
    )

    cfg = self_obj.base._config
    mix = species_mix_from_params(vol.params, str(getattr(cfg, f"{kind.prefix}_default_species")))
    variant_sets = {name: get_procedural(name) for name, _ in mix}
    variant_counts = {name: vs.n_variants for name, vs in variant_sets.items()}
    inst = bake_tree_instances(
        vol, cfg, self_obj.chunk_provider.chunks, mix, variant_counts, kind=kind.tag
    )
    if inst.count == 0:
        self_obj._volume_nodes[(kind.tag, vol.id)] = []
        self_obj._volume_occluders.pop((kind.tag, vol.id), None)
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
        occ_sigma[mask] = np.float32(species_canopy_sigma(self_obj, name, vs)) / np.maximum(
            inst.scale[mask], 1e-3
        )
        bark_rgb, leaf_rgb = species_splat_rgb(self_obj, name, vs)
        occ_bark[mask] = bark_rgb
        occ_leaf[mask] = leaf_rgb
    self_obj._volume_occluders[(kind.tag, vol.id)] = TreeOccluderSet(
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
    kroot = self_obj._kind_roots[kind.tag]
    nodes: list[Any] = []

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
            geom_node.add_geom(mesh_geom(self_obj, name, v, vs))
            node = kroot.attach_new_node(geom_node)
            # Shader and instance count MUST live on the same node (the
            # grass caveat): set_instance_count creates a node-level
            # ShaderAttrib that REPLACES an inherited one.  ShaderInput
            # attribs compose, so the kind/root uniforms still arrive.
            node.set_shader(self_obj._mesh_shader)
            node.set_instance_count(n)
            node.set_shader_input(
                "u_inst_tex", to_data_texture_f32(instances_data_block(inst, mask))
            )
            node.set_shader_input("u_atlas", species_atlas(self_obj, name, vs))
            # Instances are shader-positioned — give the node the
            # volume's real box plus the scaled tree reach, and stop
            # bounds recomputation (Panda3D would cull by the base
            # Geom's origin-local bounds otherwise).
            geom_node.set_bounds(bounds)
            geom_node.set_final(True)
            nodes.append(node)

        # --- impostor draw (far LOD), one per species ----------------
        imp_node = GeomNode(f"tree_imp_{kind.tag}_vol_{vol.id}_{name}")
        imp_node.add_geom(impostor_geom(self_obj, name, vs))
        inode = kroot.attach_new_node(imp_node)
        inode.set_shader(self_obj._impostor_shader)
        inode.set_instance_count(int(np.count_nonzero(species_mask)))
        inode.set_shader_input(
            "u_inst_tex", to_data_texture_f32(instances_data_block(inst, species_mask))
        )
        inode.set_shader_input("u_sprite", species_impostor(self_obj, name, vs))
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

    self_obj._volume_nodes[(kind.tag, vol.id)] = nodes
