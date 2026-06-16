"""
Private implementation of LeafLitterComponent, split from mote_renderer.py
to satisfy the one-public-class-per-module rule.

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import Any

from panda3d.core import (
    Geom,
    GeomNode,
    LVecBase3f,
    NodePath,
    Shader,
    TransparencyAttrib,
)

from fire_engine.core import get_logger
from fire_engine.render.component import Component
from fire_engine.render.vegetation import mote_shaders
from fire_engine.render.vegetation._impl.zone_renderer import set_volume_bounds
from fire_engine.zones import leaf_hash_seed, leaf_instance_count

__all__ = ["LeafLitterComponent"]

_log = get_logger("world.motes")

# Leaf carry reach used to pad each volume's culling box (meters): a leaf can
# stream out of its volume by roughly (gust-scaled carry) × life — a few meters
# is plenty for the demo densities, and over-padding only relaxes culling.
_LEAF_CARRY_PAD_M = 6.0


class LeafLitterComponent(Component):
    """
    Render component for GPU-instanced leaf litter on ``"trees"`` volumes.

    One instanced node per :class:`~fire_engine.zones.ZoneVolume` tagged
    ``"trees"`` (the grass per-volume pattern).  Rebuilds when the
    ``ZoneStore.version`` changes, so a future tree/forest system that registers
    canopy volumes tagged ``"trees"`` gets leaf litter with zero wind-system
    changes.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``terrain_root`` and ``_config``.
    zone_store : fire_engine.zones.ZoneStore
        Volumes tagged ``"trees"`` get litter; ``version`` triggers a rebuild.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` disables.

    Units: meters, seconds.  World-space Z-up.

    Docs: docs/systems/render.vegetation._impl.md
    """

    def __init__(
        self, base: Any = None, zone_store: Any = None, lighting_pipeline: Any = None
    ) -> None:
        super().__init__()
        self.base = base
        self.zone_store = zone_store
        self.lighting_pipeline = lighting_pipeline
        self._root: NodePath | None = None
        self._shader: Shader | None = None
        self._quad_geom: Geom | None = None
        self._leaf_tex = None
        self._volume_nodes: dict[int, NodePath] = {}
        self._store_version_built: int = -1
        self._time_s: float = 0.0

    def start(self) -> None:
        """Build the shared geom/shader and per-volume instanced nodes (once).

        Docs: docs/systems/render.vegetation._impl.md
        """
        if self.base is None or self.zone_store is None:
            _log.warning("LeafLitterComponent: missing base/zone_store — disabled")
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "LeafLitterComponent: GPU lighting pipeline required "
                '(lighting_backend = "gpu") — disabled'
            )
            self.enabled = False
            return

        from fire_engine.render.vegetation._impl.mote_utils import (
            build_quad_geom,
            mote_texture,
        )

        self._quad_geom = build_quad_geom()
        self._leaf_tex = mote_texture("leaf_sprite")
        self._root = self.base.terrain_root.attach_new_node("leaf_litter_root")
        self._shader = Shader.make(
            Shader.SL_GLSL, vertex=mote_shaders.LEAF_VERTEX, fragment=mote_shaders.LEAF_FRAGMENT
        )
        # Leaves are opaque-ish billboards drawn with an alpha-test discard; use
        # dual transparency so the fragment discard works and depth stays sane.
        self._root.set_transparency(TransparencyAttrib.M_binary)
        self._root.set_two_sided(True)
        # u_time_s is the shared real-time animation clock; grass binds it on its
        # own node, so bind + refresh our own on leaf_litter_root (ShaderInput
        # attribs COMPOSE down to the per-volume nodes even though those carry
        # their own node-level shader for instancing — same split as grass).
        self._root.set_shader_input("u_time_s", 0.0)

        self._build_volumes()

    def late_update(self, dt: float) -> None:
        """Advance the animation clock; rebuild nodes if the zone store changed.

        Docs: docs/systems/render.vegetation._impl.md
        """
        if self._root is None:
            return
        self._time_s += dt
        self._root.set_shader_input("u_time_s", self._time_s)
        if self.zone_store.version != self._store_version_built:
            self._build_volumes()

    def on_destroy(self) -> None:
        """Detach all leaf nodes.

        Docs: docs/systems/render.vegetation._impl.md
        """
        if self._root is not None:
            self._root.remove_node()
            self._root = None
        self._volume_nodes.clear()

    # ------------------------------------------------------------------

    def _build_volumes(self) -> None:
        """(Re)create one instanced node per ``"trees"`` volume."""
        for node in self._volume_nodes.values():
            node.remove_node()
        self._volume_nodes.clear()

        assert self._root is not None
        cfg = self.base._config
        total = 0
        for vol in self.zone_store.volumes("trees"):
            count = leaf_instance_count(vol, cfg)
            if count <= 0:
                continue
            geom_node = GeomNode(f"leaf_vol_{vol.id}")
            geom_node.add_geom(self._quad_geom)
            node = self._root.attach_new_node(geom_node)
            # Shader + instance count on the SAME node (node-level ShaderAttrib
            # replaces inherited; ShaderInputs compose — same caveat as grass).
            node.set_shader(self._shader)
            node.set_instance_count(count)
            node.set_shader_input("u_bounds_min", LVecBase3f(*vol.min_corner))
            node.set_shader_input("u_bounds_max", LVecBase3f(*vol.max_corner))
            node.set_shader_input("u_hash_seed", leaf_hash_seed(vol))
            node.set_shader_input("u_leaf_size_m", float(cfg.wind_leaf_size_m))
            node.set_shader_input("u_leaf_life_s", float(cfg.wind_mote_life_s))
            node.set_shader_input("u_leaf_tex", self._leaf_tex)

            # Instances are shader-positioned (and stream out of the volume on
            # gusts) — Panda3D would cull by the base quad's origin bounds.  Give
            # the node the volume box padded by the carry reach + leaf size, and
            # stop bounds recomputation (grass culling caveat).
            pad = _LEAF_CARRY_PAD_M + float(cfg.wind_leaf_size_m)
            set_volume_bounds(geom_node, vol, pad)
            self._volume_nodes[vol.id] = node
            total += count

        self._store_version_built = self.zone_store.version
        _log.info(
            "Leaf litter built: %d volume(s), %d instances total", len(self._volume_nodes), total
        )
