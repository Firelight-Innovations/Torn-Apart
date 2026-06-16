"""
world/mote_renderer.py — GPU-instanced wind particles: dust motes + leaf litter.

Two render components, both consuming the wind field uploaded on
``App.terrain_root`` by ``WindSystemComponent`` (``world/wind_renderer.py``).
Neither needs any new uniform: parented under ``terrain_root`` they inherit the
wind contract (``u_wind_tex``/``u_wind_origin``/``u_wind_cell_m``/
``u_wind_cells``/``u_wind_enabled``), and — for leaves — the full
radiance-cascade + froxel-fog lighting set (the lit-surface contract, bound on
``render`` by GpuLightingPipeline) plus the camera (``u_cam_pos``), all by
scene-graph inheritance (the same mechanism that lights the grass).  The one uniform NOT
inherited is the animation clock ``u_time_s``: grass binds it on grass_root (its
own node), not terrain_root, so each mote component accumulates and binds its
own ``u_time_s`` in ``late_update`` (the same dt-accumulation grass uses).  Both
are **GPU-instanced with zero CPU per-particle state**: every
instance derives its placement / life / motion in the vertex shader from
``gl_InstanceID`` (``world/mote_shaders.py``), so the CPU only allocates a node
and a count — never a particle array.

:class:`DustMoteComponent`
    ``config.wind_mote_count`` ever-present dust/pollen specks in a single
    camera-anchored **wrapping lattice** (home cell = ``floor(cam/box)*box`` +
    hashed offset, so motes tile space and recycle with no spawn pop as the
    camera flies).  Each mote loops a ``sin(life*PI)`` life, is carried downwind
    by the local field, jittered by a re-hashed Brownian step and lifted gently
    by turbulence.  Additive, depth-test on / depth-write off, soft radial
    ``dust_mote`` texture.  A storm looks denser purely from faster motion — the
    count is fixed.

:class:`LeafLitterComponent`
    One hardware-instanced node per ``ZoneStore`` volume tagged ``"trees"`` (the
    grass ``_build_volumes`` / ``store.version`` rebuild pattern, with
    ``leaf_hash_seed``/``leaf_instance_count`` from ``zones/grass_placement.py``).
    Leaves spawn inside the volume biased low, tumble under two hashed angular
    rates, and are carried by ``local wind × (0.3 + 0.7·gust)`` so litter settles
    in calm air and **streams** in gusts/storms; a looping life recycles them.
    Alpha-blended (leaves are opaque-ish), lit by the SAME cascades as the grass
    (``mote_leaf.frag`` copies ``grass.frag``'s lighting/fog taps).

    Implementation lives in ``_impl.leaf_litter``; re-exported here so the
    public import path is unchanged.

Both are **GPU lighting backend only** (they need the live ``GpuLightingPipeline``
binding the inherited uniforms on ``terrain_root``); on the CPU backend — or with
no wind field — they disable themselves with a log line, exactly like grass.

Example (wired by main.py)
--------------------------
    dust_go = instantiate()
    dust_go.add_component(DustMoteComponent, base=app,
                          lighting_pipeline=pipeline)
    leaf_go = instantiate()
    leaf_go.add_component(LeafLitterComponent, base=app, zone_store=zone_store,
                          lighting_pipeline=pipeline)

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    GeomNode,
    NodePath,
    Shader,
)

from fire_engine.core import get_logger
from fire_engine.core.rng import for_domain
from fire_engine.render._impl.quad import setup_additive_instanced_node as _setup_additive
from fire_engine.render.component import Component
from fire_engine.render.vegetation import mote_shaders
from fire_engine.render.vegetation._impl.leaf_litter import LeafLitterComponent
from fire_engine.render.vegetation._impl.mote_utils import build_quad_geom as _build_quad_geom
from fire_engine.render.vegetation._impl.mote_utils import mote_texture as _mote_texture

__all__ = ["DustMoteComponent", "LeafLitterComponent"]

_log = get_logger("world.motes")


# ---------------------------------------------------------------------------
# Dust motes
# ---------------------------------------------------------------------------


class DustMoteComponent(Component):
    """
    Render component for GPU-instanced ever-present dust/pollen motes.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``terrain_root`` and ``_config``.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` (CPU backend)
        disables the component (no wind field / inherited uniforms there).

    Units: meters, seconds.  World-space Z-up.
    """

    def __init__(self, base: Any = None, lighting_pipeline: Any = None) -> None:
        super().__init__()
        self.base = base
        self.lighting_pipeline = lighting_pipeline
        self._node: NodePath | None = None
        self._time_s: float = 0.0

    def start(self) -> None:
        """Build the instanced mote node (once) and bind its uniforms."""
        if self.base is None:
            _log.warning("DustMoteComponent: missing base — disabled")
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "DustMoteComponent: GPU lighting pipeline required "
                '(lighting_backend = "gpu") — disabled'
            )
            self.enabled = False
            return

        cfg = self.base._config
        count = int(cfg.wind_mote_count)
        if count <= 0:
            _log.info("DustMoteComponent: wind_mote_count <= 0 — nothing to draw")
            self.enabled = False
            return

        shader = Shader.make(
            Shader.SL_GLSL, vertex=mote_shaders.DUST_VERTEX, fragment=mote_shaders.DUST_FRAGMENT
        )

        geom_node = GeomNode("dust_motes")
        geom_node.add_geom(_build_quad_geom())
        node = self.base.terrain_root.attach_new_node(geom_node)
        # Shader + instance count on the SAME node: set_instance_count creates a
        # node-level ShaderAttrib which REPLACES (not composes with) an inherited
        # one, so the shader must live here too.  ShaderInput attribs DO compose,
        # so the inherited wind/fog/camera uniforms still arrive from terrain_root.
        node.set_shader(shader)
        node.set_instance_count(count)
        node.set_shader_input("u_hash_seed", _dust_hash_seed())
        node.set_shader_input("u_mote_box_m", float(cfg.wind_mote_box_m))
        node.set_shader_input("u_mote_size_m", float(cfg.wind_mote_size_m))
        node.set_shader_input("u_mote_life_s", float(cfg.wind_mote_life_s))
        node.set_shader_input("u_dust_tex", _mote_texture("dust_mote"))
        # u_time_s is the shared real-time animation clock.  Grass binds it on
        # ITS own node (grass_root), so it is NOT inherited here — bind + refresh
        # our own copy each frame (same dt-accumulation grass uses; the two
        # clocks start together at component start).
        node.set_shader_input("u_time_s", 0.0)

        # Additive glow: depth-test ON (motes hide behind terrain) but
        # depth-write OFF (no sorting needed — additive is order-independent).
        # Instances are positioned in the shader — give the node an infinite
        # bounding box so Panda3D never culls by the base quad's origin bounds.
        _setup_additive(node, geom_node)
        self._node = node

        _log.info(
            "Dust motes online: %d instances, %.1f m lattice box", count, float(cfg.wind_mote_box_m)
        )

    def late_update(self, dt: float) -> None:
        """Advance the shared animation clock (the wind advection rides on it)."""
        if self._node is None:
            return
        self._time_s += dt
        self._node.set_shader_input("u_time_s", self._time_s)

    def on_destroy(self) -> None:
        """Detach the mote node."""
        if self._node is not None:
            self._node.remove_node()
            self._node = None


# ---------------------------------------------------------------------------
# Hash seeds (Hard Rule 2 — all randomness via for_domain)
# ---------------------------------------------------------------------------


def _dust_hash_seed() -> int:
    """Deterministic dust-mote instance-chain seed via
    ``for_domain("wind", "motes")``.  Bounded to ``[0, 2**31)`` (Panda3D passes
    shader-input ints as signed)."""
    return int(for_domain("wind", "motes").integers(0, 2**31))
