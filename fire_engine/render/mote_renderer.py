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
"""

from __future__ import annotations

from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    BoundingBox,
    ColorBlendAttrib,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LPoint3,
    LVecBase3f,
    NodePath,
    Shader,
    TransparencyAttrib,
)

from fire_engine.core import get_logger
from fire_engine.core.rng import for_domain
from fire_engine.render.component import Component
from fire_engine.render import mote_shaders
from fire_engine.zones import leaf_hash_seed, leaf_instance_count

__all__ = ["DustMoteComponent", "LeafLitterComponent"]

_log = get_logger("world.motes")

# Leaf carry reach used to pad each volume's culling box (meters): a leaf can
# stream out of its volume by roughly (gust-scaled carry) × life — a few meters
# is plenty for the demo densities, and over-padding only relaxes culling.
_LEAF_CARRY_PAD_M = 6.0


# ---------------------------------------------------------------------------
# Shared billboard quad (built once per component, drawn N times via instancing)
# ---------------------------------------------------------------------------

def _build_quad_geom() -> Geom:
    """
    Build the shared unit billboard quad: corners at xy ∈ {-1,+1}, z=0, UV 0–1.

    The vertex shaders offset these corners (in view space for dust, after a
    tumble rotation for leaves), so one tiny 4-vertex / 2-triangle Geom is the
    base for every instance — a fixed handful of vertices, never a per-particle
    array.
    """
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("mote_quad", fmt, Geom.UH_static)
    vdata.set_num_rows(4)
    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")
    # (-1,-1) (1,-1) (1,1) (-1,1)
    corners = ((-1.0, -1.0, 0.0, 0.0), (1.0, -1.0, 1.0, 0.0),
               (1.0, 1.0, 1.0, 1.0), (-1.0, 1.0, 0.0, 1.0))
    for x, y, u, v in corners:
        vw.add_data3(x, y, 0.0)
        tw.add_data2(u, v)
    tris = GeomTriangles(Geom.UH_static)
    tris.add_vertices(0, 1, 2)
    tris.add_vertices(0, 2, 3)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    return geom


def _mote_texture(name: str):
    """The procedural ``name`` texture as a Panda3D texture (linear-filtered
    so the soft dust falloff / leaf edges don't look chunky billboarded)."""
    from fire_engine.procedural import get as get_procedural
    from fire_engine.render.texture_bridge import to_panda_texture
    from panda3d.core import SamplerState  # type: ignore[import]
    tex = to_panda_texture(get_procedural(name))
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex


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
            _log.warning("DustMoteComponent: GPU lighting pipeline required "
                         "(lighting_backend = \"gpu\") — disabled")
            self.enabled = False
            return

        cfg = self.base._config
        count = int(cfg.wind_mote_count)
        if count <= 0:
            _log.info("DustMoteComponent: wind_mote_count <= 0 — nothing to draw")
            self.enabled = False
            return

        shader = Shader.make(Shader.SL_GLSL,
                             vertex=mote_shaders.DUST_VERTEX,
                             fragment=mote_shaders.DUST_FRAGMENT)

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
        node.set_transparency(TransparencyAttrib.M_none)
        node.set_attrib(ColorBlendAttrib.make(
            ColorBlendAttrib.M_add,
            ColorBlendAttrib.O_incoming_alpha,
            ColorBlendAttrib.O_one))
        node.set_depth_write(False)
        node.set_bin("fixed", 0)
        node.set_two_sided(True)

        # Instances are positioned in the shader (camera-anchored lattice), so
        # the box wanders with the camera; Panda3D would cull by the base quad's
        # origin bounds.  Give it an effectively-infinite box + set_final so it
        # never culls the whole node.  (Per-mote off-screen specks cost ~nothing.)
        big = 1.0e9
        geom_node.set_bounds(BoundingBox(LPoint3(-big, -big, -big),
                                         LPoint3(big, big, big)))
        geom_node.set_final(True)
        self._node = node

        _log.info("Dust motes online: %d instances, %.1f m lattice box",
                  count, float(cfg.wind_mote_box_m))

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
# Leaf litter
# ---------------------------------------------------------------------------

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
    """

    def __init__(self, base: Any = None, zone_store: Any = None,
                 lighting_pipeline: Any = None) -> None:
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
        """Build the shared geom/shader and per-volume instanced nodes (once)."""
        if self.base is None or self.zone_store is None:
            _log.warning("LeafLitterComponent: missing base/zone_store — disabled")
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning("LeafLitterComponent: GPU lighting pipeline required "
                         "(lighting_backend = \"gpu\") — disabled")
            self.enabled = False
            return

        self._quad_geom = _build_quad_geom()
        self._leaf_tex = _mote_texture("leaf_sprite")
        self._root = self.base.terrain_root.attach_new_node("leaf_litter_root")
        self._shader = Shader.make(Shader.SL_GLSL,
                                   vertex=mote_shaders.LEAF_VERTEX,
                                   fragment=mote_shaders.LEAF_FRAGMENT)
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
        """Advance the animation clock; rebuild nodes if the zone store changed."""
        if self._root is None:
            return
        self._time_s += dt
        self._root.set_shader_input("u_time_s", self._time_s)
        if self.zone_store.version != self._store_version_built:
            self._build_volumes()

    def on_destroy(self) -> None:
        """Detach all leaf nodes."""
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
            geom_node.set_bounds(BoundingBox(
                LPoint3(vol.min_corner[0] - pad, vol.min_corner[1] - pad,
                        vol.min_corner[2] - pad),
                LPoint3(vol.max_corner[0] + pad, vol.max_corner[1] + pad,
                        vol.max_corner[2] + pad)))
            geom_node.set_final(True)
            self._volume_nodes[vol.id] = node
            total += count

        self._store_version_built = self.zone_store.version
        _log.info("Leaf litter built: %d volume(s), %d instances total",
                  len(self._volume_nodes), total)


# ---------------------------------------------------------------------------
# Hash seeds (Hard Rule 2 — all randomness via for_domain)
# ---------------------------------------------------------------------------

def _dust_hash_seed() -> int:
    """Deterministic dust-mote instance-chain seed via
    ``for_domain("wind", "motes")``.  Bounded to ``[0, 2**31)`` (Panda3D passes
    shader-input ints as signed)."""
    return int(for_domain("wind", "motes").integers(0, 2 ** 31))
