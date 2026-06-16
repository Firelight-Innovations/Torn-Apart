"""
world/building_renderer.py — free-form buildings (render component).

``BuildingRendererComponent`` draws every building owned by a
:class:`~fire_engine.buildings.manager.BuildingManager` as real lit geometry.
Each building is meshed once (``buildings.meshing.mesh_building`` → building-
LOCAL :class:`MeshArrays`) and uploaded as one ``GeomNode`` via
``geometry_bridge.to_geom_node``; the node carries the building's
``position``/``rotation`` as its transform, so moving or rotating a building is
a transform write, never a remesh (``building.vert`` reconstructs the world
position from ``p3d_ModelMatrix``).  Buildings are unique (no instancing).

The component subscribes to :class:`BuildingChangedEvent`: ``"added"`` /
``"modified"`` re-mesh that building's node, ``"removed"`` detaches it.  All
node work happens in ``late_update`` (events only mark dirty), matching the
tree/zone renderers.

Lighting (radiance cascades + froxel fog) inherits by scene-graph parenting
under ``App.terrain_root`` where ``GpuLightingPipeline`` binds the lit-surface
contract — buildings light exactly like the terrain they stand on.  GPU
lighting backend only: on the CPU backend the component disables itself with a
log line (the building shader is the GPU contract).

Example (wired by main.py, behind ``debug_demo_building``)
----------------------------------------------------------
    go = instantiate()
    go.add_component(BuildingRendererComponent, base=app,
                     building_manager=mgr, lighting_pipeline=pipeline, bus=bus)
"""

from __future__ import annotations

from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    LQuaternionf,
    NodePath,
    Shader,
    Texture,
)

from fire_engine.core import get_logger
from fire_engine.core.event_bus import BuildingChangedEvent
from fire_engine.render import building_shaders
from fire_engine.render.component import Component

__all__ = ["BuildingRendererComponent"]

_log = get_logger("world.buildings")


class BuildingRendererComponent(Component):
    """
    Render component for free-form buildings.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``terrain_root`` and ``_config``.
    building_manager : fire_engine.buildings.BuildingManager
        Source of buildings; its ``version`` counter triggers a full rebuild
        and its ``BuildingChangedEvent``s trigger per-building rebuilds.
    lighting_pipeline : GpuLightingPipeline | None
        Must be the active GPU lighting pipeline; ``None`` (CPU backend)
        disables the component.
    bus : EventBus | None
        Subscribed to ``BuildingChangedEvent``.

    Units: meters, radians; world-space Z-up.
    """

    def __init__(
        self,
        base: Any = None,
        building_manager: Any = None,
        lighting_pipeline: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.manager = building_manager
        self.lighting_pipeline = lighting_pipeline
        self.bus = bus

        self._root: NodePath | None = None
        self._shader: Shader | None = None
        self._albedo: Texture | None = None
        self._nodes: dict[int, NodePath] = {}  # building id → its node
        self._dirty: set[int] = set()  # ids to (re)build
        self._removed: set[int] = set()  # ids to detach
        self._store_version_built: int = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Compile the shader, build the root, and draw every building once."""
        if self.base is None or self.manager is None:
            _log.warning("BuildingRendererComponent: missing base/manager — disabled")
            self.enabled = False
            return
        if self.lighting_pipeline is None:
            _log.warning(
                "BuildingRendererComponent: GPU lighting pipeline "
                'required (lighting_backend = "gpu") — disabled'
            )
            self.enabled = False
            return

        self._shader = Shader.make(
            Shader.SL_GLSL,
            vertex=building_shaders.BUILDING_VERTEX,
            fragment=building_shaders.BUILDING_FRAGMENT,
        )
        self._albedo = self._load_albedo()

        # Parent under terrain_root so the lit-surface cascade/fog uniforms are
        # inherited; bind the shadow-refinement gate ON (terrain-grade — walls
        # are large surfaces that deserve crisp shadow edges).
        self._root = self.base.terrain_root.attach_new_node("building_root")
        self._root.set_shader_input("u_refine", 1.0)

        self._build_all()

        if self.bus is not None:
            self.bus.subscribe(BuildingChangedEvent, self._on_changed)

    def late_update(self, dt: float) -> None:
        """Process queued rebuilds/removals; full rebuild on a version jump."""
        if self._root is None:
            return
        if (
            self.manager.version != self._store_version_built
            and not self._dirty
            and not self._removed
        ):
            # A change we didn't get an event for (e.g. bulk apply) — resync.
            self._build_all()
            return
        for bid in tuple(self._removed):
            self._detach(bid)
        self._removed.clear()
        for bid in tuple(self._dirty):
            self._rebuild_one(bid)
        self._dirty.clear()
        self._store_version_built = self.manager.version

    def on_destroy(self) -> None:
        """Detach all building nodes and unsubscribe from the bus."""
        if self.bus is not None:
            self.bus.unsubscribe(BuildingChangedEvent, self._on_changed)
        if self._root is not None:
            self._root.remove_node()
            self._root = None
        self._nodes.clear()
        self._dirty.clear()
        self._removed.clear()

    # ------------------------------------------------------------------
    # Events (mark dirty only — work happens in late_update)
    # ------------------------------------------------------------------

    def _on_changed(self, event: BuildingChangedEvent) -> None:
        if event.change == "removed":
            self._removed.add(event.building_id)
            self._dirty.discard(event.building_id)
        else:  # "added" / "modified"
            self._dirty.add(event.building_id)
            self._removed.discard(event.building_id)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_all(self) -> None:
        """(Re)create nodes for every building the manager currently holds."""
        for node in self._nodes.values():
            node.remove_node()
        self._nodes.clear()
        self._dirty.clear()
        self._removed.clear()
        for b in self.manager.buildings():
            self._rebuild_one(b.id)
        self._store_version_built = self.manager.version
        _log.info("Buildings built: %d node(s)", len(self._nodes))

    def _rebuild_one(self, building_id: int) -> None:
        """Re-mesh and re-place one building (or detach it if it's gone)."""
        self._detach(building_id)
        building = self.manager.get(building_id)
        if building is None:
            return
        from fire_engine.buildings.meshing import mesh_building
        from fire_engine.render.geometry_bridge import to_geom_node

        mesh = mesh_building(building, self.base._config)
        if mesh.positions.shape[0] == 0:
            return
        geom_node = to_geom_node(mesh, name=f"building_{building_id}")
        assert self._root is not None
        node = self._root.attach_new_node(geom_node)
        node.set_shader(self._shader)
        if self._albedo is not None:
            node.set_texture(self._albedo)
        # Building-local mesh + node transform (D6): position + rotation.
        p = building.position
        node.set_pos(float(p.x), float(p.y), float(p.z))
        q = building.rotation
        node.set_quat(LQuaternionf(float(q.w), float(q.x), float(q.y), float(q.z)))
        self._nodes[building_id] = node

    def _detach(self, building_id: int) -> None:
        node = self._nodes.pop(building_id, None)
        if node is not None:
            node.remove_node()

    def _load_albedo(self) -> Texture | None:
        """The plaster-wall albedo texture (procedural; flat fallback if absent)."""
        try:
            from fire_engine.procedural import get as get_procedural
            from fire_engine.render.texture_bridge import to_panda_texture

            return to_panda_texture(get_procedural("plaster_wall"))
        except Exception as exc:  # pragma: no cover - content optional
            _log.warning("plaster_wall texture unavailable (%s) — flat albedo", exc)
            return None
