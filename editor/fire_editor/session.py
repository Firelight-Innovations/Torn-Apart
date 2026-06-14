"""EditorSession — a headless engine world the daemon edits and renders.

Wires the engine's terrain + lighting + save systems into one object the
services drive. Imports only headless ``fire_engine`` APIs (never ``fire_engine``
``world``'s panda3d-bound package init; the object model arrives in Phase E2 once
the engine split lands). Determinism is preserved: the session sets the world
seed via ``core.rng`` and adds no unseeded randomness (EDITOR_PRD hard rule 4),
so editor preview of seed N matches the game world of seed N.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.config import Config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.lighting import LightGrid, SunlightComputer, make_light_sampler
from fire_engine.save import SaveManager
from fire_engine.world.terrain import (
    BoxBrush,
    BrushMode,
    ChunkManager,
    CylinderBrush,
    SphereBrush,
    apply_brush,
    generate_chunk,
    raycast_voxel,
)

from .scene_objects import SceneObjectStore

# Z band of streamed chunks relative to the camera chunk. Mirrors the private
# band in fire_engine.world.terrain.chunk_manager (_Z_MIN/_Z_MAX); guarded against
# drift by tests/editor/test_session.py::test_region_matches_engine_desired_set.
_Z_MIN: int = -2
_Z_MAX: int = 4


class EditorSession:
    """One open world: terrain chunks, sunlight, and delta saves.

    Construct via :meth:`from_seed` or :meth:`from_save`; the bare constructor
    takes a fully-resolved :class:`~fire_engine.core.config.Config`.

    Attributes:
        config: The frozen engine config for this session.
        seed: ``config.world_seed`` (the active world seed).
        cm: ``ChunkManager`` — chunk store, provider, terrain ``Saveable``.
        lg / sc / sampler: light grid, sunlight computer, mesher light sampler.
        scene: ``SceneObjectStore`` — the authoring hierarchy of placeable objects.
        save_manager: ``SaveManager`` with terrain + scene systems registered.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.seed = int(config.world_seed)
        set_world_seed(self.seed)
        self.bus = EventBus()
        self.clock = Clock(fixed_dt=config.fixed_dt, bus=self.bus)
        self.cm = ChunkManager(config, self.bus)
        self.lg = LightGrid()
        self.sc = SunlightComputer(config, self.cm, self.lg, self.bus)
        self.sampler = make_light_sampler(self.lg, config)
        # Authoring scene graph (placeable objects) — persists in the save after
        # terrain so deltas apply in a stable order (EDITOR_PRD Phase E2).
        self.scene = SceneObjectStore()
        self.save_manager = SaveManager(config, self.clock)
        self.save_manager.register(self.cm)
        self.save_manager.register(self.scene)
        # Per-world hash offset for the procedural ground pattern — the SAME
        # derivation as main.py's terrain-shader setup (for_domain is seeded by
        # set_world_seed above), so the editor viewport's ground matches the
        # game's for the same seed.
        self.ground_seed = float(for_domain("terrain", "ground").integers(0, 65536))
        self._ground_lut: np.ndarray | None = None

    def ground_lut(self) -> np.ndarray:
        """The posterised ground palette LUT ``(rows, 256, 4) uint8`` (cached).

        Built from the same grass/dirt palettes the game's terrain shader bakes
        (world/terrain_shader.py), minus the demo-only GI test-room rows. Pure
        numpy — never import ``world.terrain_shader`` here (it pulls panda3d).
        """
        if self._ground_lut is None:
            from fire_engine.procedural.textures.dirt_ground import DIRT_PALETTE, DIRT_THRESHOLDS
            from fire_engine.procedural.textures.grass_ground import GRASS_PALETTE, GRASS_THRESHOLDS
            from fire_engine.procedural.textures.ground_lut import build_ground_lut
            from fire_engine.world.terrain import MATERIAL_DIRT, MATERIAL_GRASS

            self._ground_lut = build_ground_lut(
                {
                    MATERIAL_DIRT: (DIRT_PALETTE, DIRT_THRESHOLDS),
                    MATERIAL_GRASS: (GRASS_PALETTE, GRASS_THRESHOLDS),
                }
            )
        return self._ground_lut

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_seed(cls, seed: int, base_config: Config | None = None) -> EditorSession:
        """Generate a fresh world from ``seed`` (overrides ``config.world_seed``)."""
        base = base_config or load_config()
        cfg = dataclasses.replace(base, world_seed=int(seed))
        return cls(cfg)

    @classmethod
    def from_save(cls, path: str, base_config: Config | None = None) -> EditorSession:
        """Open a ``.ta`` save: regen baseline from seed, then apply terrain deltas.

        The save's header seed must match ``base_config.world_seed`` (game saves
        use the ``config.toml`` seed, so this matches by default); otherwise the
        engine raises ``SaveIncompatibleError``, surfaced to the client.
        """
        base = base_config or load_config()
        session = cls(base)
        session.save_manager.load(path)  # validates header; applies terrain delta
        return session

    # ------------------------------------------------------------------ #
    # Region / meshing
    # ------------------------------------------------------------------ #
    def region_coords(self, center: Vec3, radius: int) -> list[tuple[int, int, int]]:
        """Chunk coords for the editor region around ``center`` (distance-sorted).

        XY square radius ``radius`` about the camera chunk, Z band ``[-2, +4]``
        relative to it — the same shape the game streams. Sorted nearest-first so
        the viewport fills in from the camera outward.
        """
        ccx, ccy, ccz = self.cm.camera_chunk(center)
        coords: list[tuple[int, int, int]] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(_Z_MIN, _Z_MAX + 1):
                    coords.append((ccx + dx, ccy + dy, ccz + dz))
        coords.sort(key=lambda c: (c[0] - ccx) ** 2 + (c[1] - ccy) ** 2 + (c[2] - ccz) ** 2)
        return coords

    def ensure_loaded(self, coords) -> None:
        """Generate (from seed) any not-yet-loaded chunks in ``coords``."""
        for c in coords:
            self.cm.get_or_create(c)

    def relight(self) -> None:
        """Recompute sunlight for all loaded columns (call before meshing)."""
        self.sc.recompute_all_loaded()

    def mesh(self, coord: tuple[int, int, int]):
        """Build (and cache) the lit mesh for ``coord`` via the engine mesher."""
        return self.cm.mesh_chunk(coord, self.sampler)

    # ------------------------------------------------------------------ #
    # Queries / persistence
    # ------------------------------------------------------------------ #
    def raycast(self, origin: Vec3, direction: Vec3, max_distance_m: float = 100.0):
        """First solid voxel hit along a ray, or ``None``."""
        return raycast_voxel(
            origin,
            direction,
            self.cm,
            max_distance_m=max_distance_m,
            chunk_size=int(self.config.chunk_size),
            voxel_size=float(self.config.voxel_size),
        )

    def edited_chunk_count(self) -> int:
        """Number of loaded chunks deviating from the procedural baseline."""
        return sum(1 for ch in self.cm.chunks.values() if ch.edited)

    # ------------------------------------------------------------------ #
    # Brush editing (Phase E3)
    # ------------------------------------------------------------------ #
    def make_brush(
        self,
        shape: str,
        *,
        radius: float = 2.0,
        hx: float = 1.0,
        hy: float = 1.0,
        hz: float = 1.0,
        height: float = 2.0,
    ):
        """Build a brush instance from a shape name and parameters."""
        s = shape.lower()
        if s == "sphere":
            return SphereBrush(radius_m=float(radius))
        if s == "box":
            return BoxBrush(half_extents_m=Vec3(float(hx), float(hy), float(hz)))
        if s == "cylinder":
            return CylinderBrush(radius_m=float(radius), height_m=float(height))
        raise ValueError(f"unknown brush shape: {shape!r}")

    def chunk_coords_for_aabb(self, mn, mx) -> list[tuple[int, int, int]]:
        """Chunk coords overlapping a world-space AABB (brush footprint)."""
        cm = float(self.config.chunk_meters)
        lo = np.floor(np.asarray(mn, dtype=float) / cm).astype(int)
        hi = np.floor(np.asarray(mx, dtype=float) / cm).astype(int)
        return [
            (cx, cy, cz)
            for cx in range(int(lo[0]), int(hi[0]) + 1)
            for cy in range(int(lo[1]), int(hi[1]) + 1)
            for cz in range(int(lo[2]), int(hi[2]) + 1)
        ]

    def snapshot(self, coords) -> dict[tuple[int, int, int], np.ndarray]:
        """Copy the material arrays of ``coords`` (generating any missing)."""
        return {c: self.cm.get_or_create(c).materials.copy() for c in coords}

    def apply_brush_edit(self, brush, center: Vec3, mode: BrushMode, material: int):
        """Apply a brush; return ``(coords, touched, before, after)`` for undo.

        ``before``/``after`` are material snapshots over the brush's AABB chunks,
        taken either side of the engine ``apply_brush`` call. ``touched`` is the
        set the engine actually modified (a subset of ``coords``).
        """
        mn, mx = brush.aabb(center.to_numpy())
        coords = self.chunk_coords_for_aabb(mn, mx)
        before = self.snapshot(coords)
        touched = apply_brush(brush, center, mode, material, chunk_provider=self.cm, bus=self.bus)
        after = self.snapshot(coords)
        # Same-frame remesh (bypasses the 2-chunk stream budget) so the edit
        # has no see-through hole while border neighbours wait their turn.
        self.cm.remesh_edited(touched)
        return coords, touched, before, after

    def restore(self, snapshot: dict) -> None:
        """Write material snapshots back (undo/redo) and fix edited/dirty flags."""
        for coord, mats in snapshot.items():
            chunk = self.cm.get_or_create(coord)
            chunk.materials[...] = mats
            chunk.edited = not np.array_equal(mats, generate_chunk(coord, self.config))
            chunk.dirty = True
            self.cm.pending_meshes.pop(coord, None)
        # Remesh the restored chunks immediately (undo/redo should not show a
        # hole while the budgeted stream path catches up).
        self.cm.remesh_edited(snapshot.keys())

    def save(self, path: str) -> None:
        """Write a delta save (terrain edits) through the engine SaveManager."""
        self.save_manager.save(path)
