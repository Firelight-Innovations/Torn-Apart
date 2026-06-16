"""
fire_engine.world.terrain — Voxel terrain: chunks, generation, meshing, brushes,
raycasting, and streaming.

Layer 2 (Structure).  Pure Python/numpy — **no panda3d imports** (the mesh
arrays are handed to ``world/geometry_bridge.py`` for upload).  Everything is
deterministic from ``(world_seed, chunk coord)`` and fully headless-testable.

Public API
----------
Chunk          — 32³ uint8 material array + dirty/edited flags + world origin.
generate_chunk — pure-function chunk generation (flat baseline, grass skin).
MATERIAL_DIRT, MATERIAL_GRASS — material ids (1 = dirt bulk, 2 = grass skin).
build_mesh     — culled-face cube mesher → MeshArrays (mesh_style="blocky").
build_mesh_faceted — flat-shaded surface-nets mesher (mesh_style="faceted",
                 the default: semi-smooth Daggerfall-ish facets + per-face
                 materials).
NEIGHBOR_OFFSETS_26 — the 26 neighbour offsets the faceted mesher needs.
MeshArrays     — dataclass of positions/normals/uvs/colors/indices arrays
                 (+ face_materials / verts_per_face).
SphereBrush, BoxBrush, CylinderBrush — brush shapes.
BrushMode      — ADD | REMOVE enum.
apply_brush    — the single terrain mutation path.
raycast_voxel  — voxel DDA raycast (click → hit point).
Hit            — raycast result dataclass.
ChunkManager   — streaming store, chunk_provider, and Saveable("terrain").
RainCoverField — top-down highest-solid-voxel cover heightmap (M6 rain cull).

See ``docs/systems/terrain.md`` for the full contract (padding rule,
light_sampler contract, chunk_provider contract, Saveable delta format).
"""

from fire_engine.world.terrain.brush import (
    BoxBrush,
    BrushMode,
    CylinderBrush,
    SphereBrush,
    apply_brush,
)
from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.generation import (
    MATERIAL_DIRT,
    MATERIAL_GRASS,
    generate_chunk,
    surface_height,
)
from fire_engine.world.terrain.meshing import WORLD_FLOOR_SOLID, MeshArrays, build_mesh
from fire_engine.world.terrain.rain_cover import OPEN_SKY_Z, RainCoverField
from fire_engine.world.terrain.raycast import Hit, raycast_voxel
from fire_engine.world.terrain.surface_nets import NEIGHBOR_OFFSETS_26, build_mesh_faceted

__all__ = [
    "MATERIAL_DIRT",
    "MATERIAL_GRASS",
    "NEIGHBOR_OFFSETS_26",
    "OPEN_SKY_Z",
    "WORLD_FLOOR_SOLID",
    "BoxBrush",
    "BrushMode",
    "Chunk",
    "ChunkManager",
    "CylinderBrush",
    "Hit",
    "MeshArrays",
    "RainCoverField",
    "SphereBrush",
    "apply_brush",
    "build_mesh",
    "build_mesh_faceted",
    "generate_chunk",
    "raycast_voxel",
    "surface_height",
]
