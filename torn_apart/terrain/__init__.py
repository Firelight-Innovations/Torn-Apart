"""
torn_apart.terrain — Voxel terrain: chunks, generation, meshing, brushes,
raycasting, and streaming.

Layer 2 (Structure).  Pure Python/numpy — **no panda3d imports** (the mesh
arrays are handed to ``world/geometry_bridge.py`` for upload).  Everything is
deterministic from ``(world_seed, chunk coord)`` and fully headless-testable.

Public API
----------
Chunk          — 32³ uint8 material array + dirty/edited flags + world origin.
generate_chunk — pure-function chunk generation (heightmap + 3-D carve).
build_mesh     — culled-face vectorised mesher → MeshArrays.
MeshArrays     — dataclass of positions/normals/uvs/colors/indices arrays.
SphereBrush, BoxBrush, CylinderBrush — brush shapes.
BrushMode      — ADD | REMOVE enum.
apply_brush    — the single terrain mutation path.
raycast_voxel  — voxel DDA raycast (click → hit point).
Hit            — raycast result dataclass.
ChunkManager   — streaming store, chunk_provider, and Saveable("terrain").

See ``docs/systems/terrain.md`` for the full contract (padding rule,
light_sampler contract, chunk_provider contract, Saveable delta format).
"""

from torn_apart.terrain.chunk import Chunk
from torn_apart.terrain.generation import generate_chunk, surface_height
from torn_apart.terrain.meshing import build_mesh, MeshArrays, WORLD_FLOOR_SOLID
from torn_apart.terrain.brush import (
    SphereBrush,
    BoxBrush,
    CylinderBrush,
    BrushMode,
    apply_brush,
)
from torn_apart.terrain.raycast import raycast_voxel, Hit
from torn_apart.terrain.chunk_manager import ChunkManager

__all__ = [
    "Chunk",
    "generate_chunk",
    "surface_height",
    "build_mesh",
    "MeshArrays",
    "WORLD_FLOOR_SOLID",
    "SphereBrush",
    "BoxBrush",
    "CylinderBrush",
    "BrushMode",
    "apply_brush",
    "raycast_voxel",
    "Hit",
    "ChunkManager",
]
