"""Headless diagnosis: GI room cascade occupancy + downsample collapse."""

import numpy as np

from fire_engine.core.config import load_config
from fire_engine.core.math3d import Vec3
from fire_engine.lighting.palette import build_default_palette
from fire_engine.lighting.volume import VolumeWindow, assemble_geometry
from fire_engine.world.terrain import BoxBrush, BrushMode, apply_brush
from fire_engine.world.terrain.chunk import Chunk

cfg = load_config("config.toml")
print("voxel_size", cfg.voxel_size, "chunk_size", cfg.chunk_size)
print(
    "c0",
    cfg.light_c0_cells,
    cfg.light_c0_cell_m,
    "c1",
    cfg.light_c1_cells,
    cfg.light_c1_cell_m,
    "c2",
    cfg.light_c2_cells,
    cfg.light_c2_cell_m,
)

# Build the room geometry into a chunk dict (mirror build_gi_test_room geometry).
chunks = {}


def provider(coord):
    if coord not in chunks:
        chunks[coord] = Chunk(coord, chunk_size=cfg.chunk_size, voxel_size=cfg.voxel_size)
    return chunks[coord]


class Bus:
    def publish(self, *a, **k):
        pass


bus = Bus()

cx, cy, z0 = 0.0, -6.0, 8.5
cz = z0 + 2.25
W, R, G, GL = 200, 201, 202, 203


def box(half, at, mode, material=1):
    apply_brush(
        BoxBrush(half_extents_m=Vec3(*half)),
        Vec3(*at),
        mode,
        material=material,
        chunk_provider=provider,
        bus=bus,
    )


box((5.5, 5.5, 3.25), (cx, cy, z0 + 2.25), BrushMode.ADD, W)
box((4.5, 4.5, 2.25), (cx, cy, cz), BrushMode.REMOVE)
box((0.5, 4.5, 2.25), (cx, cy - 5.0, cz), BrushMode.ADD, R)
box((0.5, 4.5, 2.25), (cx, cy + 5.0, cz), BrushMode.ADD, G)
box((1.5, 1.5, 0.5), (cx, cy, z0 + 4.75), BrushMode.ADD, GL)
box(
    (0.75, 1.25, 1.5), (cx - 5.0, cy, z0 + 1.5), BrushMode.REMOVE
)  # doorway (along_x False default? camera fwd +Y)
box((0.75, 0.75, 1.0), (cx + 2.8, cy + 2.8, z0 + 5.0), BrushMode.REMOVE)

pal = build_default_palette()

# Camera "inside" pos from screenshot.py:
cam = (cx, cy - 4.2, z0 + 2.4)  # (0, -10.2, 10.9)
print("\ncamera(inside)=", cam, " room interior center=", (cx, cy, cz))


def report(cells, cell_m, name, campos):
    win = VolumeWindow(cells=cells, cell_m=cell_m)
    win.recenter(campos)
    vol = assemble_geometry(win, chunks, pal, chunk_size=cfg.chunk_size, voxel_size=cfg.voxel_size)
    occ = vol.albedo_occ[..., 3]
    ox, oy, oz = win.origin_cell
    # interior sample points (world m) inside the empty room volume
    pts = [
        (cx, cy, cz),
        (cx, cy - 2, cz),
        (cx, cy + 2, cz),
        (cx - 2, cy, cz),
        (cx + 2, cy, cz),
        (cx, cy, cz - 1.5),
        (cx, cy, cz + 1.5),
    ]
    print(
        f"\n[{name}] cells={cells} cell_m={cell_m} origin_cell={win.origin_cell} "
        f"solid_frac={occ.mean() / 255:.3f}"
    )
    for p in pts:
        i = int(np.floor(p[0] / cell_m)) - ox
        j = int(np.floor(p[1] / cell_m)) - oy
        k = int(np.floor(p[2] / cell_m)) - oz
        if 0 <= i < cells and 0 <= j < cells and 0 <= k < cells:
            s = occ[i, j, k]
            print(f"   world{p} -> cell({i},{j},{k}) occ={'SOLID' if s > 127 else 'air'}")
        else:
            print(f"   world{p} -> OUTSIDE window")


report(cfg.light_c0_cells, cfg.light_c0_cell_m, "C0", cam)
report(cfg.light_c1_cells, cfg.light_c1_cell_m, "C1", cam)
report(cfg.light_c2_cells, cfg.light_c2_cell_m, "C2", cam)

# Now simulate camera backed AWAY so room leaves c0 (c0 box=48m). Move 40m away.
print("\n\n===== camera backed away 40m (room should leave c0) =====")
camfar = (cx, cy - 40.0, z0 + 2.4)
report(cfg.light_c0_cells, cfg.light_c0_cell_m, "C0-far", camfar)
report(cfg.light_c1_cells, cfg.light_c1_cell_m, "C1-far", camfar)
