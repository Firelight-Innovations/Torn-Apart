import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import os

import numpy as np

import main as demo
from fire_engine.core.math3d import Vec3

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
cam = app.camera_go.transform
y = float(cam.position.y)
for i in range(700):
    y += 1.2
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
for _ in range(20):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
cm = app.chunk_manager
# inspect a far chunk's materials and re-mesh it directly
for coord in [(0, 51, 0), (0, 0, 0)]:
    ch = cm.chunks.get(coord)
    if ch is None:
        print(coord, "NOT LOADED")
        continue
    mats = ch.materials
    solid = int((mats > 0).sum())
    # find surface: any air-solid boundary in z
    print(
        f"{coord}: solid_voxels={solid}/{mats.size} zmaxsolid={np.argwhere(mats > 0)[:, 2].max() if solid else None} zminsolid={np.argwhere(mats > 0)[:, 2].min() if solid else None}"
    )
    # Try meshing directly
    try:
        from fire_engine.world.terrain.meshing import build_mesh
    except Exception:
        import fire_engine.world.terrain as T

        build_mesh = getattr(T, "build_mesh", None)
    if build_mesh:
        try:
            # build_mesh signature varies; try with chunk + neighbors
            import inspect

            print("  build_mesh sig:", str(inspect.signature(build_mesh)))
        except Exception as e:
            print("  sig err", e)
print("\nchunk_manager stream debug: any mesh-budget/skip far chunks?")
print(" pending_meshes:", len(cm.pending_meshes))
sys.stdout.flush()
os._exit(0)
