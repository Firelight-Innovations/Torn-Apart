import math
import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import os

import main as demo
from fire_engine.core.math3d import Quat, Vec3

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
cam = app.camera_go.transform
cam.local_rotation = Quat.from_axis_angle(Vec3.RIGHT, math.radians(-35.0)).normalized()
y = float(cam.position.y)
for i in range(700):
    y += 1.2
    cam.position = Vec3(0.0, y, 10.0)
    app.taskMgr.step()
    app._clock.game_time_of_day = hold
cam.position = Vec3(0.0, y, 10.0)
for _ in range(20):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold

cm = app.chunk_manager
print("cam_y=%.0f loaded chunks=%d" % (y, len(cm.chunks)))
ks = [k for k in cm.chunks if k[2] == 0]
ys = sorted(set(k[1] for k in ks))
print("chunk cy values:", ys, " => world y", [round(v * 16, 0) for v in (min(ys), max(ys))])
# Is there terrain in front of camera? Camera looks -Y... no, forward at yaw0 = +Y. pitch -35 down.
# Forward ground intersection: from z=10 going +Y and down. ground z=8.
# We need chunks at y > cam_y (in front).
front = [k for k in ks if k[1] * 16 > y]
behind = [k for k in ks if (k[1] + 1) * 16 < y]
print("chunks fully in FRONT (cy*16>cam_y):", len(front))
print("chunks fully BEHIND:", len(behind))
# Terrain root children / geom nodes
tr = app.terrain_root
print("terrain_root children:", tr.get_num_children())
# bounds of terrain root
b = tr.get_tight_bounds()
if b:
    print(
        "terrain_root tight bounds:", [round(v, 0) for v in b[0]], "->", [round(v, 0) for v in b[1]]
    )
# how many GeomNodes
gn = tr.find_all_matches("**/+GeomNode")
print("GeomNodes under terrain_root:", gn.get_num_paths())
sys.stdout.flush()
os._exit(0)
